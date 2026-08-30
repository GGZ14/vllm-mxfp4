// pybind entry for the AutoRound int4 (g128, symmetric) x fp8 W4A8 GEMM on gfx1201.
//
// The kernels themselves are in ar_kernels.h, which is compiled unchanged by the correctness
// harness (ar_harness.hip). That sharing is deliberate: a kernel that is gated in a harness and
// then re-typed into the serving module is a kernel that has not been gated.
#include "radiance_autoround_kernels.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

#define AR_DEC_MAX_N 32768
#define AR_DEC_KS 4                     // scratch is still SIZED for the widest split
#define AR_DEC_MAX_TM 8                 // covers M <= 128: dflash SPEC=7 x max_num_seqs=16
                                        // reaches batch x 8 = 128. Extended 2026-08-29 in
                                        // lockstep with the MXFP4 kernel; RADIANCE_AR_DECODE_MAX_M
                                        // still gates WHICH M take this kernel (64 by default,
                                        // so the 8-and-under band routes exactly as before), and
                                        // radiance_autoround.py sizes the scratch from that env.
#define AR_DEC_DWN 8

// Split-K partial slab, owned by torch and handed over at weight-load time.
//
// Allocating it here would be wrong twice over, exactly as the MXFP4 module documents: a lazy
// hipMalloc from launch() lands inside CUDA-graph capture whenever the torch.compile cache is warm,
// where it is illegal; and any C++ exception escaping this module is relabelled by quark's
// TileLang pybind exception translator as "libamdhip64.so not found", which points at the wrong
// frame entirely.
static float *g_partial = nullptr;
static size_t g_partial_bytes = 0;
static int *g_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_partial = reinterpret_cast<float *>(ptr);
  g_partial_bytes = bytes;
  g_cnt = reinterpret_cast<int *>(cnt);
}

// How far to split K.
//
// Split-K exists to get enough blocks to fill the machine, and it is not free: the partial buffer
// costs DKS*M*N floats of write-then-read traffic, which is constant in the weight stream but
// LINEAR IN M. On gate_up that is 11.1 MB against 44.6 MB of weights at M=40 and 17.8 MB at M=64,
// so the split that pays for itself at M=5 is a large loss at M=64. It also serialises: the
// reduction runs on the ONE block per n-range that finishes last, while the other DKS-1 have
// exited.
//
// So take the SMALLEST split that still fills the machine, and shrink it further as M grows.
// Measured on an idle GPU: gate_up (N=17408, 136 n-blocks) wants DKS=1 at every M, down and out
// (N=5120, 40 n-blocks) want 4 up to M=40 and 2 at M=64. This rule picks the best measured cell
// in all fifteen shape/M combinations.
static int split_k_for(int nblk, int M, int K) {
  if (const char *e = getenv("RADIANCE_AR_DECODE_KS")) return atoi(e);
  if (M > 64) {
    // The 16-concurrent band, swept cell-by-cell 2026-08-29 (arsweep.hip, idle GPU, DRAM-fed)
    // in lockstep with the MXFP4 kernel's extension; M <= 64 below is untouched and its rule
    // re-verified optimal by the same sweep. Wide shapes never split; the nblk=40 shapes split
    // by 2 until the partial traffic catches up, and at M=128 they diverge by K depth exactly
    // as the MXFP4 band does (down 141.6 us at ks4 vs 158.2 at ks2; out 95.0 at ks1 vs 106.1).
    if (nblk >= 48) return 1;
    if (M <= 112) return 2;
    return (K >= 6144) ? AR_DEC_KS : 1;
  }
  int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : AR_DEC_KS;
  if (M > 48 && ks > 1) ks >>= 1;
  return ks;
}

// N-tile width for prefill.
//
// TN=4 (BNF=128) amortises the A-tile staging across twice as many output columns, and A is the
// bulk of the staging traffic at prefill -- AR_BMF*BK = 16 KB per k-slab against W's 4 KB. But the
// wide tile HALVES the block count in N, so below a few thousand rows it cannot fill the machine
// and loses. Measured on an idle GPU, DRAM-resident, us:
//
//   gate_up  M=512  TN2 510.8  TN4 516.7      down  M=512  TN2 248.6  TN4 288.5
//            M=2048     1931.7      1838.9          M=2048     987.7      955.2
//            M=4096     4023.2      3711.1          M=4096    1984.5     1856.9
//
// So TN=4 from M >= 2048, which is also where the MXFP4 kernel puts its own crossover. Serving
// chunks prefill at --max-num-batched-tokens (8192 here), so the common case is the TN=4 side.
static int tn4_min_m() {
  static const int v = [] {
    const char *e = getenv("RADIANCE_AR_TN4_MIN_M");
    return e ? atoi(e) : 2048;
  }();
  return v;
}

static int decode_max_m() {
  static const int v = [] {
    const char *e = getenv("RADIANCE_AR_DECODE_MAX_M");
    return e ? atoi(e) : 64;
  }();
  return v;
}

// A: [M, K] e4m3   W: [N, K/8] uint32 (8 codes per word along K)
// S: [K/128, N] fp16 group scales   As: [M] fp32 per-token activation scale
// C: [M, N] bf16
static void launch(uintptr_t a, uintptr_t w, uintptr_t s, uintptr_t as, uintptr_t c, int M, int N,
                   int K, uintptr_t stream) {
  if (K % AR_GROUP)
    throw std::runtime_error("AutoRound kernel: K must be a multiple of the 128 group size");

  const auto *A = reinterpret_cast<const unsigned char *>(a);
  const auto *W = reinterpret_cast<const unsigned int *>(w);
  const auto *S = reinterpret_cast<const __half *>(s);
  const auto *As = reinterpret_cast<const float *>(as);
  auto *C = reinterpret_cast<__bf16 *>(c);
  auto st = reinterpret_cast<hipStream_t>(stream);

  constexpr int BND = AR_DEC_DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  const int ks = split_k_for(nblk, M, K);
  // DKS==1 writes C directly and needs no scratch at all; anything wider does.
  const bool have_scratch =
      g_partial && g_cnt && (size_t)ks * M * N * sizeof(float) <= g_partial_bytes;
  if (M > 0 && M <= decode_max_m() && M <= DEC_MTILE * AR_DEC_MAX_TM && N <= AR_DEC_MAX_N &&
      (ks == 1 || have_scratch)) {
    const dim3 grid(nblk, 1, ks), block(AR_DEC_DWN * 32);
    const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;    // smallest tile that covers M
#define AR_DEC(KS_, TM_)                                                                        \
    hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<AR_DEC_DWN, KS_, TM_>), grid, block, 0,          \
                       st, A, W, S, As, g_partial, g_cnt, C, M, N, K)
#define AR_DEC_TM(KS_)                                                                          \
    do { if (tm == 1) AR_DEC(KS_, 1); else if (tm == 2) AR_DEC(KS_, 2);                          \
         else if (tm == 3) AR_DEC(KS_, 3); else if (tm == 4) AR_DEC(KS_, 4);                     \
         else if (tm == 5) AR_DEC(KS_, 5); else if (tm == 6) AR_DEC(KS_, 6);                     \
         else if (tm == 7) AR_DEC(KS_, 7); else AR_DEC(KS_, 8); } while (0)
    if (ks == 1) AR_DEC_TM(1);
    else if (ks == 2) AR_DEC_TM(2);
    else AR_DEC_TM(4);
#undef AR_DEC_TM
#undef AR_DEC
    return;
    // If the scratch was never registered the guard above fails and we fall through to prefill,
    // which is correct at any M -- never produce nothing.
  }

  const dim3 bl(AR_NTHREADS);
  const int by = (M + AR_BMF - 1) / AR_BMF;
  // TN=4 only when the shape can use the 128-wide tile -- the MXFP4 launcher's N guard, carried
  // over even though no quantized AutoRound layer is currently that narrow (the N=48 gdn
  // in_proj_ba stays bf16 in this checkpoint): a future shape should degrade, not fall over.
  if (M >= tn4_min_m() && N >= AR_WN * 4 * 16) {
    constexpr int BNF_T = AR_WN * 4 * 16;
    hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<4, true>), dim3((N + BNF_T - 1) / BNF_T, by), bl,
                       0, st, A, W, S, As, C, M, N, K);
  } else {
    constexpr int BNF_T = AR_WN * 2 * 16;
    hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<2, true>), dim3((N + BNF_T - 1) / BNF_T, by), bl,
                       0, st, A, W, S, As, C, M, N, K);
  }
}

PYBIND11_MODULE(radiance_autoround_kernel, m) {
  m.def("launch", &launch, "AutoRound int4 g128-sym x fp8 W4A8 GEMM (gfx1201)");
  m.def("set_decode_scratch", &set_decode_scratch,
        "register the split-K partial buffer (ptr, bytes) and the zeroed block counter");
}
