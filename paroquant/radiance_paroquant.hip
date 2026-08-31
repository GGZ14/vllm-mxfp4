// pybind entry for the ParoQuant int4 (g128, asymmetric, pairwise-rotated) x fp8 W4A8 stack on
// gfx1201.
//
// The kernels live in par_kernels.h, compiled unchanged by par_harness.hip -- same discipline as
// the AutoRound module: a kernel gated in a harness and then re-typed into the serving module is a
// kernel that has not been gated.
#include "par_kernels.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

#define PQ_DEC_MAX_N 32768
#define PQ_DEC_KS 4                     // scratch is still SIZED for the widest split
#define PQ_DEC_MAX_TM 4                 // covers M <= 64
#define PQ_DEC_DWN 8

static float *g_partial = nullptr;
static size_t g_partial_bytes = 0;
static int *g_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_partial = reinterpret_cast<float *>(ptr);
  g_partial_bytes = bytes;
  g_cnt = reinterpret_cast<int *>(cnt);
}

// Split-K policy copied from the AutoRound launcher; the shapes are the same family and the
// partial-buffer economics (linear in M) are unchanged by the PARO deltas.
static int split_k_for(int nblk, int M) {
  if (const char *e = getenv("RADIANCE_PQ_DECODE_KS")) return atoi(e);
  int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : PQ_DEC_KS;
  if (M > 48 && ks > 1) ks >>= 1;
  return ks;
}

static int decode_max_m() {
  static const int v = [] {
    const char *e = getenv("RADIANCE_PQ_DECODE_MAX_M");
    return e ? atoi(e) : 64;
  }();
  return v;
}

// Fused rotate + channel-scale + per-group fp8 quantize + code-domain row-sums (mode 0), or
// prefill pass A: rotate + write bf16 XR + per-group scales only (mode 1).
// X [M, K] bf16; T [P, krot, K/2, 4] u16; CS [P, K] f16
// -> A [P, M, K] e4m3 (mode 0) / XR bf16 via the same pointer (mode 1),
//    ASG [P, M, K/128] f32, RS [P, M, K/128] f32 (mode 0 only; carries rowsum*asg)
static void launch_rotate_quant(uintptr_t x, uintptr_t t, uintptr_t cs, uintptr_t a,
                                uintptr_t asg, uintptr_t rs, int M, int K, int P, int krot,
                                int mode, uintptr_t stream) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const dim3 grid(K / PQ_GROUP, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P);
  const dim3 block(PQ_ROT_WAVES * 32);
  auto st = reinterpret_cast<hipStream_t>(stream);
  if (mode == 1)
    hipLaunchKernelGGL(pq_rotate_quant<true>, grid, block, 0, st,
                       reinterpret_cast<const __bf16 *>(x),
                       reinterpret_cast<const unsigned short *>(t),
                       reinterpret_cast<const __half *>(cs),
                       reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(asg),
                       reinterpret_cast<float *>(rs), M, K, krot);
  else
    hipLaunchKernelGGL(pq_rotate_quant<false>, grid, block, 0, st,
                       reinterpret_cast<const __bf16 *>(x),
                       reinterpret_cast<const unsigned short *>(t),
                       reinterpret_cast<const __half *>(cs),
                       reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(asg),
                       reinterpret_cast<float *>(rs), M, K, krot);
}

// Prefill pass C: per-token scale + encode + plain code row-sums.
// XR [P, M, K] bf16; ASG [P, M, K/128] f32 -> A [P, M, K] e4m3, AS [P, M] f32, RS [P, M, K/128]
static void launch_token_quant(uintptr_t xr, uintptr_t asg, uintptr_t a, uintptr_t as,
                               uintptr_t rs, int M, int K, int P, uintptr_t stream) {
  const dim3 grid((M + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, P);
  hipLaunchKernelGGL(pq_token_quant, grid, dim3(PQ_ROT_WAVES * 32), 0,
                     reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const __bf16 *>(xr), reinterpret_cast<const float *>(asg),
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as),
                     reinterpret_cast<float *>(rs), M, K);
}

// A [P, M, K] e4m3   W [N, K/8] u32   SZ [K/128, N, 2] f16 {scale, zscale}
// ASG/RS [P, M, K/128] f32   C [M, N] bf16   pb1/pb2 partition boundary columns (INT_MAX unused)
static void launch_gemm(uintptr_t a, uintptr_t w, uintptr_t sz, uintptr_t asg, uintptr_t rs,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, int ptok,
                        uintptr_t stream) {
  if (K % PQ_GROUP)
    throw std::runtime_error("ParoQuant kernel: K must be a multiple of the 128 group size");

  const auto *A = reinterpret_cast<const unsigned char *>(a);
  const auto *W = reinterpret_cast<const unsigned int *>(w);
  const auto *SZ = reinterpret_cast<const __half *>(sz);
  const auto *ASG = reinterpret_cast<const float *>(asg);
  const auto *RS = reinterpret_cast<const float *>(rs);
  auto *C = reinterpret_cast<__bf16 *>(c);
  auto st = reinterpret_cast<hipStream_t>(stream);

  constexpr int BND = PQ_DEC_DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  const int ks = split_k_for(nblk, M);
  const bool have_scratch =
      g_partial && g_cnt && (size_t)ks * M * N * sizeof(float) <= g_partial_bytes;
  // ptok tensors (per-token As, plain row-sums) only fit the prefill kernel's PTOK template;
  // the decode band expects per-group ASG/RS, so a ptok call always takes the prefill path.
  if (!ptok && M > 0 && M <= decode_max_m() && M <= DEC_MTILE * PQ_DEC_MAX_TM &&
      N <= PQ_DEC_MAX_N && (ks == 1 || have_scratch)) {
    const dim3 grid(nblk, 1, ks), block(PQ_DEC_DWN * 32);
    const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;    // smallest tile that covers M
#define PQ_DEC(KS_, TM_)                                                                        \
    hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<PQ_DEC_DWN, KS_, TM_>), grid, block, 0,          \
                       st, A, W, SZ, ASG, RS, g_partial, g_cnt, C, M, N, K, pb1, pb2)
#define PQ_DEC_TM(KS_)                                                                          \
    do { if (tm == 1) PQ_DEC(KS_, 1); else if (tm == 2) PQ_DEC(KS_, 2);                          \
         else if (tm == 3) PQ_DEC(KS_, 3); else PQ_DEC(KS_, 4); } while (0)
    if (ks == 1) PQ_DEC_TM(1);
    else if (ks == 2) PQ_DEC_TM(2);
    else PQ_DEC_TM(4);
#undef PQ_DEC_TM
#undef PQ_DEC
    return;
    // If the scratch was never registered the guard above fails and we fall through to prefill,
    // which is correct at any M -- never produce nothing.
  }

  constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
  const dim3 grid((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
  if (ptok)
    hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, true>), grid, block, 0, st, A, W,
                       SZ, ASG, RS, C, M, N, K, pb1, pb2);
  else
    hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true>), grid, block, 0, st, A, W, SZ, ASG,
                       RS, C, M, N, K, pb1, pb2);
}

PYBIND11_MODULE(radiance_paroquant_kernel, m) {
  m.def("launch_token_quant", &launch_token_quant,
        "prefill pass C: per-token scale + e4m3 encode + code row-sums (gfx1201)");
  m.def("launch_rotate_quant", &launch_rotate_quant,
        "fused pairwise-rotation + channel-scale + per-group fp8 quant + row-sums (gfx1201)");
  m.def("launch_gemm", &launch_gemm,
        "ParoQuant int4 g128-asym x fp8 W4A8 GEMM with partition select (gfx1201)");
  m.def("set_decode_scratch", &set_decode_scratch,
        "register the split-K partial buffer (ptr, bytes) and the zeroed block counter");
}
