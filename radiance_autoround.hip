// pybind entry for the AutoRound int4 (g128, symmetric) x fp8 W4A8 GEMM on gfx1201.
//
// The kernels themselves are in ar_kernels.h, which is compiled unchanged by the correctness
// harness (ar_harness.hip). That sharing is deliberate: a kernel that is gated in a harness and
// then re-typed into the serving module is a kernel that has not been gated.
#include "radiance_autoround_kernels.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

#define AR_DEC_MAX_N 32768
#define AR_DEC_KS 4
#define AR_DEC_MAX_TM 4                 // covers M <= 64
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
  if (M > 0 && M <= decode_max_m() && M <= DEC_MTILE * AR_DEC_MAX_TM && N <= AR_DEC_MAX_N &&
      (size_t)AR_DEC_KS * M * N * sizeof(float) <= g_partial_bytes && g_partial && g_cnt) {
    const dim3 grid((N + BND - 1) / BND, 1, AR_DEC_KS), block(AR_DEC_DWN * 32);
    const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;    // smallest tile that covers M
#define AR_DEC(TM_)                                                                             \
    hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<AR_DEC_DWN, AR_DEC_KS, TM_>), grid, block, 0,    \
                       st, A, W, S, As, g_partial, g_cnt, C, M, N, K)
    if (tm == 1) AR_DEC(1);
    else if (tm == 2) AR_DEC(2);
    else if (tm == 3) AR_DEC(3);
    else AR_DEC(4);
#undef AR_DEC
    return;
    // If the scratch was never registered the guard above fails and we fall through to prefill,
    // which is correct at any M -- never produce nothing.
  }

  constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
  const dim3 grid((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
  hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, true>), grid, block, 0, st, A, W, S, As, C, M, N, K);
}

PYBIND11_MODULE(radiance_autoround_kernel, m) {
  m.def("launch", &launch, "AutoRound int4 g128-sym x fp8 W4A8 GEMM (gfx1201)");
  m.def("set_decode_scratch", &set_decode_scratch,
        "register the split-K partial buffer (ptr, bytes) and the zeroed block counter");
}
