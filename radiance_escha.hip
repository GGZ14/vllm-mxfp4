// pybind entry for the escha (EXL3 trellis) W2 linear on gfx1201.
//
// The kernels are in escha/escha_kernels.h and escha/escha_act.h, included unchanged -- the same
// headers the correctness harnesses compile. A kernel that is gated in a harness and then re-typed
// into the serving module is a kernel that has not been gated.
//
// This module owns the WHOLE layer, not just the GEMM, because the escha forward is three kernels:
//
//     y = Had128( (x * s_in) * rin ) @ decode(code)  ->  Had128  ->  * rout  ->  * s_out
//
// taken from the reference runtime's own serving path. Keeping all three behind one entry point
// means the M-dependent decode/prefill choice is made in C++, where dynamo cannot see it -- the
// MXFP4 path measured a Python-level shape branch at ~30% of decode throughput.
#include "escha/escha_kernels.h"
#include "escha/escha_act.h"

#include <pybind11/pybind11.h>

#define ESCHA_DEC_MAX_TM 4              // covers M <= 64
#define ESCHA_DEC_DWN 8
#define ESCHA_DEC_KS_MAX 8

// Split-K partial slab and block counter, owned by torch and handed over at load time. Allocating
// here would land a lazy hipMalloc inside CUDA-graph capture, which is illegal; and any C++
// exception escaping this module gets relabelled by quark's TileLang pybind translator as
// "libamdhip64.so not found", pointing at entirely the wrong frame.
static float *g_partial = nullptr;
static size_t g_partial_bytes = 0;
static int *g_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_partial = reinterpret_cast<float *>(ptr);
  g_partial_bytes = bytes;
  g_cnt = reinterpret_cast<int *>(cnt);
}

static int prefill_min_m() {
  static const int v = [] {
    const char *e = getenv("RADIANCE_ESCHA_PREFILL_MIN_M");
    return e ? atoi(e) : 65;            // decode kernel covers M <= 64 (DTM <= 4)
  }();
  return v;
}

// Full layer. A/As/C are caller-owned scratch; out may alias nothing.
static void launch(uintptr_t x, uintptr_t code, uintptr_t rin, uintptr_t rout, uintptr_t s_in,
                   uintptr_t s_out, uintptr_t A, uintptr_t As, uintptr_t amax, uintptr_t C,
                   uintptr_t out, int M, int N, int K, int Kbits, int ldo, int col0,
                   uintptr_t stream) {
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  const auto *xp = reinterpret_cast<const __bf16 *>(x);
  const auto *cp = reinterpret_cast<const unsigned int *>(code);
  const auto *rip = reinterpret_cast<const __half *>(rin);
  const auto *rop = reinterpret_cast<const __half *>(rout);
  const auto *sip = reinterpret_cast<const float *>(s_in);
  const auto *sop = reinterpret_cast<const float *>(s_out);
  auto *Ap = reinterpret_cast<unsigned char *>(A);
  auto *Asp = reinterpret_cast<float *>(As);
  auto *amaxp = reinterpret_cast<float *>(amax);
  auto *Cp = reinterpret_cast<__bf16 *>(C);
  auto *op = reinterpret_cast<__bf16 *>(out);

  // Two launches: the per-token amax is a reduction over the whole row, so it must complete
  // before anything can be quantized against it.
  const dim3 preg((K + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M),
                  blk(ESCHA_ACT_THREADS);
  HIP_CHECK(hipMemsetAsync(amaxp, 0, (size_t)M * sizeof(float), st));
  hipLaunchKernelGGL(escha_pre_amax, preg, blk, 0, st, xp, sip, rip, amaxp, M, K);
  hipLaunchKernelGGL(escha_pre_quant, preg, blk, 0, st, xp, sip, rip, amaxp, Ap, Asp, M, K);

  constexpr int DWN = ESCHA_DEC_DWN, BND = DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  const int ks = escha_decode_split_k(nblk, K / 16);
  const bool have_scratch = g_partial && g_cnt &&
      (size_t)ks * M * N * sizeof(float) <= g_partial_bytes;
  if (M > 0 && M < prefill_min_m() && have_scratch &&
      M <= ESCHA_TILE * ESCHA_DEC_MAX_TM) {
    const int tm = (M + ESCHA_TILE - 1) / ESCHA_TILE;
#define ESCHA_DEC(KS_, TM_, KB_)                                                              \
    hipLaunchKernelGGL((escha_gemm_decode_fp8<DWN, KS_, TM_, KB_, 4, 8, false>),               \
                       dim3(nblk, 1, KS_), dim3(DWN * 32), 0, st, Ap, cp, Asp, g_partial,      \
                       g_cnt, Cp, M, N, K)
#define ESCHA_BY_TM(KS_, KB_)                                                                 \
    if (tm == 1)      ESCHA_DEC(KS_, 1, KB_);                                                  \
    else if (tm == 2) ESCHA_DEC(KS_, 2, KB_);                                                  \
    else if (tm == 3) ESCHA_DEC(KS_, 3, KB_);                                                  \
    else              ESCHA_DEC(KS_, 4, KB_)
#define ESCHA_BY_KS(KB_)                                                                      \
    if (ks >= 8)      { ESCHA_BY_TM(8, KB_); }                                                 \
    else if (ks >= 4) { ESCHA_BY_TM(4, KB_); }                                                 \
    else if (ks >= 2) { ESCHA_BY_TM(2, KB_); }                                                 \
    else              { ESCHA_BY_TM(1, KB_); }
    if (Kbits == 2) { ESCHA_BY_KS(2); } else { ESCHA_BY_KS(3); }
#undef ESCHA_DEC
#undef ESCHA_BY_TM
#undef ESCHA_BY_KS
  } else {
    // TM=8 covers 512 rows and cuts codec redundancy in half, but starves the grid when the
    // column blocks alone do not fill the machine; the same rule the benchmark fitted.
    constexpr int EB = EP_WN * 2 * 16;
    const int ncol = (N + EB - 1) / EB;
    const bool wide = (size_t)ncol * ((M + 511) / 512) >= 128;
    const dim3 bl(EP_NTHREADS);
    if (wide) {
      const dim3 g(ncol, (M + 511) / 512);
      if (Kbits == 2) hipLaunchKernelGGL((escha_gemm_prefill<2, 2, 8>), g, bl, 0, st, Ap, cp, Asp, Cp, M, N, K);
      else            hipLaunchKernelGGL((escha_gemm_prefill<2, 3, 8>), g, bl, 0, st, Ap, cp, Asp, Cp, M, N, K);
    } else {
      const dim3 g(ncol, (M + 255) / 256);
      if (Kbits == 2) hipLaunchKernelGGL((escha_gemm_prefill<2, 2, 4>), g, bl, 0, st, Ap, cp, Asp, Cp, M, N, K);
      else            hipLaunchKernelGGL((escha_gemm_prefill<2, 3, 4>), g, bl, 0, st, Ap, cp, Asp, Cp, M, N, K);
    }
  }

  hipLaunchKernelGGL(escha_post_rot,
                     dim3((N + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M),
                     blk, 0, st, Cp, rop, sop, op, M, N, ldo, col0);
}

PYBIND11_MODULE(radiance_escha_kernel, m) {
  m.def("launch", &launch, "escha W2 trellis linear (rotate + W2A8 GEMM + rotate) on gfx1201");
  m.def("set_decode_scratch", &set_decode_scratch,
        "register the split-K partial buffer (ptr, bytes) and the zeroed block counter");
}
