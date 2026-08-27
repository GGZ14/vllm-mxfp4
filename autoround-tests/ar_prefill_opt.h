// NOTE (2026-08-26): the OPT bits below are now the SHIPPED behaviour -- they were folded
// into ar_kernels.h / radiance_autoround_kernels.h after they were measured. This file is
// kept to re-measure, and it holds a COPY of the kernel, so it can drift from the shipped
// one. If an A/B here shows no difference between a variant and 'shipped', check that the
// copy still matches before believing it.
// Prefill-kernel optimisation variants for the AutoRound int4 W4A8 GEMM, gfx1201.
//
// Structurally identical to ar_int4_fp8_gemm_prefill<TN, true> in ar_kernels.h -- same tile
// (BMF=256, BK=64, PAD=8), same IMAJOR fold, same LDS footprint -- so any timing delta is
// attributable to the OPT bits and nothing else.
//
// OPT bit 0  BRANCHLESS  staging clamps the global row/col index instead of predicating the load.
//                        The shipped kernel wraps each staging load in s_and_saveexec/execz, and
//                        the compiler cannot carry a counted s_wait_loadcnt across an exec-mask
//                        merge -- so the ISA is load -> s_wait_loadcnt 0x0 -> ds_store, SEVEN
//                        times per k-slab (2 scale + 4 A + 1 W). Seven serialised memory round
//                        trips where one would do. Clamping is safe because rows >= M and columns
//                        >= N are already discarded by the epilogue's `m < M && ncol[j] < N`, and
//                        a clamped read returns real (finite) weight/activation data, never the
//                        0xFF byte that is e4m3 NaN.
// OPT bit 1  WHOIST      hoist the eight W fragments out of the i loop. IMAJOR re-reads them once
//                        per M-fragment: the ISA shows offset pairs {0,144},{2,146},{4,148},
//                        {6,150} repeated four times. Costs 16 VGPRs, removes 12 of 32 LDS
//                        instructions per slab.
// OPT bit 2  PREFETCH    issue the next slab's global loads immediately after the first barrier,
//                        so their latency lands under this slab's WMMA run instead of between
//                        runs. Costs ASTEPS*4 + WSTEPS*2 + TN VGPRs.
// OPT bit 3  SETPRIO     raise wave priority across the WMMA run.
#pragma once
#include "../radiance_autoround_kernels.h"

// WN (waves along N) is a template parameter so a single translation unit can hold several tile
// shapes and interleave them in one timing loop. Comparing tiles across separate binaries is not
// sound on this box: the fixed MXFP4 reference moved 35% between builds, so build-to-build machine
// state swamps the tile effect.
template <int TN, int OPT, int WN = AR_WN>
__global__ __launch_bounds__(WN * AR_WM * 32) void ar_prefill_opt(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ S, const float *__restrict__ As, __bf16 *__restrict__ C, int M,
    int N, int K) {
  constexpr int NTHR = WN * AR_WM * 32;
  constexpr int BNF_T = WN * TN * 16;
  constexpr int ASTEPS = (AR_BMF * AR_BK) / (NTHR * 16);
  constexpr int WSTEPS = BNF_T * (AR_BK / 16) / NTHR;
  constexpr bool BRANCHLESS = OPT & 1, WHOIST = OPT & 2, PREFETCH = OPT & 4, SETPRIO = OPT & 8;
  __shared__ unsigned char sA[AR_BMF * AR_ASTR];
  __shared__ unsigned char sW[BNF_T * AR_ASTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * AR_BMF, n0 = blockIdx.x * BNF_T;
  const int kw = K / 8;
  const int mlast = M - 1 - m0, nlast = N - 1 - n0;   // clamped local indices, >= 0

  floatx8 acc[AR_TM][TN];
#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  int ncol[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) ncol[j] = n0 + wn * TN * 16 + j * 16 + col;

  uint4_t abuf[ASTEPS];
  uint2_t wbuf[WSTEPS];
  unsigned short sbuf[TN];

  // Issue the global loads for the slab starting at k0 into registers. Nothing is waited on here;
  // the consumer (the LDS store, or the fp16 convert) is what forces the wait.
#define AR_LOAD_SLAB(k0)                                                                       \
  do {                                                                                         \
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + (k0);                          \
    const unsigned int *__restrict__ Wb = W + (size_t)n0 * kw + (k0) / 8;                      \
    const int g_ = (k0) / AR_GROUP;                                                            \
    _Pragma("unroll")                                                                          \
    for (int j = 0; j < TN; ++j) {                                                             \
      if constexpr (BRANCHLESS) {                                                              \
        const int nc = ncol[j] < N ? ncol[j] : N - 1;                                          \
        sbuf[j] = *(const unsigned short *)&S[(size_t)g_ * N + nc];                            \
      } else {                                                                                 \
        sbuf[j] = (ncol[j] < N) ? *(const unsigned short *)&S[(size_t)g_ * N + ncol[j]] : 0;    \
      }                                                                                        \
    }                                                                                          \
    _Pragma("unroll")                                                                          \
    for (int s = 0; s < ASTEPS; ++s) {                                                       \
      const int idx = s * NTHR * 16 + tid * 16;                                          \
      const int r = idx / AR_BK, c = idx % AR_BK;                                               \
      if constexpr (BRANCHLESS) {                                                               \
        const int rc = r < mlast ? r : mlast;                                                   \
        abuf[s] = *(const uint4_t *)(Ab + (rc * K + c));                                        \
      } else {                                                                                  \
        abuf[s] = uint4_t{0, 0, 0, 0};                                                          \
        if (m0 + r < M) abuf[s] = *(const uint4_t *)(Ab + (r * K + c));                         \
      }                                                                                         \
    }                                                                                           \
    _Pragma("unroll")                                                                           \
    for (int s = 0; s < WSTEPS; ++s) {                                                          \
      const int idx = s * NTHR + tid;                                                    \
      const int r = idx / (AR_BK / 16), c = idx % (AR_BK / 16);                                  \
      if constexpr (BRANCHLESS) {                                                               \
        const int rc = r < nlast ? r : nlast;                                                   \
        wbuf[s] = *(const uint2_t *)(Wb + (size_t)rc * kw + c * 2);                              \
      } else {                                                                                  \
        wbuf[s] = uint2_t{0u, 0u};                                                               \
        if (n0 + r < N) wbuf[s] = *(const uint2_t *)(Wb + (size_t)r * kw + c * 2);               \
      }                                                                                         \
    }                                                                                           \
  } while (0)

#define AR_STORE_SLAB()                                                                        \
  do {                                                                                         \
    _Pragma("unroll")                                                                          \
    for (int s = 0; s < ASTEPS; ++s) {                                                       \
      const int idx = s * NTHR * 16 + tid * 16;                                           \
      const int r = idx / AR_BK, c = idx % AR_BK;                                                \
      *(uint4_t *)(&sA[r * AR_ASTR + c]) = abuf[s];                                              \
    }                                                                                            \
    _Pragma("unroll")                                                                            \
    for (int s = 0; s < WSTEPS; ++s) {                                                           \
      const int idx = s * NTHR + tid;                                                     \
      const int r = idx / (AR_BK / 16), c = idx % (AR_BK / 16);                                   \
      *(uint2_t *)(&sW[r * AR_ASTR + c * 16]) = ar_unpack8(wbuf[s][0]);                           \
      *(uint2_t *)(&sW[r * AR_ASTR + c * 16 + 8]) = ar_unpack8(wbuf[s][1]);                       \
    }                                                                                             \
  } while (0)

  if constexpr (PREFETCH) AR_LOAD_SLAB(0);

  for (int k0 = 0; k0 < K; k0 += AR_BK) {
    if constexpr (!PREFETCH) AR_LOAD_SLAB(k0);
    float sc[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) sc[j] = __half2float(*(const __half *)&sbuf[j]);
    AR_STORE_SLAB();
    __syncthreads();
    if constexpr (PREFETCH) {
      if (k0 + AR_BK < K) AR_LOAD_SLAB(k0 + AR_BK);
    }

    if constexpr (SETPRIO) __builtin_amdgcn_s_setprio(1);
    if constexpr (WHOIST) {
      int2_t wfa[AR_BK / 16][TN];
#pragma unroll
      for (int step = 0; step < AR_BK / 16; ++step)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + step * 16 + kb8];
          wfa[step][j][0] = *(const int *)p;
          wfa[step][j][1] = *(const int *)(p + 4);
        }
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        floatx8 t[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) t[j][e] = 0.f;
#pragma unroll
        for (int step = 0; step < AR_BK / 16; ++step) {
          int2_t af;
          const unsigned char *pa =
              &sA[(wm * AR_TM * 16 + i * 16 + col) * AR_ASTR + step * 16 + kb8];
          af[0] = *(const int *)pa; af[1] = *(const int *)(pa + 4);
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int j = 0; j < TN; ++j)
            t[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wfa[step][j], t[j]);
        }
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += sc[j] * t[j][e];
      }
    } else {
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        floatx8 t[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) t[j][e] = 0.f;
#pragma unroll
        for (int step = 0; step < AR_BK / 16; ++step) {
          const int kk = step * 16 + kb8;
          int2_t af, wf[TN];
          const unsigned char *pa = &sA[(wm * AR_TM * 16 + i * 16 + col) * AR_ASTR + kk];
          af[0] = *(const int *)pa; af[1] = *(const int *)(pa + 4);
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + kk];
            wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
          }
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int j = 0; j < TN; ++j)
            t[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wf[j], t[j]);
        }
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += sc[j] * t[j][e];
      }
    }
    if constexpr (SETPRIO) __builtin_amdgcn_s_setprio(0);
    __syncthreads();
  }
#undef AR_LOAD_SLAB
#undef AR_STORE_SLAB

#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * AR_TM * 16 + i * 16 + kb8 + e;
        if (m < M && ncol[j] < N) C[(size_t)m * N + ncol[j]] = (__bf16)(acc[i][j][e] * As[m]);
      }
}
