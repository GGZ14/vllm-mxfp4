// NOTE (2026-08-26): the OPT bits below are now the SHIPPED behaviour -- they were folded
// into ar_kernels.h / radiance_autoround_kernels.h after they were measured. This file is
// kept to re-measure, and it holds a COPY of the kernel, so it can drift from the shipped
// one. If an A/B here shows no difference between a variant and 'shipped', check that the
// copy still matches before believing it.
// Decode-band optimisation variants for the AutoRound int4 W4A8 GEMM, gfx1201.
//
// Structurally identical to ar_int4_fp8_gemm_decode<DWN,DKS,DTM,true> -- same tile, same LDS,
// same IMAJOR fold -- so a timing delta is attributable to the OPT bits alone.
//
// OPT bit 0  BRANCHLESS  clamp the global row/column index instead of predicating the staging
//                        load. The shipped kernel's ISA at DTM=3 is six global loads each alone
//                        in its own s_and_saveexec region with its own s_wait_loadcnt 0x0 -- and
//                        the A staging is NESTED two deep (the `idx <` guard inside the `r < M`
//                        guard). Six serialised memory round trips per slab.
//                        This also repairs the scale hoist: the source issues the group-scale
//                        load before staging deliberately, but the ISA shows
//                        `global_load_d16_b16` followed immediately by `s_wait_loadcnt 0x0`,
//                        so the hoist buys nothing as compiled.
// OPT bit 1  DIRECT      DKS==1 only: accumulate straight into C and skip the split-K partial
//                        buffer, the threadfence, the atomic and the reduction pass. Partial
//                        traffic is DKS*M*N floats, which is constant in the weight stream but
//                        LINEAR IN M -- 11.1 MB against gate_up's 44.6 MB of weights at M=40,
//                        17.8 MB at M=64. That is the term the shipped kernel pays for split-K
//                        parallelism it may not need: gate_up is 136 n-blocks before any split.
#pragma once
#include "../radiance_autoround_kernels.h"

template <int DWN, int DKS, int DTM, int OPT>
__global__ __launch_bounds__(DWN * 32) void ar_decode_opt(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ S, const float *__restrict__ As, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int M, int N, int K) {
  constexpr int DBK = AR_GROUP;
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  constexpr bool BRANCHLESS = OPT & 1, DIRECT = (OPT & 2) && DKS == 1;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  const int n_lane = n0 + wave * 16 + col;
  const int kw = K / 8;
  const int mlast = M - 1, nlast = N - 1;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    float sc;
    if constexpr (BRANCHLESS) {
      const int nl = n_lane < N ? n_lane : nlast;
      sc = __half2float(S[(size_t)s * N + nl]);
    } else {
      sc = (n_lane < N) ? __half2float(S[(size_t)s * N + n_lane]) : 0.f;
    }
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        if constexpr (BRANCHLESS) {
          // Clamp, do not predicate. Rows past M are discarded by the epilogue's `m < M`, and a
          // clamped read returns real activation data, never the 0xFF byte that is e4m3 NaN.
          const int rc = r < mlast ? r : mlast;
          *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
        } else {
          uint4_t v = uint4_t{0, 0, 0, 0};
          if (r < M) v = *(const uint4_t *)(A + (size_t)r * K + k0 + c);
          *(uint4_t *)(&sA[r * DASTR + c]) = v;
        }
      }
    }
#pragma unroll
    for (int off = 0; off < BND * (DBK / 16); off += DNTHREADS) {
      const int idx = off + tid;
      const int r = idx / (DBK / 16), c = idx % (DBK / 16), gn = n0 + r;
      uint2_t wv;
      if constexpr (BRANCHLESS) {
        const int gc = gn < N ? gn : nlast;
        wv = *(const uint2_t *)(W + (size_t)gc * kw + k0 / 8 + c * 2);
      } else {
        wv = uint2_t{0u, 0u};
        if (gn < N) wv = *(const uint2_t *)(W + (size_t)gn * kw + k0 / 8 + c * 2);
      }
      *(uint2_t *)(&sW[r * DWSTR + c * 16]) = ar_unpack8(wv[0]);
      *(uint2_t *)(&sW[r * DWSTR + c * 16 + 8]) = ar_unpack8(wv[1]);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < DTM; ++i) {
      floatx8 t;
#pragma unroll
      for (int e = 0; e < 8; ++e) t[e] = 0.f;
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const int kk = step * 16 + kb8;
        int2_t af, wf;
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[0] = *(const int *)pa; af[1] = *(const int *)(pa + 4);
        const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
        wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
        t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wf, t);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] += sc * t[e];
    }
    __syncthreads();
  }

  if constexpr (DIRECT) {
    // No partial buffer, no threadfence, no atomic, no reduction pass.
    if (n_lane < N) {
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n_lane] = (__bf16)(acc[i][e] * As[m]);
        }
    }
    return;
  }

  if (n_lane < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n_lane] = acc[i][e];
      }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * As[m]);
    }
  }
}
