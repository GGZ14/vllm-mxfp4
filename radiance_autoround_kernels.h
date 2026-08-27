#pragma once
// AutoRound int4 (group-128, symmetric) x fp8-e4m3 activation GEMM for gfx1201, decode band.
//
// FORMAT. AutoRound exports `auto_round:auto_gptq`: qweight [K/8, N] int32 with eight 4-bit codes
// packed along K per int32, scales [K/128, N] fp16, qzeros all 0x77777777. That last fact is the
// whole design. qzeros stores zero-1, so nibble 7 means zero = 8 for EVERY group -- the checkpoint
// is symmetric, and the dequant collapses from (q - z[g])*s[g] to (q - 8)*s[g] with a CONSTANT
// offset.
//
// WHY THE ZERO POINT COSTS NOTHING HERE. A constant offset of 8 means the stored code c in 0..15
// represents the integer (c - 8) in [-8, 7], and every one of those sixteen integers is EXACTLY
// representable in e4m3 -- 3 mantissa bits cover 1.000..1.111 at any exponent, and -8..7 all land
// on normals. So the offset folds into the unpack table and NEVER enters the matmul. There is no
// zero-point correction term and no activation row-sum, which is what a general asymmetric GPTQ
// kernel has to carry. Verified exhaustively in ~/mxfp4_work/ar/lut.py.
//
// WHY THE SCALE IS ALMOST FREE. The MXFP4 kernel folds its per-32 E8M0 scale into the element
// because E8M0 is a power of two. An fp16 group scale cannot be folded that way, so it must be
// applied as a rescale at each group boundary. Two things make that cheap:
//   1. group_size is 128 and the tuned decode slab DBK is ALSO 128, so a group boundary is exactly
//      a slab boundary -- the rescale lands on the __syncthreads() that already exists, and the
//      inner 8-step WMMA loop is untouched.
//   2. In the gfx12 16x16x16 wave32 C layout a lane's N index is (lane & 15), identical across all
//      eight of its accumulator slots. So a lane needs ONE scale value per group, not eight: one
//      fp16 load and 8 v_fmac_f32 per slab, amortised over 8 WMMA. About 1 VALU per WMMA.
//
// WHY THIS IS BANDWIDTH-BOUND ANYWAY. At decode the kernel streams weights and does almost no
// reuse, so what sets the time is bits/weight: 4 + 16/128 = 4.125 here against MXFP4's 4.25. The
// unpack (10 ops per 4 codes) and the rescale both hide under the weight stream -- at 635 GB/s and
// ~2.4 GHz there are ~31 VALU slots per streamed byte and this kernel needs ~5. Do not micro-tune
// the ALU; tune the traffic.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

#define AR_GROUP 128
#define DEC_MTILE 16
#define DEC_PAD 8

// Sixteen e4m3 bytes for (code - 8), split into the negative half (codes 0..7) and the positive
// half (codes 8..15). Constants emitted and checked by lut.py.
#define AR_NEG_LO 0xCACCCED0u
#define AR_NEG_HI 0xB8C0C4C8u
#define AR_POS_LO 0x44403800u
#define AR_POS_HI 0x4E4C4A48u

// Four codes (one per byte) -> four e4m3 bytes.
//
// Both halves of the sixteen-entry table are computed unconditionally with an 8-entry v_perm each
// and blended by a per-byte mask derived from bit3.
//
// Only v_perm selector values 0..7 are used, and that restriction is deliberate. The selector
// space above 7 was measured on gfx1201 (permprobe.hip) and is not the clean zero window the
// obvious optimisation assumes: 13..15 return 0xFF unconditionally and 11 is data dependent. An
// earlier version drove the unwanted half's selector into 8..11 expecting 0x00 and got 0xFF --
// which is e4m3 NaN -- so the whole GEMM produced NaN. A later version built the mask by
// sign-replication and got the byte lanes crossed, which no uniform-code test can see. Hence
// unpacktest.hip, which gates all eight nibbles of random words against the table.
__device__ __forceinline__ unsigned int ar_lut4(unsigned int c4) {
  const unsigned int sel = c4 & 0x07070707u;
  // 0xFF per byte iff the code is >= 8, i.e. iff the value is non-negative. Built with v_perm
  // restricted to selector values 0 and 1, which are the only semantics worth relying on: a
  // 2-entry pool {0x00, 0xFF} indexed by bit3.
  const unsigned int mask =
      __builtin_amdgcn_perm(0u, 0x0000FF00u, (c4 & 0x08080808u) >> 3);
  const unsigned int neg = __builtin_amdgcn_perm(AR_NEG_HI, AR_NEG_LO, sel);
  const unsigned int pos = __builtin_amdgcn_perm(AR_POS_HI, AR_POS_LO, sel);
  return (pos & mask) | (neg & ~mask);
}

// Eight packed codes (one uint32, low nibble = lowest k) -> eight e4m3 bytes in k order.
__device__ __forceinline__ uint2_t ar_unpack8(unsigned int wv) {
  const unsigned int be = ar_lut4(wv & 0x0F0F0F0Fu);          // k = 0,2,4,6
  const unsigned int bo = ar_lut4((wv >> 4) & 0x0F0F0F0Fu);   // k = 1,3,5,7
  return uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                 __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}

// DTM = number of 16-row M-fragments; the launcher picks ceil(M/16), same policy as the MXFP4
// decode kernel (smallest tile that covers M -- a wider tile computes rows nobody asked for).
// ABLATE is a measurement hook, not a feature. Bit 0 skips the int4->e4m3 unpack (writing raw
// code bytes, so the answer is wrong but the memory traffic and WMMA count are unchanged); bit 1
// skips the group scale load and fold; bit 2 skips the LDS round-trip for the WEIGHT -- the global
// read and unpack still happen, but the result stays in a register and feeds the WMMA directly
// instead of going out to sW and back. Together they bound how much any unpack, rescale or
// register-resident-weight optimisation could possibly buy, which is the only honest way to decide
// whether to chase one.
//
// Bit 2 exists because at DTM=1 -- single-stream decode, the common case -- each staged weight
// byte is read exactly ONCE by exactly one wave. There is no reuse, so the LDS write, the barrier
// and the LDS read are pure overhead, and a fragment-order weight layout (as libr4d and the MXFP4
// WPERM path use) would let the global load land straight in the fragment register. This measures
// the ceiling on that before the layout change is built.
template <int DWN, int DKS, int DTM, bool IMAJOR = true, int ABLATE = 0>
__global__ __launch_bounds__(DWN * 32) void ar_int4_fp8_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ S, const float *__restrict__ As, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int M, int N, int K) {
  constexpr int DBK = AR_GROUP;            // one scale group per slab, by construction
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
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

  // The N column this lane's accumulators belong to is fixed for the whole kernel, so the group
  // scale is a single scalar per slab rather than one per accumulator slot.
  const int n_lane = n0 + wave * 16 + col;
  const int kw = K / 8;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  unsigned int wsink = 0u;   // ABLATE & 4 only; keeps the elided weight loads live

  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    // Issue the group-scale load BEFORE staging, not after the barrier. It depends on nothing the
    // staging computes, so hoisting it here gives the whole A+W stage and the __syncthreads() for
    // the load to land instead of stalling at the head of the WMMA run. Costs one live VGPR.
    // Measured worth far more than the single FMA per WMMA that the fold itself costs -- at M=40
    // the ablation attributed 29% of runtime to the scale, and the arithmetic can only account
    // for a few percent of that.
    // CLAMP, NEVER PREDICATE. A bounds-predicated load lands in its own s_and_saveexec region,
    // and the compiler cannot carry a COUNTED s_wait_loadcnt across an exec-mask merge -- so it
    // emits s_wait_loadcnt 0x0 after every one. As predicated code this loop was six serialised
    // memory round trips per slab (1 scale + 2 A + 4 W at DTM=3), with the A staging nested two
    // deep, where one round trip would do. It is also what made the hoist below a no-op: the ISA
    // was `global_load_d16_b16` followed immediately by `s_wait_loadcnt 0x0`, so the scale never
    // actually overlapped the staging it was hoisted above.
    // Clamping is safe: columns past N are dropped by `n_lane < N` in the epilogue and rows past
    // M by `m < M`, and a clamped read returns real finite data -- never the 0xFF byte that is
    // e4m3 NaN. Bit-identical to the predicated form on every gated shape.
    const float sc = (ABLATE & 2)
                         ? 1.f
                         : __half2float(S[(size_t)s * N + (n_lane < N ? n_lane : N - 1)]);
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;          // clamp, see the note on sc above
        *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
    // Weight staging width, in uint32 per thread. 4 bytes was 11% slower than 8; AR_DEC_WU
    // exists to ask whether 16 is better still, since decode sits at ~81% of the DRAM roofline
    // and what is left is memory efficiency rather than ALU.
#ifndef AR_DEC_WU
#define AR_DEC_WU 2
#endif
#pragma unroll
    for (int off = 0; off < BND * (DBK / (8 * AR_DEC_WU)); off += DNTHREADS) {
      const int idx = off + tid;
      const int r = idx / (DBK / (8 * AR_DEC_WU)), c = idx % (DBK / (8 * AR_DEC_WU));
      const int gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;          // clamp, see the note on sc above
      const unsigned int *src = W + (size_t)gc * kw + k0 / 8 + c * AR_DEC_WU;
      unsigned char *dst = &sW[r * DWSTR + c * 8 * AR_DEC_WU];
#pragma unroll
      for (int u = 0; u < AR_DEC_WU; u += 2) {
        const uint2_t wv = *(const uint2_t *)(src + u);
        if constexpr (ABLATE & 4) {
          // Global read and unpack still happen; the result never reaches LDS. XOR-ed into a sink
          // that is consumed below, so the load cannot be dead-code eliminated.
          const uint2_t a0 = ar_unpack8(wv[0]), a1 = ar_unpack8(wv[1]);
          wsink ^= a0[0] ^ a0[1] ^ a1[0] ^ a1[1];
        } else if constexpr (ABLATE & 1) {
          *(uint2_t *)(dst + u * 8) = uint2_t{wv[0], wv[0]};
          *(uint2_t *)(dst + u * 8 + 8) = uint2_t{wv[1], wv[1]};
        } else {
          *(uint2_t *)(dst + u * 8) = ar_unpack8(wv[0]);
          *(uint2_t *)(dst + u * 8 + 8) = ar_unpack8(wv[1]);
        }
      }
    }
    __syncthreads();


    if constexpr (IMAJOR) {
      // One M-fragment at a time. DBK equals the group size here, so the WMMA chain for tile i
      // lives entirely inside this slab and the temp accumulator is ONE tile regardless of DTM.
      // At DTM=1 this is identical to the group-temp form; at DTM=3 it saves 24 VGPRs, which is
      // the difference between 9 and 12 waves/SIMD and shows up as a large win at M=33..48.
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
        for (int e = 0; e < 8; ++e) {
          if constexpr (ABLATE & 2) acc[i][e] += t[e];
          else acc[i][e] += sc * t[e];
        }
      }
      __syncthreads();
      continue;
    }

    // Group temp: one accumulator set per M-tile, folded at the end of the slab.
    floatx8 g[DTM];
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) g[i][e] = 0.f;

#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa;
        af[i][1] = *(const int *)(pa + 4);
      }
      int2_t wfr;
      if constexpr (ABLATE & 4) { wfr[0] = (int)(0x38383838u ^ kk); wfr[1] = wfr[0]; }
      else {
        const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
        wfr[0] = *(const int *)pw; wfr[1] = *(const int *)(pw + 4);
      }
      wf = wfr;
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        g[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, g[i]);
    }

#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] += sc * g[i][e];

    __syncthreads();
  }

  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the
  // threadfence, the atomic and the reduction pass are all pure overhead. That overhead is not
  // small: partial traffic is DKS*M*N floats, constant in the weight stream but LINEAR IN M --
  // 11.1 MB against gate_up's 44.6 MB of weights at M=40, 17.8 MB at M=64. Choosing DKS by shape
  // is the launcher's job (see split_k_for in radiance_autoround.hip); this is the path that
  // makes DKS==1 actually free. Bit-identical to the DKS==1 reduction path, which summed one term.
  if constexpr (DKS == 1) {
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

  if constexpr (ABLATE & 4) {
    if (wsink == 0xFFFFFFFFu) sW[0] = 1;   // never true; consumes the sink
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

  // Fused split-K reduction: the last block to finish this n-range reduces in place, so there is
  // no second launch. Same scheme as the MXFP4 decode kernel.
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

// ------------------------------------------------------------------ prefill path (large M)
//
// Same tile as the MXFP4 folded kernel -- BMF=256 via AR_TM=4, BK=64, PAD=8 -- because that tile
// was won by a correctness-gated sweep over 28 shapes and the taller M tile is what buys A reuse
// per weight read.
//
// THE ONE STRUCTURAL DIFFERENCE. MXFP4 folds its block exponent into the weight byte, so its inner
// loop is pure WMMA with no temp accumulator. An fp16 group scale cannot be folded that way, so a
// rescale is unavoidable. The cheap place to put it falls out of the arithmetic: group_size is 128
// and BK is 64, so a group is EXACTLY TWO SLABS. Accumulate into `tmp` across the pair, then fold
// once with `acc += s * tmp`.
//
// Why not BK=128 to make a group one slab, as the decode kernel does? LDS. At BMF=256 that is
// (256+64)*136 = 42.5 KB and resident blocks drop 2 -> 1, which is exactly the mechanism that cost
// the MXFP4 prefill kernel 34%. Spanning two slabs keeps LDS at (256+64)*72 = 22.5 KB, unchanged.
//
// COST. Both slabs of a group share one scale, so the fold is TM*TN*8 FMA per 8 WMMA k-steps --
// about 1 VALU per WMMA, the same ratio as the decode kernel. The price is registers: `tmp`
// doubles the accumulator set, which is why TN is capped at 2 here (TN=4 would need 256 VGPRs of
// accumulator alone and spills; a spilling variant also depresses clocks for everything measured
// after it).
// Overridable so the M-tile split between waves and per-wave register tiles can be swept without
// editing the kernel. BMF = AR_WM * AR_TM * 16 is what actually matters for A reuse; moving work
// from AR_TM into AR_WM keeps BMF at 256 while shrinking the per-wave accumulator set.
#ifndef AR_TM
#define AR_TM 4
#endif
#ifndef AR_WM
#define AR_WM 4
#endif
#ifndef AR_WN
#define AR_WN 2
#endif
#ifndef AR_BK
#define AR_BK 64
#endif
#define AR_PAD 8
#define AR_ASTR (AR_BK + AR_PAD)
#define AR_NWAVE (AR_WM * AR_WN)
#define AR_NTHREADS (AR_NWAVE * 32)
#define AR_BMF (AR_WM * AR_TM * 16)

// Measurement hook only. The (__bf16) cast costs 5 VALU on gfx1201 -- bfe / or / cmp_u / add3 /
// cndmask -- because it does round-to-nearest-even AND NaN propagation, and gfx1201 has no
// hardware f32->bf16 convert (cvt_pk_bf16_f32 is CDNA-only). AR_BF16_TRUNC replaces it with a
// bare truncation so the delta bounds what any cheaper conversion could buy. It is WRONG to ship:
// truncation roughly doubles the output rounding error.
__device__ __forceinline__ __bf16 ar_f2bf(float x) {
#ifdef AR_BF16_TRUNC
  unsigned int u;
  __builtin_memcpy(&u, &x, 4);
  const unsigned short h = (unsigned short)(u >> 16);
  __bf16 r;
  __builtin_memcpy(&r, &h, 2);
  return r;
#else
  return (__bf16)x;
#endif
}

template <int TN, bool IMAJOR, int ABLATE = 0>
__global__ __launch_bounds__(AR_NTHREADS) void ar_int4_fp8_gemm_prefill(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ S, const float *__restrict__ As, __bf16 *__restrict__ C, int M,
    int N, int K) {
  constexpr int BNF_T = AR_WN * TN * 16;
  __shared__ unsigned char sA[AR_BMF * AR_ASTR];
  __shared__ unsigned char sW[BNF_T * AR_ASTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / AR_WN, wn = wave % AR_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * AR_BMF, n0 = blockIdx.x * BNF_T;
  const int kw = K / 8;

  floatx8 acc[AR_TM][TN], tmp[IMAJOR ? 1 : AR_TM][TN];
#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
  if constexpr (!IMAJOR)
#pragma unroll
    for (int i = 0; i < AR_TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;

  // The N column of accumulator tile j is fixed for the whole kernel, so the group scale is TN
  // scalars per lane per group rather than one per accumulator slot.
  int ncol[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) ncol[j] = n0 + wn * TN * 16 + j * 16 + col;

  for (int k0 = 0; k0 < K; k0 += AR_BK) {
    // Wave-uniform 64-bit bases so the compiler emits the SADDR form of global_load and a lane
    // pays one 32-bit add instead of a 64-bit add pair. r*K fits an int: r < 256, K <= 17408.
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    const unsigned int *__restrict__ Wb = W + (size_t)n0 * kw + k0 / 8;
    // Group scale for this slab, issued before staging so the load overlaps it. Both slabs of a
    // group carry the SAME scale, so folding once per slab is arithmetically identical to folding
    // once per group -- s*(P1+P2) either way. That equivalence is what lets IMAJOR keep only TN
    // temp tiles instead of AR_TM*TN.
    float sc[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int g = k0 / AR_GROUP;
      // Clamped, not predicated -- see the note in the decode kernel. As predicated code this
      // slab's seven staging loads (2 scale + 4 A + 1 W) each sat alone in an s_and_saveexec
      // region with its own s_wait_loadcnt 0x0: seven serialised memory round trips per k-slab.
      // Clamped, they issue back to back under a single wait.
      sc[j] = (ABLATE & 2)
                  ? 1.f
                  : __half2float(S[(size_t)g * N + (ncol[j] < N ? ncol[j] : N - 1)]);
    }
#pragma unroll
    for (int off = 0; off < AR_BMF * AR_BK; off += AR_NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / AR_BK, c = idx % AR_BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(uint4_t *)(&sA[r * AR_ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    // Eight bytes of codes per thread, as in the decode kernel -- see the note there.
#pragma unroll
    for (int off = 0; off < BNF_T * (AR_BK / 16); off += AR_NTHREADS) {
      const int idx = off + tid;
      const int r = idx / (AR_BK / 16), c = idx % (AR_BK / 16);
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;              // clamp, never predicate
      const uint2_t wv = *(const uint2_t *)(Wb + (size_t)rc * kw + c * 2);
      if constexpr (ABLATE & 1) {
        *(uint2_t *)(&sW[r * AR_ASTR + c * 16]) = uint2_t{wv[0], wv[0]};
        *(uint2_t *)(&sW[r * AR_ASTR + c * 16 + 8]) = uint2_t{wv[1], wv[1]};
      } else {
        *(uint2_t *)(&sW[r * AR_ASTR + c * 16]) = ar_unpack8(wv[0]);
        *(uint2_t *)(&sW[r * AR_ASTR + c * 16 + 8]) = ar_unpack8(wv[1]);
      }
    }
    __syncthreads();

    if constexpr (IMAJOR) {
      // One M-fragment at a time: the WMMA chain for tile (i,j) lives entirely inside this slab,
      // so the temp accumulator is TN tiles rather than AR_TM*TN. That is 48 fewer VGPRs at
      // TM=4/TN=2, which is the difference between 8 and 10 waves/SIMD. The price is re-reading
      // the sW fragments once per i instead of once per slab.
      // The W fragments do not depend on i, so read them ONCE per slab rather than once per
      // M-fragment. As written before, the ISA re-issued the same eight ds_load_2addr_b64
      // (offset pairs {0,144},{2,146},{4,148},{6,150}) four times over: 32 of the slab's 32 LDS
      // instructions where 20 suffice. Costs 6 VGPRs (122 -> 128) and does not move occupancy.
      int2_t wfa[AR_BK / 16][TN];
#pragma unroll
      for (int step = 0; step < AR_BK / 16; ++step)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const unsigned char *p =
              &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + step * 16 + kb8];
          wfa[step][j][0] = *(const int *)p; wfa[step][j][1] = *(const int *)(p + 4);
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
      __syncthreads();
      continue;
    }

#pragma unroll
    for (int step = 0; step < AR_BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[AR_TM], wf[TN];
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        const unsigned char *p = &sA[(wm * AR_TM * 16 + i * 16 + col) * AR_ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Hard scheduling barrier per k-step, as in the MXFP4 kernel: without it the compiler
      // hoists every step's fragment loads above the first WMMA and the register shuffling that
      // holds four k-steps live at once swamps the matmul run.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < AR_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
    }
    __syncthreads();

    // Non-IMAJOR: the temp spans the whole group, so fold at the end of the second slab.
    if (((k0 / AR_BK) & 1) == 1) {
#pragma unroll
      for (int i = 0; i < AR_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[i][j][e] += sc[j] * tmp[i][j][e];
            tmp[i][j][e] = 0.f;
          }
    }
  }

  // Epilogue. Two things matter here and neither is arithmetic.
  //
  // ADDRESSES. `C[(size_t)m * N + ncol[j]]` is a full 64-bit address per element, and there are
  // AR_TM*TN*8 of them -- the ISA showed 142 v_add_co_u32/v_add_co_ci pairs and 132 64-bit shifts
  // in a kernel with only 32 WMMA. Hoisting a wave-uniform base and indexing it with a 32-bit
  // offset lets the compiler keep the base in SGPRs and emit the SADDR form, so a lane pays one
  // 32-bit add instead of a 64-bit add pair. Same trick the MXFP4 kernel uses for its staging
  // bases, where it was worth 1.8-3.4%. The offset cannot overflow an int: it is at most
  // (kb8 + (AR_TM-1)*16 + 7) * N + N <= 64 * 34816.
  //
  // PREDICATION. The bounds test put every one of those stores in its own s_and_saveexec region
  // (95 of them). A block that lies entirely inside M and N needs no test at all, and on a real
  // prefill nearly every block does -- only the last row-block and last column-block are ragged.
  __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * AR_TM * 16) * N;
  const float *__restrict__ Asb = As + m0 + wm * AR_TM * 16;
  const bool full = (m0 + wm * AR_TM * 16 + (AR_TM - 1) * 16 + kb8 + 7 < M) &&
                    (ncol[TN - 1] < N);
  if (full) {
#pragma unroll
    for (int i = 0; i < AR_TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = i * 16 + kb8 + e;
          Cb[(size_t)(r * N + ncol[j])] = ar_f2bf(acc[i][j][e] * Asb[r]);
        }
    return;
  }
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


