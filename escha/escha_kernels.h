#pragma once
// EXL3 trellis-decode GEMM for gfx1201.  C[M,N] = A[M,K] @ decode(code)[K,N]
//
// The weight format is EXL3 (exllamav3, MIT, (c) Turboderp), as shipped by EschaLabs' W2 build.
// Format spec and the evidence for every constant here: ~/mxfp4_work/escha/FORMAT.md.
//
// WHAT THIS KERNEL DOES *NOT* DO. The Hadamard rotations are not its job. EXL3's forward applies
// them to ACTIVATIONS -- `had_r_128(x, xh, suh, ...)` before the GEMM and `had_r_128(y, y, ...,
// svh)` after -- so they are two M x 128 passes, not a weight transform. Keeping them out means
// this kernel has the same shape as the MXFP4 and int4 W4A8 kernels and can reuse their tiling.
//
// HOW IT DIFFERS FROM THOSE, AND WHY THAT INVERTS THE TUNING. There the codebook was a 16-entry
// v_perm LUT and the kernel was bandwidth-bound at ~95% of the streaming roofline; the ALU was
// nearly free and bytes were everything. Here the codebook is COMPUTED -- one multiply, an
// and-xor, and an fp16 add per weight -- so the arithmetic per byte is far higher while the bytes
// are roughly halved (2.469 bits/weight against MXFP4's 4.25). Expect this to be much closer to
// ALU-bound at decode, which is the opposite regime, so do not assume the MXFP4 tuning transfers.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
// gfx12's f16 WMMA takes 8 halves per lane (16 B), unlike the fp8 pipe's 8 bytes.
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

#define ESCHA_MCG 0xCBAC1FEDu
#ifndef ESCHA_MUL24
#define ESCHA_MUL24 0
#endif
#define ESCHA_TILE 16

// ---------------------------------------------------------------------------------------------
// Codebook (cb = 1, "MCG").  state -> half
//
//   v = state * 0xCBAC1FED;  v = (v & 0x8fff8fff) ^ 0x3b603b60;  value = lo_half(v) + hi_half(v)
//
// The mask keeps each half's sign and low mantissa bits and the xor forces the exponent, so the
// two halves are near-uniform in a fixed binade; their sum is the QTIP "3INST" Gaussian. The CUDA
// source writes the middle step as lop3(a, 0x8fff8fff, 0x3b603b60, 0x6a), and immLut 0x6a is
// exactly (a & b) ^ c -- there is no cheaper identity hiding in it.
//
// cb=0 and cb=2 also exist in EXL3 and are NOT what this checkpoint uses: measured against the
// fp8 base they score 0.005 and 0.002 correlation where cb=1 scores 0.907.
__device__ __forceinline__ __half escha_decode(unsigned int state) {
  unsigned int v = state * ESCHA_MCG;
  v = (v & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const __half lo = __ushort_as_half((unsigned short)(v & 0xFFFFu));
  const __half hi = __ushort_as_half((unsigned short)(v >> 16));
  return __hadd(lo, hi);
}

// Two at once. Each state's two fp16 halves are already packed in one uint32, so pairing the
// states lets the add run as a single packed fp16 op instead of two scalar ones -- the decode is
// ~4 ALU per weight and this is the only one of them that packs.
// Decode two trellis states into one packed half2.
//
// The codec is a horizontal add: each state maps to a dword whose two halves are summed in fp16.
// Written naively that needs the two halves of `a` and the two of `b` shuffled into a (lo,lo) and
// a (hi,hi) pair before v_pk_add_f16 -- which the compiler built out of v_mov_b16 / v_and_or /
// v_lshl_or, 49 instructions per four tiles. Both shuffles are exactly one v_perm_b32.
//
// Selector semantics on gfx1201, established the hard way in the int4 work: sel 0-3 index the
// SECOND source's bytes, 4-7 the FIRST source's. (Selectors >= 8 sign-replicate on this part,
// which is why nothing here goes above 7.)
template <bool MUL24 = (ESCHA_MUL24 != 0)>
__device__ __forceinline__ __half2 escha_decode2(unsigned int s0, unsigned int s1) {
  unsigned int a0, b0;
  if constexpr (MUL24) {
  // v_mul_lo_u32 is quarter-rate. States are 16-bit, so the product splits into two full-rate
  // 24-bit multiplies; whether that wins is a measurement, not an assumption -- see the A/B.
    a0 = __umul24(s0, ESCHA_MCG & 0xFFFFu) +
         (__umul24(s0, ESCHA_MCG >> 16) << 16);
    b0 = __umul24(s1, ESCHA_MCG & 0xFFFFu) +
         (__umul24(s1, ESCHA_MCG >> 16) << 16);
  } else {
    a0 = s0 * ESCHA_MCG;
    b0 = s1 * ESCHA_MCG;
  }
  const unsigned int a = (a0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const unsigned int b = (b0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const unsigned int lo = __builtin_amdgcn_perm(b, a, 0x05040100u);  // {a.lo, b.lo}
  const unsigned int hi = __builtin_amdgcn_perm(b, a, 0x07060302u);  // {a.hi, b.hi}
  __half2 x, y;
  __builtin_memcpy(&x, &lo, 4);
  __builtin_memcpy(&y, &hi, 4);
  return __hadd2(x, y);
}

__device__ __forceinline__ unsigned long long escha_fshift(unsigned int b, unsigned int a, int s) {
  return (((unsigned long long)a << 32) | (unsigned long long)b) >> s;
}

// K = 2: the aligned fast path (dq8_aligned_2bits). 16 uint32 per tile.
__device__ __forceinline__ void escha_states8_k2(const unsigned int *__restrict__ u32, int lane,
                                                 unsigned int out[8]) {
  const int t = lane * 8;
  const int i1 = (t >> 4) & 15;
  const int i0 = (i1 + 15) & 15;
  const unsigned long long b = escha_fshift(u32[i1], u32[i0], ((~t) & 8) << 1);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = (unsigned int)(b >> (14 - 2 * j)) & 0xFFFFu;
}

// K = 3: the generic reader (dq8<bits, cb, align=4>). 24 uint32 per tile.
// The shifts derive from the UNWRAPPED bit indices; only the array access wraps.
// ---------------------------------------------------------------------------------------------
// K-BLOCKED WEIGHT FETCH.
//
// The original loop staged ONE 16-wide k-tile between two __syncthreads(), so each wave had
// exactly one weight load outstanding at a time: memory-level parallelism of 1. Measured, that
// held gate_up decode to 106 GB/s -- 17% of the 635 GB/s roofline -- while the same shape on the
// MXFP4 kernel ran 2x faster on twice the bytes. The trellis math was never the problem; the
// latency simply had nothing to hide behind.
//
// Both K=2 and K=3 need exactly TWO dwords per lane per tile (a funnel shift across a word
// boundary), and consecutive k-tiles are contiguous in the code tensor. So the address math
// splits out, KB tiles' worth of dwords are issued back to back into registers, and the barrier
// pair amortizes over KB tiles instead of one.
template <int K>
__device__ __forceinline__ void escha_widx(int lane, int *ia, int *ib, int *sh) {
  const int t = lane * 8;
  if constexpr (K == 2) {
    const int i1 = (t >> 4) & 15;
    *ia = i1; *ib = (i1 + 15) & 15; *sh = ((~t) & 8) << 1;
  } else {
    constexpr int NW = 256 * K / 32;
    const int b1 = (t + 257) * K, b0 = b1 - 16, b2 = b1 + K * 7;
    const int i2 = (b2 - 1) / 32;
    *ia = i2 % NW; *ib = (b0 / 32) % NW; *sh = (i2 + 1) * 32 - b2;
  }
}

// Same bit extraction as escha_states8_*, but from two dwords already in registers.
//
// For K=2 the eight sliding 16-bit windows sit at bit offsets 14 down to 0, so the highest bit any
// of them needs is 29 -- the whole extraction fits in the LOW 32 bits of the funnel shift. Taking
// that word first lets the shift lower to a single v_alignbit_b32 instead of a 64-bit shift pair.
// K=3 reaches bit 36 and still needs the 64-bit form.
template <int K>
__device__ __forceinline__ void escha_states8_from(unsigned int lo, unsigned int hi, int sh,
                                                   unsigned int out[8]) {
  if constexpr (K == 2) {
    const unsigned int w = __builtin_amdgcn_alignbit(hi, lo, (unsigned int)sh);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (w >> (14 - 2 * j)) & 0xFFFFu;
  } else {
    const unsigned long long m = escha_fshift(lo, hi, sh);
    const unsigned int wa = (unsigned int)m, wb = (unsigned int)(m >> 16);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      out[7 - j] = j < 6 ? ((wa >> (K * j)) & 0xFFFFu) : ((wb >> (K * j - 16)) & 0xFFFFu);
  }
}

__device__ __forceinline__ void escha_states8_k3(const unsigned int *__restrict__ u32, int lane,
                                                 unsigned int out[8]) {
  constexpr int K = 3, NW = 256 * K / 32;
  const int t = lane * 8;
  const int b1 = (t + 257) * K, b0 = b1 - 16, b2 = b1 + K * 7;
  const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
  const int s2 = (i2 + 1) * 32 - b2;
  const unsigned long long m = escha_fshift(u32[i2 % NW], u32[i0 % NW], s2);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[7 - j] = (unsigned int)(m >> (K * j)) & 0xFFFFu;
}

template <int K>
__device__ __forceinline__ void escha_states8(const unsigned int *__restrict__ u32, int lane,
                                              unsigned int out[8]) {
  if constexpr (K == 2) escha_states8_k2(u32, lane, out);
  else escha_states8_k3(u32, lane, out);
}

// Where lane `lane`'s symbol j lands in the 16x16 tile (row = K index, col = N index).
// This is tensor_core_perm from exl3_lib/quantize.py, evaluated per lane instead of tabulated.
__device__ __forceinline__ void escha_tile_pos(int lane, int j, int *row, int *col) {
  const int r0 = (lane % 4) * 2;
  const int rr[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
  *row = rr[j & 3];
  *col = lane / 4 + ((j >= 4) ? 8 : 0);
}

// ---------------------------------------------------------------------------------------------
// Decode GEMM, M <= 64.   C[M,N] = Ah[M,K] @ Wdec[K,N]
//
// Ah is fp16 and ALREADY rotated by the caller (had_r_128 with suh). The N-side rotation and svh
// are applied afterwards on the M x N output, so they are not this kernel's business.
//
// WHY f16 AND NOT fp8. The codebook emits fp16, and gfx1201 has no mixed fp8 x f16 WMMA, so this
// uses v_wmma_f32_16x16x16_f16_f16 at 207 TF/s where the MXFP4 and int4 kernels get 412 on the
// fp8 pipe. At DECODE that costs nothing -- those kernels run at ~95% of the streaming roofline
// with the matrix units mostly idle -- and it buys exactness: fp16 activations need no
// quantisation at all, so the only error in the whole path is the trellis itself. At PREFILL the
// halved matrix rate WILL bite, and the answer there is to convert the decoded weights to e4m3
// and take the fp8 pipe: e4m3's ~6% relative error adds essentially nothing in quadrature to a
// 2-bit quantisation already sitting at 0.34 relative. That is a numerics change, so it is left
// behind a flag and measured rather than assumed.
//
// WEIGHT PATH. Each lane decodes 8 CONSECUTIVE symbols out of ONE 64-bit merge, which is the
// arrangement the EXL3 packing exists to permit; handing a lane the 8 symbols its WMMA fragment
// wants instead would cost 8 separate window extractions. So decode in EXL3 lane order, stage
// through LDS transposed to [n][k], and read the fragment back in gfx1201 order. The LDS round
// trip is affordable: ablating it entirely out of the int4 decode kernel measured ZERO, because it
// hides under global-load latency (mxfp4_work/ar/RESULTS.md).
#define ESCHA_DEC_PAD 8

// ABL is a MEASUREMENT-ONLY escape hatch and produces WRONG RESULTS when non-zero. It exists to
// attribute the runtime between the trellis codec and everything else:
//   ABL=1  skip the codec math, keep the loads / LDS / WMMA  -> the no-codec floor
//   ABL=2  skip the global loads, keep the codec math        -> the pure-ALU floor
// Nothing outside the benchmark may instantiate it with ABL != 0; the gates run ABL=0 only.
template <int DWN, int DKS, int DTM, int K, int KB = 1, int PAD = 8, bool MUL24 = false,
          int ABL = 0>
__global__ __launch_bounds__(DWN * 32) void escha_gemm_decode(
    const __half *__restrict__ Ah, const unsigned int *__restrict__ code, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int M, int N, int Kdim) {
  constexpr int BND = DWN * 16;
  constexpr int WORDS = 256 * K / 32;
  constexpr int KW = KB * 16;                 // k-values staged per barrier round
  constexpr int STR = KW + PAD;               // shared stride, in halves
  // PAD keeps the fragment reads both 8-byte aligned and bank-conflict-free. A row is STR halves
  // = 2*STR bytes; the 8-byte loads need STR % 4 == 0, and the 16 lanes of a fragment read start
  // STR/2 dwords apart, which spreads over all 32 banks when STR/2 is 2 mod 4. KB=4/PAD=4 gives
  // STR=68: 136-byte rows (8-byte aligned) and 34-dword strides (conflict-free).
  static_assert(STR % 4 == 0, "STR must be a multiple of 4 halves for 8-byte fragment loads");
  __shared__ __half sW[BND * STR];
  // sA is double-buffered so the loop needs ONE __syncthreads per trip instead of two. The
  // trailing barrier existed only to stop the next trip's staging from overwriting activations
  // other waves were still reading; alternating buffers removes that hazard outright. sA is
  // 16*DTM*STR halves -- 2.2 KB at DTM=1 -- so the second copy is nearly free, and dropping a
  // block-wide barrier matters here because the loop is stall-bound, not issue-bound.
  // ... but only when it fits. At KB=8/DTM=4 the pair would need 67.6 KB against the 64 KB
  // limit, so wide-k + wide-M configurations fall back to a single buffer and the trailing
  // barrier. ABUF is the switch for both.
  static constexpr int ATILE = ESCHA_TILE * DTM * STR;
  static constexpr int ABUF = (2 * (BND * STR + 2 * ATILE) + 4 <= 65536) ? 2 : 1;
  __shared__ __half sA[ABUF * ATILE];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kh = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND, ks = blockIdx.z;
  const int ktiles = Kdim / 16, ntiles = N / 16;
  const int per = (ktiles + DKS - 1) / DKS;
  const int t_lo = ks * per, t_hi = min(ktiles, t_lo + per);
  const int n_lane = n0 + wave * 16 + col;

  int ia, ib, sh;
  escha_widx<K>(lane, &ia, &ib, &sh);
  const int gn = n0 / 16 + wave;
  // Byte offsets, not pointers. The tile stride is ntiles*WORDS*4 B -- far past the 13-bit
  // immediate field -- so the naive `base + (size_t)kt*stride` cost a 64-bit multiply-add per
  // tile (25 v_add_co / 10 v_mad_co_i64 in the census). A 32-bit running offset against one
  // wave-uniform base lowers to the SADDR form and one v_add per tile.
  const unsigned int *__restrict__ wbase =
      code + (size_t)(gn < ntiles ? gn : ntiles - 1) * WORDS;
  const unsigned int wstride = (unsigned int)ntiles * WORDS;
  unsigned int woff = (unsigned int)t_lo * wstride;
  // Trellis position is fixed per lane; only the k-offset moves.
  const int tr0 = (lane % 4) * 2, tc0 = lane / 4;
  const int khi = t_hi * 16;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  int abuf = 0;
  for (int kt = t_lo; kt < t_hi; kt += KB, woff += KB * wstride, abuf ^= (ABUF - 1)) {
    unsigned int wlo[KB], whi[KB];
#pragma unroll
    for (int t = 0; t < KB; ++t) {
      // A dead tile past t_hi is CLAMPED, not predicated, and its garbage is killed on the
      // activation side instead: sA is zeroed for k >= khi, so the product is zero regardless of
      // what the weight decode produced. That deletes the per-element select entirely.
      const unsigned int o = kt + t < t_hi ? woff + t * wstride : woff;
      if constexpr (ABL == 2) { wlo[t] = o + ia; whi[t] = o + ib; }
      else { wlo[t] = wbase[o + ia]; whi[t] = wbase[o + ib]; }
    }
#pragma unroll
    for (int off = 0; off < ESCHA_TILE * DTM * KW; off += DWN * 32 * 2) {
      const int idx = off + tid * 2;
      if (idx < ESCHA_TILE * DTM * KW) {
        const int r = idx / KW, c = idx % KW;
        const int rc = r < M - 1 ? r : M - 1;
        const int kk = kt * 16 + c;
        *(unsigned int *)(&sA[abuf * ATILE + r * STR + c]) =
            kk + 1 < khi ? *(const unsigned int *)(Ah + (size_t)rc * Kdim + kk) : 0u;
      }
    }
#pragma unroll
    for (int t = 0; t < KB; ++t) {
      unsigned int st[8];
      escha_states8_from<K>(wlo[t], whi[t], sh, st);
      // The trellis pairs (j, j+1) land on ADJACENT rows of the same column -- tile_pos gives
      // rows {r0, r0+1, r0+8, r0+9} -- and sW is transposed to [n][k], so each pair is one
      // contiguous 4-byte store. escha_decode2 already returns them in (low=j, high=j+1) order,
      // so the half2 goes straight down as ds_write_b32: 4 stores per lane per tile, not 8, and
      // the __low2half/__high2half unpacking disappears with them.
      __half *w0 = &sW[(wave * 16 + tc0) * STR + t * 16 + tr0];
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int dr = (j & 2) ? 8 : 0, dc = (j >= 4) ? 8 : 0;
        if constexpr (ABL == 1) {
          __half2 raw;
          const unsigned int r = st[j] | (st[j + 1] << 16);
          __builtin_memcpy(&raw, &r, 4);
          *(__half2 *)(w0 + dc * STR + dr) = raw;
        } else {
          *(__half2 *)(w0 + dc * STR + dr) = escha_decode2<MUL24>(st[j], st[j + 1]);
        }
      }
    }
    // One block-wide barrier per trip, and it is there for sA alone. sW is written and read by
    // the SAME wave -- wave w owns rows [w*16, w*16+16) -- so it needs only intra-wave ordering,
    // which the LDS dependency plus a wave barrier gives without stalling the other seven waves.
    __syncthreads();
    __builtin_amdgcn_wave_barrier();

#pragma unroll
    for (int t = 0; t < KB; ++t) {
      half8_t af[DTM], wf;
      const __half *pw = &sW[(wave * 16 + col) * STR + t * 16 + kh];
      *(uint2_t *)&wf = *(const uint2_t *)pw;
      *((uint2_t *)&wf + 1) = *(const uint2_t *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const __half *pa = &sA[abuf * ATILE + (i * 16 + col) * STR + t * 16 + kh];
        *(uint2_t *)&af[i] = *(const uint2_t *)pa;
        *((uint2_t *)&af[i] + 1) = *(const uint2_t *)(pa + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[i], wf, acc[i]);
    }
    if constexpr (ABUF == 1) __syncthreads();
  }

  if constexpr (DKS == 1) {           // no partials, no atomic -- write C directly
    if (n_lane < N)
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kh + e;
          if (m < M) C[(size_t)m * N + n_lane] = (__bf16)acc[i][e];
        }
    return;
  }
  if (n_lane < N)
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kh + e;
        if (m < M) P[((size_t)ks * M + m) * N + n_lane] = acc[i][e];
      }
  __syncthreads();
  if (tid == 0) { __threadfence(); s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1); }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DWN * 32)
    for (int m = 0; m < M; ++m) {
      float s = 0.f;
      for (int k = 0; k < DKS; ++k) s += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)s;
    }
}

// ---------------------------------------------------------------------------------------------
// Prefill GEMM.   C[M,N] = Ah[M,K] @ decode(code)[K,N] * As[m]
//
// WHY THIS ONE TAKES THE fp8 PIPE AND THE DECODE KERNEL DOES NOT. Prefill here is entirely
// compute-bound: at M=8192 on gate_up it is 1.46 TFLOP against 27.5 MB of weights, so the weight
// stream is 0.6% of the compute time and the WMMA rate IS the ceiling. gfx1201 runs fp8 WMMA at
// 412 TF/s against f16's 207, so converting the decoded weights to e4m3 is worth a clean 2x.
//
// That conversion is a numerics change, so it was measured rather than assumed: rounding the
// decoded weights to e4m3 and re-running the reconstruction against the fp8 base adds
// 0.19% / 0.59% / 0.28% relative on the three projections tested. The two errors add in
// quadrature and e4m3's ~3.6% RMS is far inside a trellis error of 0.22-0.45, so it is free.
// (mxfp4_work/escha/fp8_cost.py.)
//
// THE DECODE IS FREE HERE, unlike in the decode kernel. BMF=256 means each decoded weight feeds
// 256 MACs, so ~4 ALU of trellis decode is ~0.016 ALU per MAC. Decode once per slab into LDS and
// the codec stops mattering -- which is why this kernel can afford a codebook the decode kernel
// has to think about.
#define EP_TM 4
#define EP_WM 4
#define EP_WN 2
#define EP_BK 64
#define EP_PAD 8
#define EP_STR (EP_BK + EP_PAD)
#define EP_NWAVE (EP_WM * EP_WN)
#define EP_NTHREADS (EP_NWAVE * 32)
#define EP_BMF (EP_WM * EP_TM * 16)

__device__ __forceinline__ unsigned int escha_pk_e4m3(float a, float b) {
  return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);   // two e4m3 in the low 16 bits
}

// TM is the number of 16-row M-fragments each wave accumulates, so the block covers
// EP_WM*TM*16 rows. It is the lever on CODEC REDUNDANCY: a workgroup decodes the whole K x BNF
// weight slab for its own row block, so every weight is decoded ceil(M/BMF) times. At M=2048 with
// the original TM=4 that is 8 decodes per weight -- and the codec is not cheap here. Doubling TM
// halves that, at the cost of TM*TN*8 accumulator VGPRs.
// WM is the wave count along M. It reaches the same block width as TM but spends WAVES instead of
// REGISTERS: WM=8/TM=4 and WM=4/TM=8 both cover 512 rows, the first with 142 VGPRs across 16 waves,
// the second with 198 across 8. LDS is identical, so which wins is an occupancy question and is
// settled by measurement.
template <int TN, int K, int TM = EP_TM, int WM = EP_WM>
__global__ __launch_bounds__(WM * EP_WN * 32) void escha_gemm_prefill(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ code,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int Kdim) {
  // TM*TN accumulators of 8 floats each is the whole register budget. TM=8/TN=4 needs 256 VGPRs
  // before a single fragment is live and spills 1146 of them -- and a spilling kernel does not
  // merely run slow, it holds the clocks down for whatever is measured after it. Refuse to build
  // the configuration rather than discover it in a benchmark.
  static_assert(TM * TN <= 16, "TM*TN accumulators exceed the register budget; this will spill");
  constexpr int BNF = EP_WN * TN * 16, BMF = WM * TM * 16;
  constexpr int NWAVE = WM * EP_WN, NTHREADS = NWAVE * 32;
  constexpr int WORDS = 256 * K / 32;
  constexpr int NT = BNF / 16, KT = EP_BK / 16;      // tiles per slab
  __shared__ unsigned char sA[BMF * EP_STR];
  __shared__ unsigned char sW[BNF * EP_STR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / EP_WN, wn = wave % EP_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ntiles = N / 16;

  int ia, ib, sh;
  escha_widx<K>(lane, &ia, &ib, &sh);
  const int tr0 = (lane % 4) * 2, tc0 = lane / 4;

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  int ncol[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) ncol[j] = n0 + wn * TN * 16 + j * 16 + col;

  for (int k0 = 0; k0 < Kdim; k0 += EP_BK) {
    // A tile: 16 B per thread. Clamped, never predicated -- a bounds-predicated load sits in its
    // own s_and_saveexec region and forces a counted s_wait_loadcnt after each one.
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * Kdim + k0;
#pragma unroll
    for (int off = 0; off < BMF * EP_BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / EP_BK, c = idx % EP_BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      *(uint4_t *)(&sA[r * EP_STR + c]) = *(const uint4_t *)(Ab + (size_t)rc * Kdim + c);
    }
    // Weights: NT*KT tiles this slab, one wave at a time. Each lane decodes 8 consecutive symbols
    // from a single 64-bit merge, converts them to e4m3 in pairs, and writes them transposed to
    // [n][k] so the fragment read below is contiguous in k.
    // Same three fixes the decode kernel got, and for the same reasons: the word indices and the
    // trellis position depend only on the lane, so they hoist clear of the tile loop; the window
    // extraction goes through the register-argument form (one v_alignbit for K=2 instead of a
    // 64-bit shift pair); and the two e4m3 bytes of a pair land on ADJACENT rows of one column,
    // so they leave as a single 16-bit store rather than two byte stores.
#pragma unroll
    for (int t = wave; t < NT * KT; t += NWAVE) {
      const int nt = t / KT, ktl = t % KT;
      const int gn = n0 / 16 + nt, gk = k0 / 16 + ktl;
      const unsigned int *__restrict__ w =
          code + ((size_t)gk * ntiles + (gn < ntiles ? gn : ntiles - 1)) * WORDS;
      unsigned int st[8];
      escha_states8_from<K>(w[ia], w[ib], sh, st);
      unsigned char *w0 = &sW[(nt * 16 + tc0) * EP_STR + ktl * 16 + tr0];
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int dr = (j & 2) ? 8 : 0, dc = (j >= 4) ? 8 : 0;
        const __half2 d = escha_decode2<>(st[j], st[j + 1]);
        const unsigned int p = escha_pk_e4m3(__half2float(__low2half(d)),
                                             __half2float(__high2half(d)));
        *(unsigned short *)(w0 + dc * EP_STR + dr) = (unsigned short)(p & 0xFFFFu);
      }
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < EP_BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * EP_STR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * EP_STR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Fence the k-step: without it the compiler hoists every step's fragment loads above the
      // first WMMA and the register shuffling that keeps four steps live costs more than the
      // prefetch buys. Worth 1.8-3.4% on the MXFP4 GEMM.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Epilogue: wave-uniform base indexed by a 32-bit offset so the compiler emits the SADDR form,
  // and a branch-free path for blocks entirely inside M and N. Worth 3.3-8.0% on the int4 and
  // MXFP4 kernels; only the last row-block and column-block are ragged on a real prefill.
  __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
  const float *__restrict__ Asb = As + m0 + wm * TM * 16;
  const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                    (ncol[TN - 1] < N);
  if (full) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = i * 16 + kb8 + e;
          Cb[r * N + ncol[j]] = (__bf16)(acc[i][j][e] * Asb[r]);
        }
    return;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        if (m < M && ncol[j] < N) C[(size_t)m * N + ncol[j]] = (__bf16)(acc[i][j][e] * As[m]);
      }
}

// Split-K selection for the decode kernel.
//
// This is the single largest tuning lever and it cannot be read off N. The `down` shape is N=5120,
// which at BND=128 is 40 workgroups on a 64-CU part -- 24 CUs idle for the whole GEMM, and DKS=8
// there is worth 1.8x. But an N-based rule is overfit: at the SAME nblk=40, K=4352 wants DKS=4
// while K=8704 wants DKS=8, because what is being divided is k-work, not columns.
//
// Two forces set the optimum, and the constants below are fitted to 18 measured (nblk, ktiles, M)
// points spanning the TP=1 and TP=2 per-GPU shapes:
//   * each split must keep enough k-tiles to be worth its own block -- below ~36 the fixed cost
//     per workgroup stops being amortized;
//   * the split writes DKS*M*N fp32 partials and reads them back, a cost linear in DKS, so past
//     roughly 576 workgroups the extra parallelism no longer pays for the traffic.
// The fit needs no M term and lands within 2.0% of the measured optimum on every point, exact on
// 14 of 18. Raising DEC_KS_MAX beyond 8 was not measured and must not be assumed.
__device__ __host__ __forceinline__ int escha_decode_split_k(int nblk, int ktiles) {
  constexpr int T = 36, WG_CAP = 576, KS_MAX = 8;
  int ks = 1;
  for (int c = 2; c <= KS_MAX; c <<= 1)
    if (ktiles / c >= T && nblk * c <= WG_CAP) ks = c;
  return ks;
}

// =================================================================================================
// DECODE ON THE FP8 PIPE
//
// The f16 decode kernel above keeps the trellis output in fp16 all the way into the matrix unit,
// which is exact but leaves half the hardware on the table: gfx1201 runs f16 WMMA at 207 TF/s and
// fp8 WMMA at 412. The ablation made the cost of that choice explicit -- at M>=40 the codec is
// only 3-14% of runtime, yet the kernel still trailed MXFP4 by 1.7-2.0x, because MXFP4 is on the
// fp8 pipe and this one was not.
//
// Rounding the decoded weight to e4m3 costs precision, but less than it sounds: MXFP4's own weights
// are e2m1 with a shared per-32 exponent -- strictly coarser than e4m3 -- so this stays on the more
// accurate side of the kernel it is being compared against, and the prefill kernel already took
// the same trade and landed on the bf16 rounding floor.
//
// Two structural wins come with it, both of which matter more than the WMMA rate at decode:
//   * fragments halve, 8 B per lane instead of 16, so LDS read traffic halves;
//   * the whole tile halves, so LDS per block drops ~19.6 KB -> ~11.5 KB and occupancy rises from
//     3 blocks/CU to 5.
// Activations arrive already quantized per token, matching the W4A8 serving path and the MXFP4
// decode kernel's signature exactly.
template <int DWN, int DKS, int DTM, int K, int KB = 4, int PAD = 8, bool MUL24 = false>
__global__ __launch_bounds__(DWN * 32) void escha_gemm_decode_fp8(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ code,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int Kdim) {
  constexpr int BND = DWN * 16;
  constexpr int WORDS = 256 * K / 32;
  constexpr int KW = KB * 16;
  constexpr int STR = KW + PAD;              // bytes now, not halves
  // 8-byte fragment loads need STR % 8 == 0; STR/4 == 2 (mod 4) puts the 16 lanes of a fragment
  // read on 16 distinct bank pairs. PAD=8 satisfies both for every KB in use.
  static_assert(STR % 8 == 0 && (STR / 4) % 4 == 2, "pick PAD so fragment reads stay conflict-free");
  static constexpr int ATILE = ESCHA_TILE * DTM * STR;
  static constexpr int ABUF = (BND * STR + 2 * ATILE + 4 <= 65536) ? 2 : 1;
  __shared__ unsigned char sW[BND * STR];
  __shared__ unsigned char sA[ABUF * ATILE];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kh = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND, ks = blockIdx.z;
  const int ktiles = Kdim / 16, ntiles = N / 16;
  const int per = (ktiles + DKS - 1) / DKS;
  const int t_lo = ks * per, t_hi = min(ktiles, t_lo + per);
  const int n_lane = n0 + wave * 16 + col;

  int ia, ib, sh;
  escha_widx<K>(lane, &ia, &ib, &sh);
  const int gn = n0 / 16 + wave;
  const unsigned int *__restrict__ wbase =
      code + (size_t)(gn < ntiles ? gn : ntiles - 1) * WORDS;
  const unsigned int wstride = (unsigned int)ntiles * WORDS;
  unsigned int woff = (unsigned int)t_lo * wstride;
  const int tr0 = (lane % 4) * 2, tc0 = lane / 4;
  const int khi = t_hi * 16;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  int abuf = 0;
  for (int kt = t_lo; kt < t_hi; kt += KB, woff += KB * wstride, abuf ^= (ABUF - 1)) {
    unsigned int wlo[KB], whi[KB];
#pragma unroll
    for (int t = 0; t < KB; ++t) {
      const unsigned int o = kt + t < t_hi ? woff + t * wstride : woff;
      wlo[t] = wbase[o + ia];
      whi[t] = wbase[o + ib];
    }
    // Activations are one byte each, so a thread moves four k at a time.
#pragma unroll
    for (int off = 0; off < ESCHA_TILE * DTM * KW; off += DWN * 32 * 4) {
      const int idx = off + tid * 4;
      if (idx < ESCHA_TILE * DTM * KW) {
        const int r = idx / KW, c = idx % KW;
        const int rc = r < M - 1 ? r : M - 1;
        const int kk = kt * 16 + c;
        *(unsigned int *)(&sA[abuf * ATILE + r * STR + c]) =
            kk + 3 < khi ? *(const unsigned int *)(A + (size_t)rc * Kdim + kk) : 0u;
      }
    }
#pragma unroll
    for (int t = 0; t < KB; ++t) {
      unsigned int st[8];
      escha_states8_from<K>(wlo[t], whi[t], sh, st);
      unsigned char *w0 = &sW[(wave * 16 + tc0) * STR + t * 16 + tr0];
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int dr = (j & 2) ? 8 : 0, dc = (j >= 4) ? 8 : 0;
        const __half2 d = escha_decode2<MUL24>(st[j], st[j + 1]);
        // The trellis pair lands on adjacent rows of one column, and sW is transposed to [n][k],
        // so the two e4m3 bytes are contiguous: one ds_write_b16 per pair.
        const unsigned int p = escha_pk_e4m3(__half2float(__low2half(d)),
                                             __half2float(__high2half(d)));
        *(unsigned short *)(w0 + dc * STR + dr) = (unsigned short)(p & 0xFFFFu);
      }
    }
    __syncthreads();
    __builtin_amdgcn_wave_barrier();

#pragma unroll
    for (int t = 0; t < KB; ++t) {
      int2_t af[DTM], wf;
      const unsigned char *pw = &sW[(wave * 16 + col) * STR + t * 16 + kh];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[abuf * ATILE + (i * 16 + col) * STR + t * 16 + kh];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    if constexpr (ABUF == 1) __syncthreads();
  }

  if constexpr (DKS == 1) {
    if (n_lane < N)
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kh + e;
          if (m < M) C[(size_t)m * N + n_lane] = (__bf16)(acc[i][e] * As[m]);
        }
    return;
  }
  if (n_lane < N)
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kh + e;
        if (m < M) P[((size_t)ks * M + m) * N + n_lane] = acc[i][e];
      }
  __syncthreads();
  if (tid == 0) { __threadfence(); s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1); }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DWN * 32)
    for (int m = 0; m < M; ++m) {
      float s = 0.f;
      for (int k = 0; k < DKS; ++k) s += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(s * As[m]);
    }
}
