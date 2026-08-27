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
__device__ __forceinline__ __half2 escha_decode2(unsigned int s0, unsigned int s1) {
  unsigned int a = s0 * ESCHA_MCG, b = s1 * ESCHA_MCG;
  a = (a & 0x8FFF8FFFu) ^ 0x3B603B60u;
  b = (b & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const __half2 lo = __halves2half2(__ushort_as_half((unsigned short)(a & 0xFFFFu)),
                                    __ushort_as_half((unsigned short)(b & 0xFFFFu)));
  const __half2 hi = __halves2half2(__ushort_as_half((unsigned short)(a >> 16)),
                                    __ushort_as_half((unsigned short)(b >> 16)));
  return __hadd2(lo, hi);
}

// ---------------------------------------------------------------------------------------------
// Trellis state extraction for one lane's eight symbols, t_offset = 8 * lane.
//
// The tile is 256*K bits held as uint32. `fshift(b, a, s) = ((a << 32) | b) >> s` puts the EARLIER
// word in the HIGH half, so stream position advances toward LOWER bit positions and a lane's eight
// symbols come out of ONE 64-bit merge -- which is why EXL3 gives a lane eight CONSECUTIVE
// symbols. Consecutive states overlap by 16-K bits; that overlap is the trellis.
//
// Reading the packer literally instead -- a continuous MSB-first uint16 stream -- is wrong and
// fails silently, producing Gaussians of the right variance in the wrong order. See FORMAT.md.
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

template <int DWN, int DKS, int DTM, int K>
__global__ __launch_bounds__(DWN * 32) void escha_gemm_decode(
    const __half *__restrict__ Ah, const unsigned int *__restrict__ code, float *__restrict__ P,
    int *__restrict__ cnt, __bf16 *__restrict__ C, int M, int N, int Kdim) {
  constexpr int BND = DWN * 16;
  constexpr int WORDS = 256 * K / 32;
  constexpr int WSTR = 16 + ESCHA_DEC_PAD;
  constexpr int ASTR = 16 + ESCHA_DEC_PAD;
  __shared__ __half sW[BND * WSTR];
  __shared__ __half sA[ESCHA_TILE * DTM * ASTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kh = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND, ks = blockIdx.z;
  const int ktiles = Kdim / 16, ntiles = N / 16;
  const int per = (ktiles + DKS - 1) / DKS;
  const int t_lo = ks * per, t_hi = min(ktiles, t_lo + per);
  const int n_lane = n0 + wave * 16 + col;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  for (int kt = t_lo; kt < t_hi; ++kt) {
    // Activations: DTM 16-row fragments, 16 K-values wide. Clamped, never predicated -- a
    // bounds-predicated load lands in its own s_and_saveexec region and forces a counted
    // s_wait_loadcnt after every one (see the int4 kernel's note).
#pragma unroll
    for (int off = 0; off < ESCHA_TILE * DTM * 16; off += DWN * 32 * 2) {
      const int idx = off + tid * 2;
      if (idx < ESCHA_TILE * DTM * 16) {
        const int r = idx / 16, c = idx % 16;
        const int rc = r < M - 1 ? r : M - 1;
        *(unsigned int *)(&sA[r * ASTR + c]) =
            *(const unsigned int *)(Ah + (size_t)rc * Kdim + kt * 16 + c);
      }
    }
    // Weights: wave w decodes the tile for its own 16 N columns.
    {
      const int gn = n0 / 16 + wave;
      const unsigned int *w =
          code + ((size_t)kt * ntiles + (gn < ntiles ? gn : ntiles - 1)) * WORDS;
      unsigned int st[8];
      escha_states8<K>(w, lane, st);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        int r0, c0, r1, c1;
        escha_tile_pos(lane, j, &r0, &c0);
        escha_tile_pos(lane, j + 1, &r1, &c1);
        const __half2 d = escha_decode2(st[j], st[j + 1]);
        sW[(wave * 16 + c0) * WSTR + r0] = __low2half(d);    // transposed to [n][k]
        sW[(wave * 16 + c1) * WSTR + r1] = __high2half(d);
      }
    }
    __syncthreads();

    // One 16x16x16 f16 step. Fragment layout matches the fp8 kernels: lane holds 8 halves,
    // A row = col, W column = col, k offset = (lane>>4)*8.
    {
      half8_t af[DTM], wf;
      // 8 halves = 16 B per lane. WSTR/ASTR are 16+8 halves, so a row start is 48 B -- 16-byte
      // aligned only on even rows, hence two 8-byte loads rather than one 16-byte load.
      const __half *pw = &sW[(wave * 16 + col) * WSTR + kh];
      *(uint2_t *)&wf = *(const uint2_t *)pw;
      *((uint2_t *)&wf + 1) = *(const uint2_t *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const __half *pa = &sA[(i * 16 + col) * ASTR + kh];
        *(uint2_t *)&af[i] = *(const uint2_t *)pa;
        *((uint2_t *)&af[i] + 1) = *(const uint2_t *)(pa + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
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

template <int TN, int K>
__global__ __launch_bounds__(EP_NTHREADS) void escha_gemm_prefill(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ code,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int Kdim) {
  constexpr int BNF = EP_WN * TN * 16;
  constexpr int WORDS = 256 * K / 32;
  constexpr int NT = BNF / 16, KT = EP_BK / 16;      // tiles per slab
  __shared__ unsigned char sA[EP_BMF * EP_STR];
  __shared__ unsigned char sW[BNF * EP_STR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / EP_WN, wn = wave % EP_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * EP_BMF, n0 = blockIdx.x * BNF;
  const int ntiles = N / 16;

  floatx8 acc[EP_TM][TN];
#pragma unroll
  for (int i = 0; i < EP_TM; ++i)
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
    for (int off = 0; off < EP_BMF * EP_BK; off += EP_NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / EP_BK, c = idx % EP_BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      *(uint4_t *)(&sA[r * EP_STR + c]) = *(const uint4_t *)(Ab + (size_t)rc * Kdim + c);
    }
    // Weights: NT*KT tiles this slab, one wave at a time. Each lane decodes 8 consecutive symbols
    // from a single 64-bit merge, converts them to e4m3 in pairs, and writes them transposed to
    // [n][k] so the fragment read below is contiguous in k.
#pragma unroll
    for (int t = wave; t < NT * KT; t += EP_NWAVE) {
      const int nt = t / KT, ktl = t % KT;
      const int gn = n0 / 16 + nt, gk = k0 / 16 + ktl;
      const unsigned int *w =
          code + ((size_t)gk * ntiles + (gn < ntiles ? gn : ntiles - 1)) * WORDS;
      unsigned int st[8];
      escha_states8<K>(w, lane, st);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        int r0, c0, r1, c1;
        escha_tile_pos(lane, j, &r0, &c0);
        escha_tile_pos(lane, j + 1, &r1, &c1);
        const __half2 d = escha_decode2(st[j], st[j + 1]);
        const unsigned int p = escha_pk_e4m3(__half2float(__low2half(d)),
                                             __half2float(__high2half(d)));
        sW[(nt * 16 + c0) * EP_STR + ktl * 16 + r0] = (unsigned char)(p & 0xFF);
        sW[(nt * 16 + c1) * EP_STR + ktl * 16 + r1] = (unsigned char)((p >> 8) & 0xFF);
      }
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < EP_BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[EP_TM], wf[TN];
#pragma unroll
      for (int i = 0; i < EP_TM; ++i) {
        const unsigned char *p = &sA[(wm * EP_TM * 16 + i * 16 + col) * EP_STR + kk];
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
      for (int i = 0; i < EP_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Epilogue: wave-uniform base indexed by a 32-bit offset so the compiler emits the SADDR form,
  // and a branch-free path for blocks entirely inside M and N. Worth 3.3-8.0% on the int4 and
  // MXFP4 kernels; only the last row-block and column-block are ragged on a real prefill.
  __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * EP_TM * 16) * N;
  const float *__restrict__ Asb = As + m0 + wm * EP_TM * 16;
  const bool full = (m0 + wm * EP_TM * 16 + (EP_TM - 1) * 16 + kb8 + 7 < M) &&
                    (ncol[TN - 1] < N);
  if (full) {
#pragma unroll
    for (int i = 0; i < EP_TM; ++i)
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
  for (int i = 0; i < EP_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * EP_TM * 16 + i * 16 + kb8 + e;
        if (m < M && ncol[j] < N) C[(size_t)m * N + ncol[j]] = (__bf16)(acc[i][j][e] * As[m]);
      }
}
