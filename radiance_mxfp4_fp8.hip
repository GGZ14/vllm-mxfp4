// RADIANCE gfx1201 MXFP4 x FP8 GEMM: C[M,N] bf16 = A_fp8[M,K] . dequant(W_mxfp4)[N,K]^T
//
// Why a hand-written kernel: Triton lowers tl.dot_scaled by upconverting e2m1 to bf16 and using the
// 16-bit WMMA. Measured on this card, register-resident: fp8 WMMA 325.2 TFLOP/s vs f16 WMMA 160.2
// (2.03x), while Triton's own tl.dot manages 102.8 bf16 and only 43.3 on fp8 operands -- i.e. Triton
// will not emit the fp8 matrix instruction, it upconverts and pays conversion on top. The only way
// to the 2x is v_wmma_f32_16x16x16_fp8_fp8 by hand.
//
// Numerics: this is W4A8, not the checkpoint's declared W4A4. E2M1's 16 values are all exactly
// representable in e4m3 so the weight upconvert is a lossless table lookup, and the E8M0 scale is a
// power of two so applying it to the fp32 accumulator is exact. Activations are fp8 rather than fp4,
// which is strictly MORE precise than what the checkpoint was calibrated against -- but it does mean
// output is no longer bit-identical to the emulated path, so it is opt-in behind RADIANCE_MXFP4_W4A8.
//
// Layouts, matching what QuarkOCP_MX.process_weights_after_loading leaves on the layer:
//   A   [M,K]      e4m3, per-token scale As[M] applied in the epilogue
//   W   [N,K/2]    packed e2m1, low nibble = even k
//   Ws  [K/32,N]   e8m0, TRANSPOSED (the non-asm branch stores .T.contiguous()), which also makes
//                  the scale read coalesced across lanes
//
// Fragment layout confirmed empirically (0/256 elements wrong on a 16x16x16 tile):
//   A: lane l holds A[l%16][(l/16)*8+j] | B: B[(l/16)*8+j][l%16] | C: C[(l/16)*8+j][l%16]
//
// LDS rows are padded 8 B: without it, 16 lanes reading rows 64 B apart collide 8 ways on the
// 32 x 4 B banks. Staging through LDS is what makes this fast at all -- reading fragments straight
// from global ran at 9.5 TFLOP/s because lanes 16 rows apart touch 16 cache lines per operand.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
namespace py = pybind11;

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

__device__ __constant__ unsigned char kLUT[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

__device__ __constant__ unsigned int kMag[16][2] = {
#define MROW(d) {\
  (unsigned int)(0u | ((unsigned)((6-(d))>0?(0x30-((d)<<3)):0)<<8) |\
   ((unsigned)((7-(d))>0?(0x38-((d)<<3)):0)<<16) |\
   ((unsigned)((7-(d))>0?(0x3C-((d)<<3)):0)<<24)),\
  (unsigned int)((unsigned)((8-(d))>0?(0x40-((d)<<3)):0) |\
   ((unsigned)((8-(d))>0?(0x44-((d)<<3)):0)<<8) |\
   ((unsigned)((9-(d))>0?(0x48-((d)<<3)):0)<<16) |\
   ((unsigned)((9-(d))>0?(0x4C-((d)<<3)):0)<<24))}
 MROW(0),MROW(1),MROW(2),MROW(3),MROW(4),MROW(5),MROW(6),MROW(7),
 MROW(8),MROW(9),MROW(10),MROW(11),MROW(12),MROW(13),MROW(14),MROW(15)
#undef MROW
};

// kLUT2[d][code]: e2m1 code -> e4m3 byte, already shifted DOWN by d binades. Folding the MX block
// exponent into the WEIGHT here removes the per-32-block rescale from the inner loop entirely --
// the scale becomes one per-row factor applied in the epilogue.
//
// Exact while the shifted value stays an e4m3 normal: smallest e2m1 magnitude is 0.5 = 2^-1 and
// e4m3's min normal is 2^-6, so d <= 5 is lossless. Measured on AMD's checkpoint: 99.998% of 761M
// blocks need d <= 5, worst case anywhere is 10, and 282 blocks (0.000037%) would flush to zero.
__device__ __constant__ unsigned char kLUT2[16][16] = {
#define RAD_ROW(d)                                                                             \
  {0x00,                                                                                       \
   (unsigned char)((6 - (d)) > 0 ? (0x30 - ((d) << 3)) : 0),                                   \
   (unsigned char)((7 - (d)) > 0 ? (0x38 - ((d) << 3)) : 0),                                   \
   (unsigned char)((7 - (d)) > 0 ? (0x3C - ((d) << 3)) : 0),                                   \
   (unsigned char)((8 - (d)) > 0 ? (0x40 - ((d) << 3)) : 0),                                   \
   (unsigned char)((8 - (d)) > 0 ? (0x44 - ((d) << 3)) : 0),                                   \
   (unsigned char)((9 - (d)) > 0 ? (0x48 - ((d) << 3)) : 0),                                   \
   (unsigned char)((9 - (d)) > 0 ? (0x4C - ((d) << 3)) : 0),                                   \
   0x80,                                                                                       \
   (unsigned char)((6 - (d)) > 0 ? (0xB0 - ((d) << 3)) : 0x80),                                \
   (unsigned char)((7 - (d)) > 0 ? (0xB8 - ((d) << 3)) : 0x80),                                \
   (unsigned char)((7 - (d)) > 0 ? (0xBC - ((d) << 3)) : 0x80),                                \
   (unsigned char)((8 - (d)) > 0 ? (0xC0 - ((d) << 3)) : 0x80),                                \
   (unsigned char)((8 - (d)) > 0 ? (0xC4 - ((d) << 3)) : 0x80),                                \
   (unsigned char)((9 - (d)) > 0 ? (0xC8 - ((d) << 3)) : 0x80),                                \
   (unsigned char)((9 - (d)) > 0 ? (0xCC - ((d) << 3)) : 0x80)}
    RAD_ROW(0), RAD_ROW(1), RAD_ROW(2),  RAD_ROW(3),  RAD_ROW(4),  RAD_ROW(5),  RAD_ROW(6),
    RAD_ROW(7), RAD_ROW(8), RAD_ROW(9),  RAD_ROW(10), RAD_ROW(11), RAD_ROW(12), RAD_ROW(13),
    RAD_ROW(14), RAD_ROW(15)
#undef RAD_ROW
};

#define BM 128
#define BN 64
#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
#define NWAVE 8

__global__ __launch_bounds__(NWAVE * 32) void radiance_mxfp4_fp8_gemm(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const float *__restrict__ As,
    __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char sA[BM * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ unsigned char sS[BN * (BK / 32)];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave >> 1, wn = wave & 1;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nblk = K / 32;

  floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    {   // A tile: 32 B per thread, guarded on the M tail
      const int r = tid >> 1, off = (tid & 1) * 32, gm = m0 + r;
      uint4_t v0, v1;
      if (gm < M) {
        const unsigned char *src = A + (size_t)gm * K + k0 + off;
        v0 = *(const uint4_t *)(src);
        v1 = *(const uint4_t *)(src + 16);
      } else {
        v0 = uint4_t{0, 0, 0, 0};
        v1 = uint4_t{0, 0, 0, 0};
      }
      *(uint4_t *)(&sA[r * ASTR + off]) = v0;
      *(uint4_t *)(&sA[r * ASTR + off + 16]) = v1;
    }
    {   // W tile: 8 packed bytes per thread -> 16 e4m3 bytes, guarded on the N tail
      const int r = tid >> 2, off = (tid & 3) * 8, gn = n0 + r;
      unsigned long long packed = 0ull;
      if (gn < N) packed = *(const unsigned long long *)(W + ((size_t)gn * K + k0) / 2 + off);
      unsigned char out[16];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const unsigned char byte = (packed >> (8 * t)) & 0xFF;
        out[2 * t] = kLUT[byte & 0xF];
        out[2 * t + 1] = kLUT[(byte >> 4) & 0xF];
      }
      *(uint4_t *)(&sW[r * WSTR + off * 2]) = *(const uint4_t *)out;
    }
    if (tid < BN * (BK / 32)) {   // scales are [K/32, N]: coalesced across n
      const int r = tid / (BK / 32), b = tid % (BK / 32);
      const int gn = n0 + r;
      sS[tid] = (gn < N) ? Ws[(size_t)(k0 / 32 + b) * N + gn] : 127;
    }
    __syncthreads();

#pragma unroll
    for (int blk = 0; blk < BK / 32; ++blk) {
      floatx8 tmp[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;
#pragma unroll
      for (int step = 0; step < 2; ++step) {     // two WMMA steps span one 32-wide MX block
        const int kk = blk * 32 + step * 16 + kb8;
        int2_t af[2], wf[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kk];
          af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kk];
          wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // E8M0 is a power of two by definition, so 2^(e-127) is exactly the IEEE-754 float
        // whose exponent field is e: bits = e << 23. One integer shift instead of a
        // transcendental, worth +4% (exp2f ran 4x per lane per K-slab). e=0 would mean 2^-127
        // (denormal) and yields 0.0f here; it only arises for all-zero blocks, where the
        // product is 0 either way.
        const float s = __int_as_float(
            (int)sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk] << 23);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += tmp[i][j][e] * s;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * 32 + i * 16 + kb8 + e;
        const int n = n0 + wn * 32 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * As[m]);
      }
}


// Autotuned folded-scale kernel. The MX block exponent is pushed into the weight byte during the
// LDS upconvert (kLUT2), so the inner loop is pure WMMA -- no temp accumulator, no per-32-block
// rescale, no LDS scale reads. Lossless while the shifted value stays an e4m3 normal (d <= 5):
// 99.998% of the checkpoint's 761M blocks qualify, worst case 10, 282 blocks flush.
//
// Tile config from a correctness-gated sweep of 28 shapes (BM/BN/BK/TM/TN/wave grid). BM=256 via
// TM=4 won at EVERY production shape and batch size -- a taller tile buys more A reuse per weight
// read. It also removed the large-M cliff the earlier hand-written BM=128 version had.
#define TM 4
#define WM 4
#define WN 2
#define NWAVE (WM * WN)
#define NTHREADS (NWAVE * 32)
#define BMF (WM * TM * 16)
#define BNF_OF(tn) (WN * (tn) * 16)
#define BNF BNF_OF(2)

// TN is templated because the winner depends on M. Widening the N tile (TN=4 -> BNF=128) amortises
// the A-tile staging, which an ablation prices at 24% of runtime -- A is BMF*BK = 16 KB per K-slab
// against W's 2 KB, and with BNF=64 it is re-staged across twice as many column blocks. Measured on
// gate_up 17408x5120, tight A/B, 10 alternations with no overlap between the distributions:
//   M=8192 +10.0%  M=6144 +9.7%  M=4096 +8.5%  M=3072 +6.9%
//   M=2048 +1.3%   M=1536 -1.6%  M=1024 -4.2%  M=512 -8.8%
// Below the crossover the wide tile cannot fill: it halves the block count in N while BMF still
// masks most of a short M. Hence TN=4 only from M >= 2048; RADIANCE_MXFP4_TN4_MIN_M overrides.
template <int TN>
__global__ __launch_bounds__(NTHREADS) void radiance_mxfp4_fp8_gemm_folded(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[WN * TN * 16 * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = WN * TN * 16;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- A tile: BM*BK bytes, 16 B per thread per iteration ----
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK, gm = m0 + r;
      uint4_t v = uint4_t{0, 0, 0, 0};
      if (gm < M) v = *(const uint4_t *)(A + (size_t)gm * K + k0 + c);
      *(uint4_t *)(&sA[r * ASTR + c]) = v;
    }
    // ---- W tile: BN*BK/2 packed bytes, 8 B per thread per iteration -> 16 e4m3 bytes ----
#pragma unroll
    for (int off = 0; off < BNF_T * BK / 2; off += NTHREADS * 8) {
      const int idx = off + tid * 8;
      const int r = idx / (BK / 2), c = idx % (BK / 2), gn = n0 + r;
      unsigned long long packed = 0ull;
      int dsh = 0;
      if (gn < N) {
        packed = *(const unsigned long long *)(W + ((size_t)gn * K + k0) / 2 + c);
        const int blk = (k0 + c * 2) / 32;
        dsh = (int)Wref[gn] - (int)Ws[(size_t)blk * N + gn];   // scales are [K/32, N]
        dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      }
      // v_perm_b32 resolves 4 table lookups per instruction from the 8-byte magnitude table;
      // the sign is separable because entries 8-15 are entries 0-7 with bit 7 set. The upconvert
      // measured 21% of runtime with per-byte constant loads (156 -> 188 TF with it removed
      // entirely), and this recovers +9-12% of that.
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * WSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]);
      }
}

static void launch(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                   uintptr_t c, int M, int N, int K, uintptr_t stream) {
  if (K % BK) throw std::runtime_error("radiance_mxfp4_fp8: K must be a multiple of 64");
  dim3 block(NWAVE * 32);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  // the tuned folded kernel wins at every shape measured, so it is the default whenever the
  // per-row reference exponent is available
  if (wref) {
    static const int tn4_min_m = [] {
      const char *e = getenv("RADIANCE_MXFP4_TN4_MIN_M");
      return e ? atoi(e) : 2048;
    }();
    dim3 fblock(NTHREADS);
    if (M >= tn4_min_m) {
      constexpr int B4 = BNF_OF(4);
      dim3 fgrid((N + B4 - 1) / B4, (M + BMF - 1) / BMF);
      hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm_folded<4>, fgrid, fblock, 0, (hipStream_t)stream,
                         (const unsigned char *)a, (const unsigned char *)w,
                         (const unsigned char *)ws, (const unsigned char *)wref,
                         (const float *)as, (__bf16 *)c, M, N, K);
    } else {
      constexpr int B2 = BNF_OF(2);
      dim3 fgrid((N + B2 - 1) / B2, (M + BMF - 1) / BMF);
      hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm_folded<2>, fgrid, fblock, 0, (hipStream_t)stream,
                         (const unsigned char *)a, (const unsigned char *)w,
                         (const unsigned char *)ws, (const unsigned char *)wref,
                         (const float *)as, (__bf16 *)c, M, N, K);
    }
  } else
    hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm, grid, block, 0, (hipStream_t)stream,
                       (const unsigned char *)a, (const unsigned char *)w,
                       (const unsigned char *)ws, (const float *)as, (__bf16 *)c, M, N, K);
}

PYBIND11_MODULE(radiance_mxfp4_fp8, m) {
  m.def("launch", &launch, "MXFP4 x FP8 WMMA GEMM (gfx1201)");
}
