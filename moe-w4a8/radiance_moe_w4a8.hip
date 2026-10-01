// RADIANCE gfx1201 grouped MoE GEMM, MXFP4 weights x FP8 activations (W4A8) on the fp8 WMMA
// (v_wmma_f32_16x16x16_fp8_fp8: 325 TFLOP/s on the R9700 against 160 for the 16-bit WMMA that the AITER
// a16w4 Triton lane uses). radiance-moe-w4a8-2/, derived from vllm-mxfp4/radiance_mxfp4_fp8.hip's
// folded-scale kernel (LDS staging, v_perm e2m1 -> e4m3 upconvert with the MX block exponent folded in,
// WMMA fragment layout, epilogue), extended to the expert-grouped schedule of the a16w4 lane:
//
//   * grid.y walks AITER's routing block_pid_map (expert | block << 16, -1 = padding), grid.x the N blocks;
//     rows of a block are the sorted (expert-grouped) rows start_m[e] + block*BM + r, r < hist[e];
//   * w13 (GATHER + SWIGLU): A rows are gathered from the per-token fp8 activations (gather[s] / topk), the
//     W tile takes the gate row 2j and the up row 2j+1 of the interleaved [E, 2I, H/2] weight into two
//     separate WMMA column tiles of the same wave, so SiLU(gate) * up is lane-local in the epilogue;
//     output bf16 [Mtot, I] in sorted order (what the a16w4 lane writes);
//   * w2: A = the per-row fp8 quantized intermediate (sorted order), gammas folded into the row scale,
//     output bf16 [Mtot, H] in sorted order; AITER's reduce_grouped sums the top-k rows afterwards.
//
// Numerics (W4A8): weights are the checkpoint's e2m1 values, upconverted LOSSLESSLY to e4m3 together with
// their e8m0 block exponent relative to the row's max exponent Wref[e][n]: byte = e2m1 * 2^(6 - d),
// d = Wref - s. The +6 up-shift (top block -> max 6 * 64 = 384 = 0x7C, no NaN encoding) makes every block
// with d <= 14 exact (e4m3 subnormals, which the gfx12 fp8 WMMA honours); on amd/Qwen3.5-35B-A3B-MXFP4 all
// expert blocks with nonzero weights have d <= 11 (results/scale_hist.json). The row factor 2^(Wref-127-6)
// and the per-row activation scale go into the fp32 epilogue. Activations: fp8 e4m3 per row (amax -> 448),
// strictly more precise than the checkpoint's declared MXFP4 (W4A4) activations.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace py = pybind11;

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

// kMagU[d]: the 8 e2m1 magnitudes times 2^(6-d) as e4m3 bytes (RNE, subnormals), 4 per word for v_perm.
// d <= 14 exact; d = 15 rounds; d > 15 clamps to 15 (only all-zero blocks reach it on this checkpoint).
__device__ __constant__ unsigned int kMagU[16][2] = {
  {0x6c686000u, 0x7c787470u}, {0x64605800u, 0x74706c68u}, {0x5c585000u, 0x6c686460u},
  {0x54504800u, 0x64605c58u}, {0x4c484000u, 0x5c585450u}, {0x44403800u, 0x54504c48u},
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u}, {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u}, {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u}, {0x03020100u, 0x0c080604u},
  {0x02010000u, 0x06040302u},
};
#define U_SHIFT 6

#define PAD 8

// sW row r of a SWIGLU block -> physical row of the interleaved w13 ([g0, u0, g1, u1, ...]). Wave wn owns sW
// rows [wn*TN*16, (wn+1)*TN*16): its first TN/2 column tiles are gate rows, the last TN/2 the matching up rows.
template <int TN>
__device__ __forceinline__ int swiglu_phys_row(int r, int jbase) {
  const int wn = r / (TN * 16), t = (r >> 4) % TN, c = r & 15;
  const int half = t / (TN / 2), tt = t % (TN / 2);
  return 2 * (jbase + wn * (TN / 2) * 16 + tt * 16 + c) + half;
}

// BK: K slab per LDS stage (64 or 128). SKIP: 0 = every wave runs every WMMA; 1 = a wave whose rows all lie past
// the expert's last row skips its WMMAs; 2 = the same per 16-row tile (rows are padded up to BM per expert block,
// and at a 2,224-token step more than half of a 128-row tile is padding).
template <int TM, int TN, int WM, int WN, bool SWIGLU, bool GATHER, int BK, int SKIP>
__global__ __launch_bounds__(WM * WN * 32) void radiance_moe_w4a8_gemm(
    const unsigned char *__restrict__ A, const float *__restrict__ As,
    const unsigned char *__restrict__ W, const unsigned char *__restrict__ Ws,
    const unsigned char *__restrict__ Wref, const int *__restrict__ gather,
    const int *__restrict__ hist, const int *__restrict__ offs, const int *__restrict__ pidmap,
    const float *__restrict__ gammas, __bf16 *__restrict__ Y, int N, int K, int topk) {
  constexpr int NT = WM * WN * 32;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int APASS = (BM * BK + NT * 16 - 1) / (NT * 16);
  constexpr int WPASS = (BN * (BK / 2) + NT * 8 - 1) / (NT * 8);
  static_assert(!SWIGLU || TN % 2 == 0, "SWIGLU needs an even TN");
  constexpr int ASTR = BK + PAD, WSTR = BK + PAD;
  __shared__ unsigned char sA[BM * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ float sScale[BM];

  const int ed = pidmap[blockIdx.y];
  if (ed < 0) return;
  const int e = ed & 0xFFFF, blk = ed >> 16;
  const int Me = hist[e], start = offs[e];
  const int m0 = blk * BM;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BN;            // physical weight row block
  const int jbase = n0 / 2;                  // SWIGLU: first intermediate column of the block
  const int KB = K / 32;

  // per-row activation scale (x gamma for the down GEMM), rows clamped to the expert's last row
  for (int r = tid; r < BM; r += NT) {
    const int m = (m0 + r < Me ? m0 + r : Me - 1);
    const int s = start + m;
    const int src = GATHER ? gather[s] / topk : s;
    float v = As[src];
    if (gammas != nullptr) v *= gammas[s];
    sScale[r] = v;
  }

  // loop-invariant staging addresses (clamp, never predicate: rows past the expert repeat its last row)
  const unsigned char *arow[APASS];
  int aoff[APASS];
#pragma unroll
  for (int p = 0; p < APASS; ++p) {
    int idx = p * NT * 16 + tid * 16;
    if (idx >= BM * BK) idx = 0;  // only when APASS does not divide evenly; duplicate of row 0 col 0
    const int r = idx / BK, c = idx % BK;
    const int m = (m0 + r < Me ? m0 + r : Me - 1);
    const int s = start + m;
    const int src = GATHER ? gather[s] / topk : s;
    arow[p] = A + (size_t)src * K + c;
    aoff[p] = r * ASTR + c;
  }
  const unsigned char *wrow[WPASS];
  const unsigned char *srow[WPASS];
  int wrf[WPASS], woff[WPASS];
#pragma unroll
  for (int p = 0; p < WPASS; ++p) {
    int idx = p * NT * 8 + tid * 8;
    if (idx >= BN * (BK / 2)) idx = 0;
    const int r = idx / (BK / 2), c = idx % (BK / 2);
    const int g = SWIGLU ? swiglu_phys_row<TN>(r, jbase) : n0 + r;
    const size_t rowi = (size_t)e * N + g;
    wrow[p] = W + rowi * (K / 2) + c;
    srow[p] = Ws + rowi * KB + (c * 2) / 32;
    wrf[p] = Wref[rowi];
    woff[p] = r * WSTR + c * 2;
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int q = 0; q < 8; ++q) acc[i][j][q] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int p = 0; p < APASS; ++p)
      *(uint4_t *)(&sA[aoff[p]]) = *(const uint4_t *)(arow[p] + k0);
#pragma unroll
    for (int p = 0; p < WPASS; ++p) {
      const unsigned long long packed = *(const unsigned long long *)(wrow[p] + k0 / 2);
      int d = wrf[p] - (int)srow[p][k0 / 32];
      d = d < 0 ? 0 : (d > 15 ? 15 : d);
      const unsigned int t0 = kMagU[d][0], t1 = kMagU[d][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
        outw[2 * h] = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[woff[p]]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
    __syncthreads();
    // valid 16-row tiles of this wave (wave-uniform)
    const int nv = SKIP == 0 ? TM : min(TM, max(0, (Me - m0 - wm * TM * 16 + 15) / 16));
    if (SKIP == 0 || nv > 0) {
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
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        if (SKIP == 2 && i >= nv) break;
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
      }
    }
    }
    __syncthreads();
  }

  // epilogue: column factor 2^(Wref - 127 - U_SHIFT) per lane column, row scale from LDS
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int r = wn * TN * 16 + j * 16 + col;
    const int g = SWIGLU ? swiglu_phys_row<TN>(r, jbase) : n0 + r;
    rf[j] = exp2f((float)((int)Wref[(size_t)e * N + g] - 127 - U_SHIFT));
  }
  if constexpr (SWIGLU) {
    const int NO = N / 2;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int tt = 0; tt < TN / 2; ++tt) {
        const int jc = jbase + wn * (TN / 2) * 16 + tt * 16 + col;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int r = wm * TM * 16 + i * 16 + kb8 + q;
          if (m0 + r < Me) {
            const float a = sScale[r];
            const float gv = acc[i][tt][q] * a * rf[tt];
            const float uv = acc[i][tt + TN / 2][q] * a * rf[tt + TN / 2];
            Y[(size_t)(start + m0 + r) * NO + jc] = (__bf16)(gv / (1.f + __expf(-gv)) * uv);
          }
        }
      }
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = n0 + wn * TN * 16 + j * 16 + col;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int r = wm * TM * 16 + i * 16 + kb8 + q;
          if (m0 + r < Me) Y[(size_t)(start + m0 + r) * N + n] = (__bf16)(acc[i][j][q] * sScale[r] * rf[j]);
        }
      }
  }
}

// ------------------------------------------------------------------ per-row fp8 activation quant
// bf16 [rows, K] (row stride ldx) -> e4m3 [rows, K] + fp32 scale [rows] (amax / 448, floor 1/(448*512) as
// in the dense lane). TPR threads per row, 8 elements (16 B) per thread per pass.
template <int TPR>
__global__ __launch_bounds__(256) void radiance_moe_quant_rows(const __bf16 *__restrict__ X, int ldx,
                                                               unsigned char *__restrict__ Q, float *__restrict__ S,
                                                               int rows, int K) {
  constexpr int RPB = 256 / TPR, WPR = TPR / 32;
  __shared__ float red[RPB][WPR > 0 ? WPR : 1];
  const int tid = threadIdx.x, rl = tid / TPR, t = tid % TPR;
  const int row = blockIdx.x * RPB + rl;
  const bool ok = row < rows;
  const __bf16 *x = X + (size_t)(ok ? row : 0) * ldx;
  float amax = 0.f;
  for (int c = t * 8; c < K; c += TPR * 8) {
    const uint4_t v = *(const uint4_t *)(x + c);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      amax = fmaxf(amax, fabsf(__uint_as_float(v[q] << 16)));
      amax = fmaxf(amax, fabsf(__uint_as_float(v[q] & 0xFFFF0000u)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
  if constexpr (WPR > 1) {
    if ((tid & 31) == 0) red[rl][t / 32] = amax;
    __syncthreads();
    amax = red[rl][0];
#pragma unroll
    for (int w = 1; w < WPR; ++w) amax = fmaxf(amax, red[rl][w]);
  }
  const float sc = fmaxf(amax * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (!ok) return;
  if (t == 0) S[row] = sc;
  unsigned char *q = Q + (size_t)row * K;
  for (int c = t * 8; c < K; c += TPR * 8) {
    const uint4_t v = *(const uint4_t *)(x + c);
    unsigned int o[2] = {0u, 0u};
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      const float a = fminf(fmaxf(__uint_as_float(v[qq] << 16) * rs, -448.f), 448.f);
      const float b = fminf(fmaxf(__uint_as_float(v[qq] & 0xFFFF0000u) * rs, -448.f), 448.f);
      o[qq >> 1] |= ((unsigned)__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false) & 0xFFFFu) << (16 * (qq & 1));
    }
    *(uint2_t *)(q + c) = uint2_t{o[0], o[1]};
  }
}

// ------------------------------------------------------------------ launchers
static void quant_rows(uintptr_t x, int ldx, uintptr_t q, uintptr_t s, int rows, int K, uintptr_t stream) {
  if (K % 8 != 0 || ldx % 8 != 0) throw std::runtime_error("radiance_moe_w4a8.quant_rows: K and ldx must be multiples of 8");
  if (rows <= 0) return;
  if (K >= 2048) {
    hipLaunchKernelGGL((radiance_moe_quant_rows<256>), dim3(rows), dim3(256), 0, (hipStream_t)stream,
                       (const __bf16 *)x, ldx, (unsigned char *)q, (float *)s, rows, K);
  } else {
    hipLaunchKernelGGL((radiance_moe_quant_rows<64>), dim3((rows + 3) / 4), dim3(256), 0, (hipStream_t)stream,
                       (const __bf16 *)x, ldx, (unsigned char *)q, (float *)s, rows, K);
  }
}

#define LAUNCH(TM_, TN_, WM_, WN_, SW_, GA_, BK_, SK_)                                                        \
  do {                                                                                                         \
    constexpr int BN_ = WN_ * TN_ * 16;                                                                       \
    if (N % BN_ != 0) throw std::runtime_error("radiance_moe_w4a8.gemm: N not a multiple of the tile");     \
    if (K % BK_ != 0) throw std::runtime_error("radiance_moe_w4a8.gemm: K not a multiple of the K slab");   \
    hipLaunchKernelGGL((radiance_moe_w4a8_gemm<TM_, TN_, WM_, WN_, SW_, GA_, BK_, SK_>), dim3(N / BN_, grid_m), \
                       dim3(WM_ * WN_ * 32), 0, (hipStream_t)stream, (const unsigned char *)a, (const float *)as, \
                       (const unsigned char *)w, (const unsigned char *)ws, (const unsigned char *)wref,      \
                       (const int *)gather, (const int *)hist, (const int *)offs, (const int *)pidmap,         \
                       (const float *)gammas, (__bf16 *)y, N, K, topk);                                        \
    return;                                                                                                    \
  } while (0)

// cfg selects the wave grid / tile / K slab / padding skip for a block_m: see CONFIGS in radiance_moe_w4a8.py.
//                 TM TN WM WN        BK  SKIP    tile      waves
#define DISPATCH(SW_, GA_)                                                                                    \
  if (bm == 128) {                                                                                             \
    if (cfg == 0) LAUNCH(2, 2, 4, 2, SW_, GA_, 64, 0);   /* 128 x  64, 8 */                                     \
    if (cfg == 1) LAUNCH(4, 2, 2, 4, SW_, GA_, 64, 0);   /* 128 x 128, 8 */                                     \
    if (cfg == 2) LAUNCH(2, 4, 4, 2, SW_, GA_, 64, 0);   /* 128 x 128, 8 */                                     \
    if (cfg == 3) LAUNCH(4, 2, 2, 4, SW_, GA_, 64, 1);                                                         \
    if (cfg == 4) LAUNCH(4, 2, 2, 4, SW_, GA_, 64, 2);                                                         \
    if (cfg == 5) LAUNCH(2, 4, 4, 2, SW_, GA_, 64, 2);                                                         \
    if (cfg == 6) LAUNCH(2, 2, 4, 2, SW_, GA_, 64, 2);                                                         \
    if (cfg == 7) LAUNCH(4, 2, 2, 4, SW_, GA_, 128, 2);                                                        \
    if (cfg == 8) LAUNCH(2, 4, 4, 2, SW_, GA_, 128, 2);                                                        \
    if (cfg == 9) LAUNCH(4, 4, 2, 4, SW_, GA_, 64, 2);   /* 128 x 256, 8 */                                     \
    if (cfg == 10) LAUNCH(2, 2, 4, 2, SW_, GA_, 128, 2);                                                       \
  } else if (bm == 64) {                                                                                       \
    if (cfg == 0) LAUNCH(1, 2, 4, 2, SW_, GA_, 64, 0);   /*  64 x  64, 8 */                                     \
    if (cfg == 1) LAUNCH(2, 2, 2, 4, SW_, GA_, 64, 0);   /*  64 x 128, 8 */                                     \
    if (cfg == 2) LAUNCH(2, 2, 2, 2, SW_, GA_, 64, 0);   /*  64 x  64, 4 */                                     \
    if (cfg == 3) LAUNCH(1, 2, 4, 2, SW_, GA_, 64, 2);                                                         \
    if (cfg == 4) LAUNCH(2, 2, 2, 4, SW_, GA_, 64, 2);                                                         \
    if (cfg == 5) LAUNCH(1, 4, 4, 2, SW_, GA_, 64, 2);   /*  64 x 128, 8 */                                     \
    if (cfg == 6) LAUNCH(1, 2, 4, 2, SW_, GA_, 128, 2);                                                        \
  } else if (bm == 32) {                                                                                       \
    if (cfg == 0) LAUNCH(1, 2, 2, 2, SW_, GA_, 64, 0);   /*  32 x  64, 4 */                                     \
    if (cfg == 1) LAUNCH(1, 2, 2, 4, SW_, GA_, 64, 0);   /*  32 x 128, 8 */                                     \
    if (cfg == 2) LAUNCH(1, 2, 2, 4, SW_, GA_, 64, 2);                                                         \
    if (cfg == 3) LAUNCH(1, 4, 2, 4, SW_, GA_, 64, 2);   /*  32 x 256, 8 */                                     \
  } else if (bm == 16) {                                                                                       \
    if (cfg == 0) LAUNCH(1, 2, 1, 4, SW_, GA_, 64, 0);   /*  16 x 128, 4 */                                     \
    if (cfg == 1) LAUNCH(1, 2, 1, 2, SW_, GA_, 64, 0);   /*  16 x  64, 2 */                                     \
  }

static void gemm(uintptr_t a, uintptr_t as, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t gather,
                 uintptr_t hist, uintptr_t offs, uintptr_t pidmap, int grid_m, uintptr_t gammas, uintptr_t y,
                 int N, int K, int topk, int bm, bool swiglu, int cfg, uintptr_t stream) {
  if (K % 64 != 0) throw std::runtime_error("radiance_moe_w4a8.gemm: K must be a multiple of 64");
  if (grid_m <= 0) return;
  if (swiglu) {
    if (gather == 0) throw std::runtime_error("radiance_moe_w4a8.gemm: the SWIGLU (w13) launch needs gather");
    DISPATCH(true, true)
  } else {
    if (gather != 0) throw std::runtime_error("radiance_moe_w4a8.gemm: the down launch takes sorted rows (no gather)");
    DISPATCH(false, false)
  }
  throw std::runtime_error("radiance_moe_w4a8.gemm: no config for block_m=" + std::to_string(bm) +
                           " cfg=" + std::to_string(cfg));
}

PYBIND11_MODULE(radiance_moe_w4a8_hip, m) {
  m.def("quant_rows", &quant_rows, "bf16 rows -> fp8 e4m3 rows + fp32 per-row scale");
  m.def("gemm", &gemm, "grouped MXFP4 x FP8 WMMA MoE GEMM (w13 + SiLU*up with gather, or w2 with gammas)");
  m.attr("U_SHIFT") = U_SHIFT;
}
