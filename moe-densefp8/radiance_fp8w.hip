// radiance_fp8w.hip - skinny W8A16 GEMM for gfx1201 (R9700): C[M,N] = (A[M,K] @ dequant(W8[N,K])^T),
// A bf16, W8 OCP e4m3fn with one fp32 scale per output row (per-output-channel), C bf16, fp32 accumulate.
// moe-densefp8/: the MoE lane's bf16 dense layers (excluded from MXFP4 by Quark) stored as fp8 per-channel.
//
// Same decomposition as libr4d's r4d_gemm_bf16_nt_m16 (v0.5.0): a block is WV output columns x SK k-splits waves;
// each wave owns one output column over one contiguous K slice and keeps MT accumulators (MT = M, compile-time);
// the SK partials are reduced in LDS in a fixed order (deterministic), so there is no second kernel. Differences:
//  * the weight row is fp8: one 128-bit load is 16 weights (vs 8 bf16), so a wave step covers 512 of K;
//  * the 16 bytes are expanded with the gfx12 hardware converter v_cvt_f32_fp8 (exact) and packed to bf16
//    pairs with v_perm (fp8 has <= 4 significant bits, so the f32 -> bf16 truncation is exact too); the product
//    is v_dot2_f32_bf16 against the bf16 activation pairs, exactly as the bf16 kernel does it;
//  * the per-row scale is applied once in the epilogue.
// M <= 16. K % (512 * SK) == 0. Up to 8 weight loads per wave are issued up front (Ks <= 4096).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#define WAVE 32
#define FP8_MAX_M 16
typedef int int4v __attribute__((ext_vector_type(4)));
typedef __bf16 bf162 __attribute__((ext_vector_type(2)));
typedef float float2v __attribute__((ext_vector_type(2)));

// two fp8 (bytes B, B+1 of w) -> two f32 (exact) -> bf16 pair (exact truncation). The packed converter
// (v_cvt_pk_f32_fp8) returned byte B in BOTH halves through this builtin on gfx1201 (scripts/test_fp8w.py), so the
// scalar one with a constant byte select is used.
template <int B>
static __device__ __forceinline__ bf162 cvt2(unsigned int w) {
  const float f0 = __builtin_amdgcn_cvt_f32_fp8((int)w, B);
  const float f1 = __builtin_amdgcn_cvt_f32_fp8((int)w, B + 1);
  const unsigned int lo = __builtin_bit_cast(unsigned int, f0);
  const unsigned int hh = __builtin_bit_cast(unsigned int, f1);
  const unsigned int p = __builtin_amdgcn_perm(hh, lo, 0x07060302u);
  return __builtin_bit_cast(bf162, p);
}

template <int MT>
__global__ __launch_bounds__(1024) void radiance_fp8w_kernel(const __hip_bfloat16* __restrict__ A,
                                                             const unsigned char* __restrict__ W,
                                                             const float* __restrict__ S,
                                                             __hip_bfloat16* __restrict__ C, int K, int N,
                                                             int WV, int SK, int lda) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int j = blockIdx.x * WV + col_local;
  const int Ks = K / SK, kbase = ksplit * Ks;
  float acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = 0.f;
  if (j < N) {
    const unsigned char* Wj = W + (size_t)j * K + kbase + lane * 16;
    const int niter = Ks / (WAVE * 16);
    int4v wbuf[8];
#pragma unroll
    for (int t = 0; t < 8; t++)
      if (t < niter) wbuf[t] = *(const int4v*)(Wj + t * WAVE * 16);
#pragma unroll
    for (int t = 0; t < 8; t++) {           // static index into wbuf (no scratch); t >= niter is skipped
      if (t >= niter) break;
      bf162 wp[8];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const unsigned int w = (unsigned int)wbuf[t][q];
        wp[2 * q] = cvt2<0>(w);
        wp[2 * q + 1] = cvt2<2>(w);
      }
      const int kk = kbase + lane * 16 + t * WAVE * 16;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int4v a0 = *(const int4v*)(A + (size_t)i * lda + kk);
        const int4v a1 = *(const int4v*)(A + (size_t)i * lda + kk + 8);
        const bf162* ap0 = (const bf162*)&a0;
        const bf162* ap1 = (const bf162*)&a1;
#pragma unroll
        for (int p = 0; p < 4; p++) acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap0[p], wp[p], acc[i], false);
#pragma unroll
        for (int p = 0; p < 4; p++) acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap1[p], wp[4 + p], acc[i], false);
      }
    }
#pragma unroll
    for (int i = 0; i < MT; i++) {
      float v = acc[i];
      for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v;
    }
  }
  __syncthreads();
  if (ksplit == 0 && j < N && lane < MT) {
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col_local * SK + s) * MT + lane];
    C[(size_t)lane * N + j] = (__hip_bfloat16)(v * S[j]);
  }
}

// a: bf16 [M, lda] (row stride lda elements), w: uint8 [N, K], s: fp32 [N], c: bf16 [M, N] contiguous.
static void gemm(uintptr_t a, uintptr_t w, uintptr_t s, uintptr_t c, int M, int K, int N, int WV, int SK, int lda,
                 uintptr_t stream) {
  if (M < 1 || M > FP8_MAX_M) throw std::runtime_error("radiance_fp8w: M must be 1..16");
  if (SK < 1 || K % SK != 0 || (K / SK) % (WAVE * 16) != 0 || (K / SK) > 8 * WAVE * 16)
    throw std::runtime_error("radiance_fp8w: K/SK must be a multiple of 512 and <= 4096");
  if (WV < 1 || WV * SK * WAVE > 1024) throw std::runtime_error("radiance_fp8w: threads/block > 1024");
  dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
  size_t smem = (size_t)WV * SK * M * sizeof(float);
  auto Ap = (const __hip_bfloat16*)a;
  auto Wp = (const unsigned char*)w;
  auto Sp = (const float*)s;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT) hipLaunchKernelGGL(radiance_fp8w_kernel<MT>, grid, block, smem, st, Ap, Wp, Sp, Cp, K, N, WV, SK, lda)
  switch (M) {
    case 1: LAUNCH(1); break;   case 2: LAUNCH(2); break;   case 3: LAUNCH(3); break;   case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;   case 6: LAUNCH(6); break;   case 7: LAUNCH(7); break;   case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;   case 10: LAUNCH(10); break; case 11: LAUNCH(11); break; case 12: LAUNCH(12); break;
    case 13: LAUNCH(13); break; case 14: LAUNCH(14); break; case 15: LAUNCH(15); break; case 16: LAUNCH(16); break;
    default: throw std::runtime_error("radiance_fp8w: M unsupported");
  }
#undef LAUNCH
}

PYBIND11_MODULE(radiance_fp8w, m) {
  m.def("gemm", &gemm, "skinny W8A16 (fp8 e4m3fn per-row scale) GEMM, M <= 16 (gfx1201)");
  m.attr("MAX_M") = FP8_MAX_M;
}
