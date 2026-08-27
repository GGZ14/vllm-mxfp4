#pragma once
// Activation-side kernels for the escha (EXL3-derived) linear layer.
//
// The GEMM kernels in escha_kernels.h deliberately do NOT do the rotations: EXL3 applies its
// incoherence transform to ACTIVATIONS, not weights, so those are two M x 128 passes that sit
// either side of the matmul. This file is those two passes.
//
// The chain, taken verbatim from the reference runtime's own serving path
// (escha/linear.py::_forward_runtime_had and sglang .../quantization/escha.py::_prefill_recon):
//
//     y = Had128( (x * s_in) * rin ) @ decode(code)  ->  Had128  ->  * rout  ->  * s_out  [+ bias]
//
// Two details that are easy to get wrong and are load-bearing:
//   * rin is a PRE-scale (applied before its Hadamard) and rout is a POST-scale (applied after
//     its own). They are not symmetric.
//   * rin already has the weight scale folded in -- the reference header says "Wscale already
//     folded in - do NOT re-apply". Nothing here re-applies it.
//
// The bias vectors in the checkpoint are deliberately NOT applied: the model card states the
// reference runtime does not apply them and that every published number was produced without
// them, so applying them would diverge from the results being reproduced.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#ifndef ESCHA_HAD
#define ESCHA_HAD 128
#endif
#define ESCHA_E4M3_MAX 448.0f

// One 128-point normalized Walsh-Hadamard, held across 128 lanes of a workgroup in LDS.
// H_128 is the Sylvester matrix, H[i][j] = (-1)^popcount(i&j), so the fast transform is the
// standard butterfly; the reference divides by sqrt(128), and H is symmetric, so applying it to a
// row vector (x @ H) and to a column (H @ x) are the same map.
__device__ __forceinline__ float escha_fwht128(float v, float *lds, int j) {
  lds[j] = v;
  __syncthreads();
#pragma unroll
  for (int s = 1; s < ESCHA_HAD; s <<= 1) {
    const float a = lds[j], b = lds[j ^ s];
    __syncthreads();                       // read both halves before either is overwritten
    lds[j] = (j & s) ? (b - a) : (a + b);
    __syncthreads();
  }
  const float r = lds[j] * 0.08838834764831845f;   // 1/sqrt(128)
  __syncthreads();
  return r;
}

// x [M, IC] bf16  ->  A [M, IC] e4m3 + As [M] fp32, with the pre-rotation applied.
//
// PARALLELISED OVER K-BLOCKS, not just tokens. The obvious shape -- one workgroup per token,
// looping the IC/128 blocks -- is fine at prefill and useless at decode, where M is 1-8: it lights
// up 8 of 64 CUs and walks 40-136 Hadamard blocks back to back with a barrier between each.
//
// The row's amax is only known once the whole row is transformed, so this is two launches. It
// still transforms twice rather than spilling a temporary: at IC=17408 a bf16 temp is 285 MB of
// real bandwidth at M=8192, against 7 adds per element to redo it.
__global__ __launch_bounds__(ESCHA_HAD) void escha_pre_amax(
    const __bf16 *__restrict__ x, const float *__restrict__ s_in, const __half *__restrict__ rin,
    float *__restrict__ amax, int M, int IC) {
  __shared__ float lds[ESCHA_HAD];
  const int m = blockIdx.y, j = threadIdx.x, k = blockIdx.x * ESCHA_HAD + j;
  if (m >= M || k >= IC) return;
  const float v = (float)x[(size_t)m * IC + k] * s_in[k] * (float)rin[k];
  float a = fabsf(escha_fwht128(v, lds, j));
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
  if ((j & 31) == 0) lds[j >> 5] = a;
  __syncthreads();
  if (j == 0) {
    float b = lds[0];
#pragma unroll
    for (int i = 1; i < ESCHA_HAD / 32; ++i) b = fmaxf(b, lds[i]);
    // Magnitudes are non-negative, so their float bits order the same as unsigned integers and
    // the integer atomicMax stands in for the float one gfx1201 does not provide.
    atomicMax((unsigned int *)&amax[m], __float_as_uint(b));
  }
}

__global__ __launch_bounds__(ESCHA_HAD) void escha_pre_quant(
    const __bf16 *__restrict__ x, const float *__restrict__ s_in, const __half *__restrict__ rin,
    const float *__restrict__ amax, unsigned char *__restrict__ A, float *__restrict__ As,
    int M, int IC) {
  __shared__ float lds[ESCHA_HAD];
  const int m = blockIdx.y, j = threadIdx.x, k = blockIdx.x * ESCHA_HAD + j;
  if (m >= M || k >= IC) return;
  // A row of exact zeros must not produce a zero scale: the GEMM epilogue multiplies by it.
  const float a = amax[m] > 0.f ? amax[m] : 1.f;
  if (blockIdx.x == 0 && j == 0) As[m] = a / ESCHA_E4M3_MAX;
  const float v = (float)x[(size_t)m * IC + k] * s_in[k] * (float)rin[k];
  const float h = escha_fwht128(v, lds, j) * (ESCHA_E4M3_MAX / a);
  A[(size_t)m * IC + k] = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(h, h, 0u, false) & 0xFFu);
}

// y [M, OC] bf16 (GEMM output, per-token scale already applied by the GEMM epilogue)
//   -> Had128 -> * rout -> * s_out, written back in place or to `out`.
// Single pass: no scale has to be discovered here.
__global__ __launch_bounds__(ESCHA_HAD) void escha_post_rot(
    const __bf16 *__restrict__ y, const __half *__restrict__ rout,
    const float *__restrict__ s_out, __bf16 *__restrict__ out, int M, int OC) {
  __shared__ float lds[ESCHA_HAD];
  const int m = blockIdx.y, j = threadIdx.x, k = blockIdx.x * ESCHA_HAD + j;
  if (m >= M || k >= OC) return;
  const float h = escha_fwht128((float)y[(size_t)m * OC + k], lds, j);
  out[(size_t)m * OC + k] = (__bf16)(h * (float)rout[k] * s_out[k]);
}
