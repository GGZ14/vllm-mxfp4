#pragma once
// Activation-side kernels for the escha (EXL3-derived) linear layer.
//
// The GEMM kernels in escha_kernels.h deliberately do NOT do the rotations: EXL3 applies its
// incoherence transform to ACTIVATIONS, so those are two M x 128 passes either side of the matmul.
//
// The chain, taken verbatim from the reference runtime's serving path (escha/linear.py::
// _forward_runtime_had and sglang .../quantization/escha.py::_prefill_recon):
//
//     y = Had128( (x * s_in) * rin ) @ decode(code)  ->  Had128  ->  * rout  ->  * s_out
//
// Two details that are load-bearing: rin is a PRE-scale (before its Hadamard) and rout a
// POST-scale (after its own) -- they are not symmetric; and rin already has the weight scale
// folded in, so nothing here re-applies it. The checkpoint's bias vectors are deliberately NOT
// applied: the model card states the reference runtime does not apply them.
//
// WHY THE TRANSFORM IS IN REGISTERS. The obvious 128-point FWHT keeps the block in LDS and
// butterflies it in place, which needs a barrier before and after every stage -- 14 __syncthreads
// per 128 values. Measured on the real prefill shapes that put the two rotation passes at 44-55%
// of the whole layer's time, against a GEMM that is at parity with MXFP4. So the transform is done
// entirely in registers instead: each lane holds 4 CONSECUTIVE elements, which makes the two
// smallest butterfly strides register-local and the remaining five pure cross-lane shuffles.
// No LDS, no barriers, and the loads stay coalesced at 8 bytes per lane.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#ifndef ESCHA_HAD
#define ESCHA_HAD 128
#endif
#define ESCHA_E4M3_MAX 448.0f
#define ESCHA_ACT_WAVES 8                       // 128-blocks handled per workgroup
#define ESCHA_ACT_THREADS (ESCHA_ACT_WAVES * 32)

// Normalized 128-point Walsh-Hadamard across one wave, 4 elements per lane.
//
// Lane l register r holds element e = l*4 + r, so a butterfly at stride s pairs e with e^s:
//   s = 1, 2   flip a register bit  -> register-local, no communication at all
//   s = 4..64  flip a lane bit      -> one v_permlane/ds_swizzle via __shfl_xor
// H_128 is symmetric, so this is equally the row-vector transform x @ H the reference applies.
__device__ __forceinline__ void escha_fwht128_x4(float v[4], int lane) {
#pragma unroll
  for (int r = 0; r < 4; r += 2) {            // stride 1: (r0,r1) and (r2,r3)
    const float a = v[r], b = v[r + 1];
    v[r] = a + b; v[r + 1] = a - b;
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {               // stride 2: (r0,r2) and (r1,r3)
    const float a = v[r], b = v[r + 2];
    v[r] = a + b; v[r + 2] = a - b;
  }
#pragma unroll
  for (int t = 1; t < 32; t <<= 1) {          // strides 4,8,16,32,64 -> lane xor 1,2,4,8,16
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float o = __shfl_xor(v[r], t, 32);
      v[r] = (lane & t) ? (o - v[r]) : (v[r] + o);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] *= 0.08838834764831845f;   // 1/sqrt(128)
}

// Load 4 consecutive pre-scaled activations for this lane's slot.
__device__ __forceinline__ void escha_load4(const __bf16 *__restrict__ x, const float *__restrict__ s_in,
                                            const __half *__restrict__ rin, int base, float v[4]) {
#pragma unroll
  for (int r = 0; r < 4; ++r)
    v[r] = (float)x[base + r] * s_in[base + r] * (float)rin[base + r];
}

// x [M, IC] bf16 -> per-token amax of the rotated row. Two launches are unavoidable: the scale is
// a reduction over the whole row, so nothing can be quantized until it is known. The transform is
// redone in the second pass rather than spilling a temporary -- at IC=17408 that temp is 285 MB of
// real bandwidth at M=8192, against a register-resident transform to redo.
__global__ __launch_bounds__(ESCHA_ACT_THREADS) void escha_pre_amax(
    const __bf16 *__restrict__ x, const float *__restrict__ s_in, const __half *__restrict__ rin,
    float *__restrict__ amax, int M, int IC) {
  __shared__ float wmax[ESCHA_ACT_WAVES];
  const int m = blockIdx.y, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int blk = blockIdx.x * ESCHA_ACT_WAVES + wave;
  const int base = blk * ESCHA_HAD + lane * 4;
  float a = 0.f;
  if (m < M && base < IC) {
    float v[4];
    escha_load4(x + (size_t)m * IC, s_in, rin, base, v);
    escha_fwht128_x4(v, lane);
#pragma unroll
    for (int r = 0; r < 4; ++r) a = fmaxf(a, fabsf(v[r]));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
  if (lane == 0) wmax[wave] = a;
  __syncthreads();                       // the ONLY barrier, and once per workgroup, not per stage
  if (threadIdx.x == 0 && m < M) {
    float b = wmax[0];
#pragma unroll
    for (int i = 1; i < ESCHA_ACT_WAVES; ++i) b = fmaxf(b, wmax[i]);
    // Magnitudes are non-negative, so their float bits order as unsigned integers and the integer
    // atomicMax stands in for the float one gfx1201 does not provide.
    if (b > 0.f) atomicMax((unsigned int *)&amax[m], __float_as_uint(b));
  }
}

__global__ __launch_bounds__(ESCHA_ACT_THREADS) void escha_pre_quant(
    const __bf16 *__restrict__ x, const float *__restrict__ s_in, const __half *__restrict__ rin,
    const float *__restrict__ amax, unsigned char *__restrict__ A, float *__restrict__ As,
    int M, int IC) {
  const int m = blockIdx.y, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int blk = blockIdx.x * ESCHA_ACT_WAVES + wave;
  const int base = blk * ESCHA_HAD + lane * 4;
  if (m >= M || base >= IC) return;
  // A row of exact zeros must not produce a zero scale: the GEMM epilogue multiplies by it.
  const float a = amax[m] > 0.f ? amax[m] : 1.f;
  if (blockIdx.x == 0 && threadIdx.x == 0) As[m] = a / ESCHA_E4M3_MAX;
  float v[4];
  escha_load4(x + (size_t)m * IC, s_in, rin, base, v);
  escha_fwht128_x4(v, lane);
  const float inv = ESCHA_E4M3_MAX / a;
  // Four e4m3 bytes leave as one dword: base is 4-aligned because lanes own 4 consecutive slots.
  const unsigned int lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * inv, v[1] * inv, 0u, false);
  const unsigned int hi = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * inv, v[3] * inv, 0u, false);
  *(unsigned int *)(A + (size_t)m * IC + base) = (lo & 0xFFFFu) | (hi << 16);
}

// y [M, OC] bf16 (per-token scale already applied by the GEMM epilogue)
//   -> Had128 -> * rout -> * s_out. Single pass: no scale to discover, so no barrier at all.
// `ldo`/`col0` let a shard write straight into its slice of the layer's final output. A merged
// module runs one chain per source tensor (gate is K=2, up is K=3, and their rin differ), and
// concatenating the per-shard results afterwards costs a full output-sized read plus write --
// 570 MB of traffic on gate_up at M=8192. Writing in place removes it entirely.
__global__ __launch_bounds__(ESCHA_ACT_THREADS) void escha_post_rot(
    const __bf16 *__restrict__ y, const __half *__restrict__ rout,
    const float *__restrict__ s_out, __bf16 *__restrict__ out, int M, int OC,
    int ldo, int col0) {
  const int m = blockIdx.y, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int blk = blockIdx.x * ESCHA_ACT_WAVES + wave;
  const int base = blk * ESCHA_HAD + lane * 4;
  if (m >= M || base >= OC) return;
  const __bf16 *__restrict__ yr = y + (size_t)m * OC;
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = (float)yr[base + r];
  escha_fwht128_x4(v, lane);
  __bf16 *__restrict__ o = out + (size_t)m * ldo + col0;
#pragma unroll
  for (int r = 0; r < 4; ++r)
    o[base + r] = (__bf16)(v[r] * (float)rout[base + r] * s_out[base + r]);
}
