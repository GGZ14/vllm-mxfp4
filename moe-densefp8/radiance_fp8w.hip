// radiance_fp8w.hip - skinny W8A16 GEMM for gfx1201 (R9700): C[M,N] = (A[M,K] @ dequant(W8[N,K])^T),
// A bf16, W8 OCP e4m3fn with one fp32 scale per output row (per-output-channel), C bf16, fp32 accumulate.
// moe-densefp8/: the MoE lane's bf16 dense layers (excluded from MXFP4 by Quark) stored as fp8 per-channel.
//
// Written for this repo. The work decomposition is the same as libr4d's r4d_gemm_bf16_nt_m16 (v0.5.0); no code is
// copied. The block is (32 lanes, SK K-slices, WV columns): every wave reads one weight row over one contiguous slice
// of K and keeps M fp32 sums (M is a template argument); the slice sums meet in LDS and are added in slice order, so
// the result is deterministic and one launch does the whole GEMM. Specific to fp8:
//  * one 128-bit load is 16 weights (vs 8 bf16), so a wave-wide load covers 512 of K;
//  * the 16 bytes are expanded with the gfx12 hardware converter v_cvt_f32_fp8 (exact) and packed to bf16 pairs with
//    v_perm (fp8 has <= 4 significant bits, so the f32 -> bf16 truncation is exact too); the product is
//    v_dot2_f32_bf16 against the bf16 activation pairs;
//  * the per-row scale is applied once in the epilogue.
// M <= 16. K % (512 * SK) == 0 and K / SK <= 4096: a wave issues all of its weight loads before the first dot.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kLanes = 32;                     // wave32
constexpr int kMaxRows = 16;                   // largest M
constexpr int kVecBytes = 16;                  // one 128-bit load: 16 fp8 weights or 8 bf16 activations
constexpr int kStepK = kLanes * kVecBytes;     // K covered by one wave-wide weight load (512)
constexpr int kMaxSteps = 8;                   // weight loads in flight per wave (w0..w7), so a K slice is <= 4096

using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));

struct Fp8wArgs {
  const __hip_bfloat16* x;  // [M, ldx] activations
  const unsigned char* w;   // [N, K] e4m3fn weights
  const float* scale;       // [N] per-row scales
  __hip_bfloat16* y;        // [M, N] output
  int k, n, ldx;
  int cols, splits;         // = blockDim.z, blockDim.y (passed, so they come from s_load, not the dispatch packet)
};

template <typename T>
__device__ __forceinline__ u32x4 load_b128(const T* p) {
  return *reinterpret_cast<const u32x4*>(p);
}

// Dword e of a 128-bit vector as a bf16 pair. The element is read into a scalar first: __builtin_bit_cast of the
// vector-element lvalue itself read element 0 for every e with the image's hipcc.
__device__ __forceinline__ bf16x2 pair_of(const u32x4 v, int e) {
  const unsigned int d = v[e];
  return __builtin_bit_cast(bf16x2, d);
}

// Butterfly over the wave, distance 16 down to 1; every lane ends with the same total.
__device__ __forceinline__ float wave_total(float v) {
#pragma unroll
  for (int d = kLanes >> 1; d != 0; d >>= 1) v += __shfl_xor(v, d);
  return v;
}

// two fp8 (bytes B, B+1 of w) -> two f32 (exact) -> bf16 pair (exact truncation). The packed converter
// (v_cvt_pk_f32_fp8) returned byte B in BOTH halves through this builtin on gfx1201 (scripts/test_fp8w.py), so the
// scalar one with a constant byte select is used.
template <int B>
__device__ __forceinline__ bf16x2 fp8x2_to_bf16x2(unsigned int w) {
  const unsigned int f0 = __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_f32_fp8((int)w, B));
  const unsigned int f1 = __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_f32_fp8((int)w, B + 1));
  return __builtin_bit_cast(bf16x2, __builtin_amdgcn_perm(f1, f0, 0x07060302u));
}

// One K step of one wave: a 128-bit word of 16 fp8 weights against the matching 16 activations of every row.
// Dot order per row: activations 0-7 (pairs 0-3), then 8-15.
template <int ROWS>
__device__ __forceinline__ void dot_step(float (&sum)[ROWS], const u32x4 wq, const __hip_bfloat16* xs, int ldx) {
  bf16x2 wv[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const unsigned int d = wq[e];
    wv[2 * e] = fp8x2_to_bf16x2<0>(d);
    wv[2 * e + 1] = fp8x2_to_bf16x2<2>(d);
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const u32x4 xlo = load_b128(xs + (size_t)r * ldx);
    const u32x4 xhi = load_b128(xs + (size_t)r * ldx + 8);
    float t = sum[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) t = __builtin_amdgcn_fdot2_f32_bf16(pair_of(xlo, e), wv[e], t, false);
#pragma unroll
    for (int e = 0; e < 4; ++e) t = __builtin_amdgcn_fdot2_f32_bf16(pair_of(xhi, e), wv[4 + e], t, false);
    sum[r] = t;
  }
}

// threadIdx = (lane, K slice, column in block). blockDim.x is one wave, so the slice and the column are per-wave.
template <int ROWS>
__global__ __launch_bounds__(1024) void fp8w_skinny_kernel(const Fp8wArgs p) {
  extern __shared__ float slice_sums[];  // [column in block][row][slice]
  const int lane = threadIdx.x;
  const int slice = threadIdx.y, nslices = p.splits;
  const int cib = threadIdx.z;
  const int col = blockIdx.x * p.cols + cib;
  const int slice_len = p.k / nslices;
  const int k_lane = slice * slice_len + lane * kVecBytes;  // first K this lane touches
  float sum[ROWS] = {};
  if (col < p.n) {
    const unsigned char* wrow = p.w + (size_t)col * p.k + k_lane;
    const __hip_bfloat16* xs = p.x + k_lane;
    const int steps = slice_len / kStepK;  // 1..kMaxSteps
    // Every weight load of the slice is issued before the first dot. Eight named registers rather than an array:
    // an array was kept as one 32-dword vector and copied whole at each guarded load (spills at M = 1..8).
    const u32x4 w0 = load_b128(wrow);
    u32x4 w1 = {}, w2 = {}, w3 = {}, w4 = {}, w5 = {}, w6 = {}, w7 = {};
    if (steps > 1) w1 = load_b128(wrow + 1 * kStepK);
    if (steps > 2) w2 = load_b128(wrow + 2 * kStepK);
    if (steps > 3) w3 = load_b128(wrow + 3 * kStepK);
    if (steps > 4) w4 = load_b128(wrow + 4 * kStepK);
    if (steps > 5) w5 = load_b128(wrow + 5 * kStepK);
    if (steps > 6) w6 = load_b128(wrow + 6 * kStepK);
    if (steps > 7) w7 = load_b128(wrow + 7 * kStepK);
    dot_step<ROWS>(sum, w0, xs, p.ldx);
    if (steps > 1) dot_step<ROWS>(sum, w1, xs + 1 * kStepK, p.ldx);
    if (steps > 2) dot_step<ROWS>(sum, w2, xs + 2 * kStepK, p.ldx);
    if (steps > 3) dot_step<ROWS>(sum, w3, xs + 3 * kStepK, p.ldx);
    if (steps > 4) dot_step<ROWS>(sum, w4, xs + 4 * kStepK, p.ldx);
    if (steps > 5) dot_step<ROWS>(sum, w5, xs + 5 * kStepK, p.ldx);
    if (steps > 6) dot_step<ROWS>(sum, w6, xs + 6 * kStepK, p.ldx);
    if (steps > 7) dot_step<ROWS>(sum, w7, xs + 7 * kStepK, p.ldx);
    float* mine = slice_sums + cib * ROWS * nslices + slice;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float t = wave_total(sum[r]);
      if (lane == 0) mine[r * nslices] = t;
    }
  }
  __syncthreads();
  // slice 0 of each live column: lane r adds row r's slice sums in slice order, scales, stores.
  if (slice != 0 || col >= p.n || lane >= ROWS) return;
  const float* part = slice_sums + (cib * ROWS + lane) * nslices;
  float total = 0.0f;
  for (int s = 0; s < nslices; ++s) total += part[s];
  p.y[(size_t)lane * p.n + col] = (__hip_bfloat16)(total * p.scale[col]);
}

// M is a template argument of the kernel: walk ROWS = 1, 2, ... up to the requested M.
template <int ROWS>
void launch_for_rows(int m, dim3 grid, dim3 block, size_t lds, hipStream_t stream, const Fp8wArgs& args) {
  if constexpr (ROWS > kMaxRows) {
    throw std::runtime_error("radiance_fp8w: M unsupported");
  } else if (m != ROWS) {
    launch_for_rows<ROWS + 1>(m, grid, block, lds, stream, args);
  } else {
    hipLaunchKernelGGL(fp8w_skinny_kernel<ROWS>, grid, block, lds, stream, args);
  }
}

// a: bf16 [M, lda] (row stride lda elements), w: uint8 [N, K], s: fp32 [N], c: bf16 [M, N] contiguous.
// WV output columns per block, SK slices of K per column.
void gemm(uintptr_t a, uintptr_t w, uintptr_t s, uintptr_t c, int M, int K, int N, int WV, int SK, int lda,
          uintptr_t stream) {
  if (M < 1 || M > kMaxRows) throw std::runtime_error("radiance_fp8w: M must be 1..16");
  const bool slices_ok = SK >= 1 && K % SK == 0 && K / SK >= kStepK && (K / SK) % kStepK == 0 &&
                         K / SK <= kMaxSteps * kStepK;
  if (!slices_ok) throw std::runtime_error("radiance_fp8w: K/SK must be a multiple of 512 and <= 4096");
  if (WV < 1 || kLanes * SK * WV > 1024) throw std::runtime_error("radiance_fp8w: threads/block > 1024");
  const Fp8wArgs args{reinterpret_cast<const __hip_bfloat16*>(a), reinterpret_cast<const unsigned char*>(w),
                      reinterpret_cast<const float*>(s), reinterpret_cast<__hip_bfloat16*>(c), K, N, lda, WV, SK};
  const unsigned nblocks = (unsigned)((N + WV - 1) / WV);
  const size_t lds = sizeof(float) * (size_t)(WV * SK * M);
  launch_for_rows<1>(M, dim3(nblocks), dim3(kLanes, SK, WV), lds, reinterpret_cast<hipStream_t>(stream), args);
}

}  // namespace

PYBIND11_MODULE(radiance_fp8w, m) {
  m.def("gemm", &gemm, "skinny W8A16 (fp8 e4m3fn per-row scale) GEMM, M <= 16 (gfx1201)");
  m.attr("MAX_M") = kMaxRows;
}
