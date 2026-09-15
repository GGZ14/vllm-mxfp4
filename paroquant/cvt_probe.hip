// Exhaustive gate: hardware v_cvt_pk_fp8_f32 / v_cvt_f32_fp8 on gfx1201 vs the software
// pq_e4m3_encode / pq_e4m3_decode in par_kernels.h, over ALL 2^32 float bit patterns (encode)
// and all 256 codes (decode).
#define PQ_HW_CVT 0
#include "par_kernels.h"
#include <cstdio>
#include <cstring>
#include <vector>

__device__ __forceinline__ unsigned char hw_encode_raw(float v) {
  return (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
}
__device__ __forceinline__ unsigned char hw_encode(float v) {
  // contract of pq_e4m3_encode: NaN -> 0, +-0 -> 0x00, saturate to +-448 (never NaN/inf)
  v = (v == v) ? v : 0.f;
  const float c = fminf(fmaxf(v, -448.f), 448.f) + 0.f;
  return (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(c, c, 0, false);
}
__device__ __forceinline__ float hw_decode(unsigned char b) {
  return __builtin_amdgcn_cvt_f32_fp8((int)b, 0);
}

struct Mis { unsigned u; unsigned char sw, hw; };
__global__ void k_encode(unsigned long long *cnt, Mis *first, int *nfirst, unsigned base) {
  const unsigned long long start = (unsigned long long)base + (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x) * 256ull;
  for (unsigned long long i = 0; i < 256; ++i) {
    const unsigned long long uu = start + i;
    if (uu > 0xFFFFFFFFull) return;
    const unsigned u = (unsigned)uu;
    const float v = __uint_as_float(u);
    const unsigned char sw = pq_e4m3_encode(v), hw = hw_encode(v), raw = hw_encode_raw(v);
    int cat = -1;
    if (sw != hw) {
      const float a = fabsf(v);
      if (v != v) cat = 0;                       // NaN
      else if (a >= 448.f) cat = 1;              // saturation band (incl inf)
      else if (a == 0.f) cat = 2;                // +-0
      else if (a < 0.015625f) cat = 3;           // subnormal result band
      else cat = 4;                              // normal
      atomicAdd(&cnt[cat], 1ull);
      int slot = atomicAdd(nfirst, 1);
      if (slot < 32) first[slot] = {u, sw, hw};
    }
    if (sw != raw) atomicAdd(&cnt[5], 1ull);     // how often the wrapper's clamp/NaN/-0 fix-ups matter
  }
}
__global__ void k_decode(float *sw, float *hw) {
  const int b = threadIdx.x;
  sw[b] = pq_e4m3_decode((unsigned char)b);
  hw[b] = hw_decode((unsigned char)b);
}
int main() {
  unsigned long long *dcnt; Mis *dfirst; int *dn;
  hipMalloc(&dcnt, 8 * 8); hipMalloc(&dfirst, sizeof(Mis) * 32); hipMalloc(&dn, 4);
  hipMemset(dcnt, 0, 64); hipMemset(dn, 0, 4);
  // 2^32 patterns / 256 per thread = 2^24 threads
  const unsigned threads = 1u << 24, tpb = 256;
  hipLaunchKernelGGL(k_encode, dim3(threads / tpb), dim3(tpb), 0, 0, dcnt, dfirst, dn, 0u);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("encode kernel failed: %s\n", hipGetErrorString(e)); return 1; }
  unsigned long long cnt[8]; Mis first[32]; int n;
  hipMemcpy(cnt, dcnt, 64, hipMemcpyDeviceToHost); hipMemcpy(first, dfirst, sizeof(first), hipMemcpyDeviceToHost);
  hipMemcpy(&n, dn, 4, hipMemcpyDeviceToHost);
  printf("encode: 2^32 inputs; mismatches sw vs hw(wrapped): NaN=%llu sat=%llu zero=%llu subnormal=%llu normal=%llu | sw vs raw-hw=%llu\n",
         cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
  for (int i = 0; i < n && i < 32; ++i) { float f; memcpy(&f, &first[i].u, 4); printf("  u=%08x v=%.9g sw=%02x hw=%02x\n", first[i].u, f, first[i].sw, first[i].hw); }
  float *dsw, *dhw; hipMalloc(&dsw, 1024); hipMalloc(&dhw, 1024);
  hipLaunchKernelGGL(k_decode, dim3(1), dim3(256), 0, 0, dsw, dhw);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("decode kernel failed: %s\n", hipGetErrorString(e)); return 1; }
  float sw[256], hw[256]; hipMemcpy(sw, dsw, 1024, hipMemcpyDeviceToHost); hipMemcpy(hw, dhw, 1024, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int b = 0; b < 256; ++b) {
    unsigned a, c; memcpy(&a, &sw[b], 4); memcpy(&c, &hw[b], 4);
    if (a != c) { if (bad < 8) printf("  decode %02x: sw=%.9g (%08x) hw=%.9g (%08x)\n", b, sw[b], a, hw[b], c); ++bad; }
  }
  printf("decode: %d/256 codes differ bitwise\n", bad);
  const bool ok = cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] == 0 && bad == 0;
  printf("RESULT %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}
