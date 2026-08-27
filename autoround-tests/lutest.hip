#include <hip/hip_runtime.h>
#include <cstdio>
#define AR_NEG_LO 0xCACCCED0u
#define AR_NEG_HI 0xB8C0C4C8u
#define AR_POS_LO 0x44403800u
#define AR_POS_HI 0x4E4C4A48u// Four codes (one per byte) -> four e4m3 bytes.
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
__global__ void k(unsigned int *out) {
  int c = threadIdx.x;
  if (c < 16) out[c] = ar_lut4((unsigned int)c * 0x01010101u);
}
int main() {
  const unsigned char want[16] = {0xD0,0xCE,0xCC,0xCA,0xC8,0xC4,0xC0,0xB8,
                                  0x00,0x38,0x40,0x44,0x48,0x4A,0x4C,0x4E};
  unsigned int *d; hipMalloc(&d, 64);
  hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, d);
  unsigned int h[16]; hipMemcpy(h, d, 64, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int c = 0; c < 16; ++c) {
    unsigned int w = want[c] * 0x01010101u;
    if (h[c] != w) { bad++; printf("  c=%2d got=0x%08X want=0x%08X\n", c, h[c], w); }
  }
  printf("device LUT mismatches: %d\n", bad);
  return bad != 0;
}
