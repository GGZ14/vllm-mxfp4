#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
#define AR_NEG_LO 0xCACCCED0u
#define AR_NEG_HI 0xB8C0C4C8u
#define AR_POS_LO 0x44403800u
#define AR_POS_HI 0x4E4C4A48u

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
__device__ __forceinline__ uint2_t ar_unpack8(unsigned int wv) {
  const unsigned int be = ar_lut4(wv & 0x0F0F0F0Fu);
  const unsigned int bo = ar_lut4((wv >> 4) & 0x0F0F0F0Fu);
  return uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                 __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}
__global__ void k(const unsigned int *in, uint2_t *out, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = ar_unpack8(in[i]);
}
int main() {
  const unsigned char LUT[16] = {0xD0,0xCE,0xCC,0xCA,0xC8,0xC4,0xC0,0xB8,
                                 0x00,0x38,0x40,0x44,0x48,0x4A,0x4C,0x4E};
  const int N = 4096;
  unsigned int *hin = (unsigned int *)malloc(N * 4);
  for (int i = 0; i < N; ++i) hin[i] = ((unsigned)rand() << 16) ^ (unsigned)rand();
  unsigned int *din; uint2_t *dout;
  hipMalloc(&din, N * 4); hipMalloc(&dout, N * 8);
  hipMemcpy(din, hin, N * 4, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k, dim3((N + 255) / 256), dim3(256), 0, 0, din, dout, N);
  unsigned char *hout = (unsigned char *)malloc(N * 8);
  hipMemcpy(hout, dout, N * 8, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < N && bad < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      const int code = (hin[i] >> (4 * j)) & 0xF;
      const unsigned char got = hout[i * 8 + j];
      if (got != LUT[code]) {
        printf("  word=0x%08X nibble %d code=%2d got=0x%02X want=0x%02X\n",
               hin[i], j, code, got, LUT[code]);
        ++bad;
      }
    }
  printf("ar_unpack8 mismatches (first 8 shown): %d\n", bad);
  return bad != 0;
}
