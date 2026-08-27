#include <hip/hip_runtime.h>
#include <cstdio>
// Ground-truth map of v_perm_b32 on gfx1201. Distinctive pool bytes so each selector value
// reveals exactly which byte it picked; a second pass with known MSBs reveals the sign behaviour.
__global__ void probe(unsigned int *out) {
  const int s = threadIdx.x;              // selector value under test, 0..15
  if (s >= 16) return;
  const unsigned int sel = (unsigned int)s * 0x01010101u;
  out[s]      = __builtin_amdgcn_perm(0x03020100u, 0x07060504u, sel);  // which byte?
  out[16 + s] = __builtin_amdgcn_perm(0x80000000u, 0x00000080u, sel);  // MSB set at S0.b3, S1.b0
}
int main() {
  unsigned int *d; hipMalloc(&d, 32 * 4);
  hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, d);
  unsigned int h[32]; hipMemcpy(h, d, 32 * 4, hipMemcpyDeviceToHost);
  printf("  perm(S0=0x03020100, S1=0x07060504, sel) and perm(S0=0x80000000, S1=0x00000080, sel)\n");
  for (int s = 0; s < 16; ++s)
    printf("   sel=%2d -> byte 0x%02X   msbtest -> 0x%02X\n", s, h[s] & 0xFF, h[16 + s] & 0xFF);
  return 0;
}
