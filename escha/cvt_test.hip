// Does __builtin_amdgcn_cvt_pk_fp8_f32 on gfx1201 agree with the reference's e4m3 rounding,
// and is it OCP e4m3 (bias 7) or the fnuz variant (bias 8)?
#include "escha_kernels.h"
#include <vector>
#include <cstdio>
__global__ void k(const float *in, unsigned char *out, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i * 2 + 1 < n) {
    unsigned int p = escha_pk_e4m3(in[i * 2], in[i * 2 + 1]);
    out[i * 2] = p & 0xFF; out[i * 2 + 1] = (p >> 8) & 0xFF;
  }
}
int main() {
  const int n = 4096;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = -4.0f + 8.0f * i / (n - 1);
  float *d; unsigned char *o;
  HIP_CHECK(hipMalloc(&d, n * 4)); HIP_CHECK(hipMalloc(&o, n));
  HIP_CHECK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(k, dim3((n / 2 + 255) / 256), dim3(256), 0, 0, d, o, n);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned char> b(n);
  HIP_CHECK(hipMemcpy(b.data(), o, n, hipMemcpyDeviceToHost));
  FILE *f = fopen("cvt_out.bin", "wb");
  fwrite(h.data(), 4, n, f); fwrite(b.data(), 1, n, f); fclose(f);
  printf("wrote cvt_out.bin: %d (float, byte) pairs\n", n);
  printf("  sample: %.4f -> 0x%02X   %.4f -> 0x%02X   %.4f -> 0x%02X\n",
         h[0], b[0], h[n/2], b[n/2], h[n-1], b[n-1]);
  return 0;
}
