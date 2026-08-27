// What streaming rate is ACHIEVABLE on this box for the access pattern the decode kernel uses?
//
// 635 GB/s is the cold-stream peak measured once with a dedicated benchmark. It is the right
// number to quote for the hardware, but the wrong denominator for "how close is the kernel",
// because the kernel also reads scales, writes C, and runs under whatever clock the machine is
// at right now. So measure the ceiling in the SAME binary, on the SAME rotated DRAM-resident
// buffers, and report the kernel as a fraction of that.
#include "../radiance_autoround_kernels.h"
#include <vector>
#include <algorithm>

__global__ __launch_bounds__(256) void stream_probe(const unsigned int *__restrict__ W,
                                                    size_t nwords, float *__restrict__ sink) {
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * 256;
  unsigned int a = 0;
  for (; i < nwords; i += stride) a ^= W[i];
  if (a == 0xFFFFFFFFu) sink[0] = 1.f;
}

// Same, but reading 8 bytes per thread -- the width the kernel's staging loop uses.
__global__ __launch_bounds__(256) void stream_probe8(const uint2_t *__restrict__ W,
                                                     size_t n8, float *__restrict__ sink) {
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * 256;
  unsigned int a = 0;
  for (; i < n8; i += stride) { uint2_t v = W[i]; a ^= v[0] ^ v[1]; }
  if (a == 0xFFFFFFFFu) sink[0] = 1.f;
}

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 6;
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}, {5120, 5120, "out"}};
  constexpr int DWN = 8, BND = DWN * 16;
  printf("%-9s %-4s %10s %10s %10s %9s %9s\n", "shape", "M", "kernel GB/s", "probe4 GB/s",
         "probe8 GB/s", "%probe8", "%635");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 64;
    const int nblk = (N + BND - 1) / BND;
    std::vector<unsigned int *> Wa(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wa[c], (size_t)N * kw * 4));
      HIP_CHECK(hipMemset(Wa[c], 0x91 + c, (size_t)N * kw * 4));
    }
    unsigned char *dA; __half *dS; float *dAs, *dP, *dSink; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));  HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2)); HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));         HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)4 * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));        HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    HIP_CHECK(hipMalloc(&dSink, 4));
    const size_t nwords = (size_t)N * kw;
    const double wbytes = (double)N * K / 2.0 + (double)G * N * 2.0;

    // Mirror split_k_for(): gate_up (136 n-blocks) runs DKS=1, down/out (40) run DKS=4.
    // Measuring every shape at DKS=1, as the first version of this file did, starves the 40-block
    // shapes of parallelism and reports a headroom that the shipped kernel does not actually leave.
    const int KS = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : 4;
    hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    for (int M : {5, 8}) {
      const dim3 grid(nblk, 1, KS), block(DWN * 32);
      // Split-K writes then reads DKS*M*N fp32 of partials; that is real traffic and belongs in
      // the byte count, or the wide-split shapes look more efficient than they are.
      const double pbytes = (KS == 1) ? 0.0 : 2.0 * KS * M * N * 4.0;
      float kb = 1e30f, p4 = 1e30f, p8 = 1e30f, ms;
      for (int rep = 0; rep < 3; ++rep) {
#define TIME(BEST, LAUNCH)                                                        \
        for (int it = 0; it < 3; ++it) { LAUNCH; }                                 \
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));           \
        for (int it = 0; it < 20; ++it) { LAUNCH; }                                 \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));          \
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); BEST = std::min(BEST, ms * 50.f);
#define LK1 hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, 1, 1>), grid, block, 0, 0, dA, \
        Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
#define LK4 hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, 4, 1>), grid, block, 0, 0, dA, \
        Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
#define LK do { if (KS == 1) LK1; else LK4; } while (0)
#define LP4 hipLaunchKernelGGL(stream_probe, dim3(2048), dim3(256), 0, 0, Wa[it % NCOPY], \
        nwords, dSink)
#define LP8 hipLaunchKernelGGL(stream_probe8, dim3(2048), dim3(256), 0, 0, \
        (const uint2_t *)Wa[it % NCOPY], nwords / 2, dSink)
        TIME(kb, LK);
        TIME(p4, LP4);
        TIME(p8, LP8);
#undef TIME
#undef LK
#undef LK1
#undef LK4
#undef LP4
#undef LP8
      }
      const double kg = (wbytes + pbytes) / (kb * 1e-6) / 1e9;
      const double g4 = (double)nwords * 4 / (p4 * 1e-6) / 1e9;
      const double g8 = (double)nwords * 4 / (p8 * 1e-6) / 1e9;
      (void)g4;
      printf("%-9s %-4d %-3d %10.1f %10.1f %8.1f%% %8.1f%%\n", sh.name, M, KS, kg, g8,
             100.0 * kg / g8, 100.0 * kg / 635.0);
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    for (int c = 0; c < NCOPY; ++c) hipFree(Wa[c]);
    hipFree(dA); hipFree(dS); hipFree(dAs); hipFree(dP); hipFree(dCnt); hipFree(dC); hipFree(dSink);
  }
  return 0;
}
