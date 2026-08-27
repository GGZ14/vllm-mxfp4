// NOTE (2026-08-26): the OPT bits below are now the SHIPPED behaviour -- they were folded
// into ar_kernels.h / radiance_autoround_kernels.h after they were measured. This file is
// kept to re-measure, and it holds a COPY of the kernel, so it can drift from the shipped
// one. If an A/B here shows no difference between a variant and 'shipped', check that the
// copy still matches before believing it.
// One binary, four prefill tile shapes, interleaved. All run the bl+wh (OPT=3) kernel; MXFP4 is
// the fixed reference so a drifting machine shows up as a drifting reference, not as a tile win.
#include "ar_prefill_opt.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
struct Shape { int N, K; const char *name; };
int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 3;
  const Shape pshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
  const int pMs[] = {512, 2048, 4096};
  printf("%-9s %-6s %9s %9s %9s %9s %9s %9s\n", "shape", "M", "mxfp4", "ship TN2",
         "ship TN4", "opt tn2", "opt tn4", "wn4tn4");
  for (const Shape &sh : pshapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 4096;
    std::vector<unsigned int *> Wa(NCOPY); std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wa[c], (size_t)N * kw * 4)); HIP_CHECK(hipMemset(Wa[c], 0x91 + c, (size_t)N * kw * 4));
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2));  HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    unsigned char *dA, *dWs, *dWref; __half *dS; float *dAs; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));      HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));     HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    for (int M : pMs) {
      const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mbk(NTHREADS);
      const int by = (M + AR_BMF - 1) / AR_BMF;
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float t[6]; for (int i = 0; i < 6; ++i) t[i] = 1e30f; float ms;
      auto F0 = [&](int it) { hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false>), mg, mbk, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dC, M, N, K); };
      auto F1 = [&](int it) { hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<2, true>), dim3((N + 63) / 64, by), dim3(256), 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      // F2 is now the SHIPPED kernel at TN=4 -- the path the launcher actually takes above
      // RADIANCE_AR_TN4_MIN_M. The ar_prefill_opt copies below are kept only as the historical
      // reference; they lack the epilogue fast path that is now shipped.
      auto F2 = [&](int it) { hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<4, true>), dim3((N + 127) / 128, by), dim3(256), 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F3 = [&](int it) { hipLaunchKernelGGL((ar_prefill_opt<2, 3, 2>), dim3((N + 63) / 64, by), dim3(256), 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F4 = [&](int it) { hipLaunchKernelGGL((ar_prefill_opt<4, 3, 2>), dim3((N + 127) / 128, by), dim3(256), 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F5 = [&](int it) { hipLaunchKernelGGL((ar_prefill_opt<4, 3, 4>), dim3((N + 255) / 256, by), dim3(512), 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
#define TM(F, D) do { for (int it = 0; it < 2; ++it) F(it); HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it = 0; it < 10; ++it) F(it); \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); D = std::min(D, ms * 100.f); } while (0)
      for (int rep = 0; rep < 4; ++rep) {
        TM(F0, t[0]); TM(F1, t[1]); TM(F2, t[2]); TM(F3, t[3]); TM(F4, t[4]); TM(F5, t[5]);
      }
      printf("%-9s %-6d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", sh.name, M, t[0], t[1], t[2], t[3], t[4], t[5]);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
