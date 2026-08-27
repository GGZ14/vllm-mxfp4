// NOTE (2026-08-26): the OPT bits below are now the SHIPPED behaviour -- they were folded
// into ar_kernels.h / radiance_autoround_kernels.h after they were measured. This file is
// kept to re-measure, and it holds a COPY of the kernel, so it can drift from the shipped
// one. If an A/B here shows no difference between a variant and 'shipped', check that the
// copy still matches before believing it.
// A/B for AutoRound int4 prefill optimisation variants against the shipped kernel and MXFP4.
// Every variant must be BIT-IDENTICAL to the shipped kernel: none of the OPT bits changes the
// arithmetic or its order, so anything other than an exact match is a bug, not a tolerance.
#include "ar_prefill_opt.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>

struct Shape { int N, K; const char *name; };
static const char *OPTNAME[16] = {"base",     "branchless", "whoist",    "bl+wh",
                                  "prefetch", "bl+pf",      "wh+pf",     "bl+wh+pf",
                                  "prio",     "bl+prio",    "wh+prio",   "bl+wh+prio",
                                  "pf+prio",  "bl+pf+prio", "wh+pf+prio", "all"};

#ifndef AR_TNX
#define AR_TNX 2
#endif
template <int OPT>
static void launch(dim3 g, dim3 b, const unsigned char *A, const unsigned int *W, const __half *S,
                   const float *As, __bf16 *C, int M, int N, int K) {
  hipLaunchKernelGGL((ar_prefill_opt<AR_TNX, OPT>), g, b, 0, 0, A, W, S, As, C, M, N, K);
}

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 3;
#ifndef AR_TNX
#define AR_TNX 2
#endif
  constexpr int TN = AR_TNX, AR_BNF = AR_WN * TN * 16;
  const int OPTS[] = {1, 2, 3, 5, 7, 8, 15};

  // ---------------- correctness: bit-exact vs the shipped kernel ----------------
  {
    printf("== correctness (must be bit-identical to shipped) ==\n");
    const int cases[][3] = {{256, 64, 256}, {512, 200, 320}, {300, 64, 5120},
                            {1024, 17408 % 4096 + 64, 640}, {97, 48, 384}};
    bool allok = true;
    for (auto &cs : cases) {
      const int M = cs[0], N = cs[1], K = cs[2], kw = K / 8, G = K / AR_GROUP;
      std::vector<unsigned int> hW((size_t)N * kw);
      std::vector<unsigned char> hA((size_t)M * K);
      std::vector<__half> hS((size_t)G * N);
      std::vector<float> hAs(M);
      srand(1234 + M + N + K);
      for (auto &x : hW) x = ((unsigned)rand() << 17) ^ (unsigned)rand();
      for (auto &x : hA) { int v = rand() & 0xFF; x = (v == 0x7F || v == 0xFF) ? 0x38 : v; }
      for (size_t i = 0; i < hS.size(); ++i) hS[i] = __float2half(0.002f * ((rand() % 200) - 100));
      for (int i = 0; i < M; ++i) hAs[i] = 0.5f + 0.001f * (rand() % 100);

      unsigned int *dW; unsigned char *dA; __half *dS; float *dAs; __bf16 *dC0, *dC1;
      HIP_CHECK(hipMalloc(&dW, hW.size() * 4));  HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dA, hA.size()));      HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dS, hS.size() * 2));  HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dAs, M * 4));         HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dC0, (size_t)M * N * 2)); HIP_CHECK(hipMalloc(&dC1, (size_t)M * N * 2));
      HIP_CHECK(hipMemset(dC0, 0x7F, (size_t)M * N * 2)); HIP_CHECK(hipMemset(dC1, 0x7F, (size_t)M * N * 2));

      const dim3 g((N + AR_BNF - 1) / AR_BNF, (M + AR_BMF - 1) / AR_BMF), b(AR_NTHREADS);
      hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, true>), g, b, 0, 0, dA, dW, dS, dAs, dC0, M, N, K);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<__bf16> h0((size_t)M * N), h1((size_t)M * N);
      HIP_CHECK(hipMemcpy(h0.data(), dC0, h0.size() * 2, hipMemcpyDeviceToHost));
      printf("  M=%-5d N=%-6d K=%-5d ", M, N, K);
      for (int oi = 0; oi < (int)(sizeof(OPTS) / sizeof(int)); ++oi) {
        const int o = OPTS[oi];
        HIP_CHECK(hipMemset(dC1, 0x7F, (size_t)M * N * 2));
        switch (o) {
          case 1: launch<1>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 2: launch<2>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 3: launch<3>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 5: launch<5>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 7: launch<7>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 8: launch<8>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
          case 15: launch<15>(g, b, dA, dW, dS, dAs, dC1, M, N, K); break;
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h1.data(), dC1, h1.size() * 2, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < h0.size(); ++i)
          if (memcmp(&h0[i], &h1[i], 2) != 0) ++bad;
        printf("%s:%s ", OPTNAME[o], bad ? "FAIL" : "ok");
        if (bad) { allok = false; printf("(%zu) ", bad); }
      }
      printf("\n");
      hipFree(dW); hipFree(dA); hipFree(dS); hipFree(dAs); hipFree(dC0); hipFree(dC1);
    }
    printf("  => %s\n\n", allok ? "ALL BIT-IDENTICAL" : "MISMATCH");
    if (!allok) return 1;
  }

  // ---------------- benchmark ----------------
  const Shape pshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
  const int pMs[] = {512, 2048, 4096};
  printf("== prefill, us (best of 3 x 10), NCOPY=%d ==\n", NCOPY);
  printf("%-9s %-6s %9s %9s", "shape", "M", "shipped", "mxfp4");
  for (int o : OPTS) printf(" %10s", OPTNAME[o]);
  printf("  %8s\n", "best/ship");
  for (const Shape &sh : pshapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 4096;
    std::vector<unsigned int *> Wa(NCOPY);
    std::vector<unsigned char *> Wm(NCOPY);
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
      const dim3 ag((N + AR_BNF - 1) / AR_BNF, (M + AR_BMF - 1) / AR_BMF), abk(AR_NTHREADS);
      const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mbk(NTHREADS);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float ship = 1e30f, mx = 1e30f, ms, var[16];
      for (int i = 0; i < 16; ++i) var[i] = 1e30f;
#define TIME(FN, DST) do {                                                            \
        for (int it = 0; it < 2; ++it) { FN(it); }                                      \
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));              \
        for (int it = 0; it < 10; ++it) { FN(it); }                                     \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));             \
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); DST = std::min(DST, ms * 100.f);  \
      } while (0)
      auto Fship = [&](int it) { hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, true>), ag, abk, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto Fmx   = [&](int it) { hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false>), mg, mbk, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dC, M, N, K); };
      auto F1  = [&](int it) { launch<1>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F2  = [&](int it) { launch<2>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F3  = [&](int it) { launch<3>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F5  = [&](int it) { launch<5>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F7  = [&](int it) { launch<7>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F8  = [&](int it) { launch<8>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      auto F15 = [&](int it) { launch<15>(ag, abk, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K); };
      for (int rep = 0; rep < 3; ++rep) {
        TIME(Fship, ship); TIME(Fmx, mx);
        TIME(F1, var[1]); TIME(F2, var[2]); TIME(F3, var[3]); TIME(F5, var[5]);
        TIME(F7, var[7]); TIME(F8, var[8]); TIME(F15, var[15]);
      }
      float best = 1e30f;
      printf("%-9s %-6d %9.1f %9.1f", sh.name, M, ship, mx);
      for (int o : OPTS) { printf(" %10.1f", var[o]); best = std::min(best, var[o]); }
      printf("  %7.3fx\n", best / ship);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
