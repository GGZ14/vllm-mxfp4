// NOTE (2026-08-26): the OPT bits below are now the SHIPPED behaviour -- they were folded
// into ar_kernels.h / radiance_autoround_kernels.h after they were measured. This file is
// kept to re-measure, and it holds a COPY of the kernel, so it can drift from the shipped
// one. If an A/B here shows no difference between a variant and 'shipped', check that the
// copy still matches before believing it.
// A/B for AutoRound int4 DECODE optimisation variants against the shipped kernel and MXFP4.
#include "ar_decode_opt.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
#include <cmath>

struct Shape { int N, K; const char *name; };

// OCP e4m3 -> double. 0x7F/0xFF are NaN and are excluded from the generated data.
static double e4m3(unsigned char b) {
  const int sgn = (b >> 7) & 1, ex = (b >> 3) & 0xF, ma = b & 7;
  double v = ex ? std::ldexp(1.0 + ma / 8.0, ex - 7) : std::ldexp(ma / 8.0, -6);
  return sgn ? -v : v;
}

// Reference in fp64, in the kernel's own summation shape: fold the group scale once per group.
// Split-K changes only the order in which whole groups are added, so both DKS=1 and DKS=4 are
// compared against this same reference and the question is which lands closer, not whether they
// agree with each other.
static void ref_gemm(const std::vector<unsigned char> &A, const std::vector<unsigned int> &W,
                     const std::vector<__half> &S, const std::vector<float> &As,
                     std::vector<double> &R, int M, int N, int K) {
  const int kw = K / 8, G = K / AR_GROUP;
  R.assign((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double tot = 0.0;
      for (int g = 0; g < G; ++g) {
        double part = 0.0;
        for (int k = g * AR_GROUP; k < (g + 1) * AR_GROUP; ++k) {
          const unsigned int word = W[(size_t)n * kw + k / 8];
          const int code = (int)((word >> (4 * (k % 8))) & 0xF);
          part += e4m3(A[(size_t)m * K + k]) * (double)(code - 8);
        }
        tot += part * (double)__half2float(S[(size_t)g * N + n]);
      }
      R[(size_t)m * N + n] = tot * (double)As[m];
    }
}
constexpr int DWN = 8, BND = DWN * 16;

#define DISPATCH(TM_, EXPR) do { if (tm == 1) { constexpr int TM_ = 1; EXPR; } \
  else if (tm == 2) { constexpr int TM_ = 2; EXPR; } \
  else if (tm == 3) { constexpr int TM_ = 3; EXPR; } \
  else { constexpr int TM_ = 4; EXPR; } } while (0)

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 3;

  // ---------------- correctness ----------------
  {
    printf("== correctness ==\n");
    bool allok = true;
    const int cases[][3] = {{5, 256, 512}, {8, 200, 640}, {16, 128, 5120},
                            {40, 1088, 1024}, {64, 48, 384}, {33, 640, 768}};
    for (auto &cs : cases) {
      const int M = cs[0], N = cs[1], K = cs[2], kw = K / 8, G = K / AR_GROUP;
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE, nblk = (N + BND - 1) / BND;
      std::vector<unsigned int> hW((size_t)N * kw);
      std::vector<unsigned char> hA((size_t)M * K);
      std::vector<__half> hS((size_t)G * N);
      std::vector<float> hAs(M);
      srand(99 + M + N + K);
      for (auto &x : hW) x = ((unsigned)rand() << 17) ^ (unsigned)rand();
      for (auto &x : hA) { int v = rand() & 0xFF; x = (v == 0x7F || v == 0xFF) ? 0x38 : v; }
      for (size_t i = 0; i < hS.size(); ++i) hS[i] = __float2half(0.002f * ((rand() % 200) - 100));
      for (int i = 0; i < M; ++i) hAs[i] = 0.5f + 0.001f * (rand() % 100);
      unsigned int *dW; unsigned char *dA; __half *dS; float *dAs, *dP; int *dCnt; __bf16 *dC0, *dC1, *dC2;
      HIP_CHECK(hipMalloc(&dW, hW.size() * 4)); HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dA, hA.size()));     HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dS, hS.size() * 2)); HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dAs, M * 4));        HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * N * 4));
      HIP_CHECK(hipMalloc(&dCnt, nblk * 4));    HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
      HIP_CHECK(hipMalloc(&dC0, (size_t)M * N * 2)); HIP_CHECK(hipMalloc(&dC1, (size_t)M * N * 2));
      HIP_CHECK(hipMalloc(&dC2, (size_t)M * N * 2));
      HIP_CHECK(hipMemset(dC0, 0x7F, (size_t)M * N * 2)); HIP_CHECK(hipMemset(dC1, 0x7F, (size_t)M * N * 2));
      HIP_CHECK(hipMemset(dC2, 0x7F, (size_t)M * N * 2));
      const dim3 g4(nblk, 1, 4), g1(nblk, 1, 1), b(DWN * 32);
      DISPATCH(TMV, hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, 4, TMV, true>), g4, b, 0, 0, dA, dW, dS, dAs, dP, dCnt, dC0, M, N, K));
      DISPATCH(TMV, hipLaunchKernelGGL((ar_decode_opt<DWN, 4, TMV, 1>), g4, b, 0, 0, dA, dW, dS, dAs, dP, dCnt, dC1, M, N, K));
      DISPATCH(TMV, hipLaunchKernelGGL((ar_decode_opt<DWN, 1, TMV, 3>), g1, b, 0, 0, dA, dW, dS, dAs, dP, dCnt, dC2, M, N, K));
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<__bf16> h0((size_t)M * N), h1(h0.size()), h2(h0.size());
      HIP_CHECK(hipMemcpy(h0.data(), dC0, h0.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h1.data(), dC1, h1.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h2.data(), dC2, h2.size() * 2, hipMemcpyDeviceToHost));
      std::vector<double> R;
      ref_gemm(hA, hW, hS, hAs, R, M, N, K);
      size_t bad = 0; double e4 = 0, e1 = 0;
      for (size_t i = 0; i < h0.size(); ++i) {
        if (memcmp(&h0[i], &h1[i], 2) != 0) ++bad;
        const double r = R[i], den = std::max(1e-4, std::fabs(r));
        e4 = std::max(e4, std::fabs((double)(float)h0[i] - r) / den);
        e1 = std::max(e1, std::fabs((double)(float)h2[i] - r) / den);
      }
      // bf16 carries 8 mantissa bits, so one ULP is 2^-8 = 3.9e-3 relative. Anything at or under
      // a couple of ULP is output rounding, not kernel error -- which is why the gate is "DKS=1
      // is no worse than the shipped DKS=4", not an absolute tolerance.
      const bool ok = bad == 0 && e1 <= std::max(e4 * 1.5, 8e-3);
      printf("  M=%-3d N=%-5d K=%-5d  branchless(DKS=4) vs shipped: %-14s  vs fp64: DKS4 %.2e  DKS1 %.2e  %s\n",
             M, N, K, bad ? "FAIL" : "bit-identical", e4, e1, ok ? "ok" : "FAIL");
      if (!ok) allok = false;
      hipFree(dW); hipFree(dA); hipFree(dS); hipFree(dAs); hipFree(dP); hipFree(dCnt);
      hipFree(dC0); hipFree(dC1); hipFree(dC2);
    }
    printf("  => %s\n\n", allok ? "PASS" : "FAIL");
    if (!allok) return 1;
  }

  // ---------------- benchmark ----------------
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}, {5120, 5120, "out"}};
  const int Ms[] = {5, 8, 16, 40, 64};
  printf("== decode, us (best of 3 x 20), NCOPY=%d ==\n", NCOPY);
  printf("%-9s %-4s %8s %8s %8s %8s %8s   %s\n", "shape", "M", "mxfp4", "shipped",
         "bl/K4", "bl/K2", "bl+dir/K1", "best vs shipped");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 64;
    const int nblk = (N + BND - 1) / BND;
    std::vector<unsigned int *> Wa(NCOPY); std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wa[c], (size_t)N * kw * 4)); HIP_CHECK(hipMemset(Wa[c], 0x91 + c, (size_t)N * kw * 4));
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2));  HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    unsigned char *dA, *dWs, *dWref; __half *dS; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));      HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));     HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)4 * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));            HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    for (int M : Ms) {
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      const dim3 g4(nblk, 1, 4), g2(nblk, 1, 2), g1(nblk, 1, 1), b(DWN * 32);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float t[5]; for (int i = 0; i < 5; ++i) t[i] = 1e30f; float ms;
      auto F0 = [&](int it) { DISPATCH(TMV, hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, 128, 4, TMV, false>), g4, b, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)); };
      auto F1 = [&](int it) { DISPATCH(TMV, hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, 4, TMV, true>), g4, b, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)); };
      auto F2 = [&](int it) { DISPATCH(TMV, hipLaunchKernelGGL((ar_decode_opt<DWN, 4, TMV, 1>), g4, b, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)); };
      auto F3 = [&](int it) { DISPATCH(TMV, hipLaunchKernelGGL((ar_decode_opt<DWN, 2, TMV, 1>), g2, b, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)); };
      auto F4 = [&](int it) { DISPATCH(TMV, hipLaunchKernelGGL((ar_decode_opt<DWN, 1, TMV, 3>), g1, b, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)); };
#define TM_(F, D) do { for (int it = 0; it < 3; ++it) F(it); HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it = 0; it < 20; ++it) F(it); \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); D = std::min(D, ms * 50.f); } while (0)
      for (int rep = 0; rep < 3; ++rep) {
        TM_(F0, t[0]); TM_(F1, t[1]); TM_(F2, t[2]); TM_(F3, t[3]); TM_(F4, t[4]);
      }
      const float best = std::min(std::min(t[2], t[3]), t[4]);
      printf("%-9s %-4d %8.1f %8.1f %8.1f %8.1f %8.1f   %6.3fx\n", sh.name, M, t[0], t[1], t[2], t[3], t[4], best / t[1]);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dP); hipFree(dCnt); hipFree(dC);
  }
  return 0;
}
