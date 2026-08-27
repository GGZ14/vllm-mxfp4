// AutoRound int4-g128 W4A8 vs the shipped MXFP4 W4A8 kernel: same binary, same conditions,
// interleaved in one timing loop.
//
// Bytes streamed per weight element:
//   AutoRound : N*K/2 codes + N*K/64  fp16 group scales          = N*K * 0.5156
//   MXFP4     : N*K/2 codes + N*K/32  E8M0 block scales + N refs = N*K * 0.5313
// AutoRound reads 2.9% fewer bytes. Whether that shows up depends entirely on where the weights
// live, which is why NCOPY exists: with one buffer a 44.6 MB weight sits inside the 64 MB
// Infinity Cache and both kernels run at ~1170 GB/s, where the ALU and latency differences
// dominate and the byte advantage is invisible. Serving streams ~14 GB from DRAM. Pass NCOPY so
// NCOPY * weight_bytes > 64 MB to measure the case that matters.
//
// Weights are random and the two formats hold DIFFERENT values; this measures time, not
// agreement. Each kernel's correctness is gated elsewhere.
#include "../radiance_autoround_kernels.h"
#include "mxfp4_kernels.h"
#include <string>
#include <vector>
#include <algorithm>

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 1;
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}, {5120, 5120, "out"}};
  const int decMs[] = {5, 8, 16, 40, 64};
  constexpr int DWN = 8, DKS = 4, BND = DWN * 16;
  constexpr int ARTN = 2, AR_BNF = AR_WN * ARTN * 16;

  printf("weight buffers rotated: %d\n", NCOPY);
  printf("%-9s %-6s %11s %11s %10s %10s %9s\n", "shape", "M", "wset MB", "ar-grouptmp", "ar-imajor", "mxfp4", "imaj/mx");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 64;
    const int nblk = (N + BND - 1) / BND;
    std::vector<unsigned int *> Wa(NCOPY);
    std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wa[c], (size_t)N * kw * 4));
      HIP_CHECK(hipMemset(Wa[c], 0x91 + c, (size_t)N * kw * 4));
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2));
      HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    unsigned char *dA, *dWs, *dWref;
    __half *dS; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));      HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));     HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)DKS * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));            HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    const double wsetMB = NCOPY * ((double)N * K / 2 + (double)G * N * 2) / (1024.0 * 1024.0);

    for (int M : decMs) {
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      const dim3 grid(nblk, 1, DKS), block(DWN * 32);
#define AR_D(TM_) hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, TM_, false>), grid, block, 0, 0, \
                    dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
#define AR_I(TM_) hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, TM_, true>), grid, block, 0, 0, \
                    dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
#define MX_D(TM_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, 128, DKS, TM_, false>), \
                    grid, block, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define RUN(MAC) do { if (tm==1) MAC(1); else if (tm==2) MAC(2); else if (tm==3) MAC(3); else MAC(4); } while (0)
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float ab = 1e30f, ib = 1e30f, mb = 1e30f, ms;
      for (int rep = 0; rep < 3; ++rep) {
        for (int it = 0; it < 3; ++it) RUN(AR_D);
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
        for (int it = 0; it < 20; ++it) RUN(AR_D);
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ab = std::min(ab, ms * 50.f);

        for (int it = 0; it < 3; ++it) RUN(AR_I);
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
        for (int it = 0; it < 20; ++it) RUN(AR_I);
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ib = std::min(ib, ms * 50.f);

        for (int it = 0; it < 3; ++it) RUN(MX_D);
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
        for (int it = 0; it < 20; ++it) RUN(MX_D);
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); mb = std::min(mb, ms * 50.f);
      }
      printf("%-9s %-6d %11.0f %11.1f %10.1f %10.1f %8.3fx\n", sh.name, M, wsetMB, ab, ib, mb,
             ib / mb);
      hipEventDestroy(e0); hipEventDestroy(e1);
#undef AR_D
#undef AR_I
#undef MX_D
#undef RUN
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref);
    hipFree(dAs); hipFree(dP); hipFree(dCnt); hipFree(dC);
  }

  printf("\n%-9s %-6s %11s %10s %10s %10s %9s\n", "shape", "M(pre)", "wset MB", "ar-grouptmp",
         "ar-imajor", "mxfp4", "imaj/mx");
  {
    const Shape pshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
    const int pMs[] = {512, 2048, 4096};
    for (const Shape &sh : pshapes) {
      const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 4096;
      std::vector<unsigned int *> Wa(NCOPY);
      std::vector<unsigned char *> Wm(NCOPY);
      for (int c = 0; c < NCOPY; ++c) {
        HIP_CHECK(hipMalloc(&Wa[c], (size_t)N * kw * 4));
        HIP_CHECK(hipMemset(Wa[c], 0x91 + c, (size_t)N * kw * 4));
        HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2));
        HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
      }
      unsigned char *dA, *dWs, *dWref; __half *dS; float *dAs; __bf16 *dC;
      HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));      HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
      HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));     HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
      HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
      HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
      HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
      HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
      const double wsetMB = NCOPY * ((double)N * K / 2 + (double)G * N * 2) / (1024.0 * 1024.0);
      for (int M : pMs) {
        const dim3 ag((N + AR_BNF - 1) / AR_BNF, (M + AR_BMF - 1) / AR_BMF), abk(AR_NTHREADS);
        const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mbk(NTHREADS);
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
        float abest = 1e30f, ibest = 1e30f, mbest = 1e30f, ms;
        for (int rep = 0; rep < 3; ++rep) {
          for (int it = 0; it < 2; ++it)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<ARTN, false>), ag, abk, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K);
          HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
          for (int it = 0; it < 10; ++it)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<ARTN, false>), ag, abk, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K);
          HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); abest = std::min(abest, ms * 100.f);

          for (int it = 0; it < 2; ++it)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<ARTN, true>), ag, abk, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K);
          HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
          for (int it = 0; it < 10; ++it)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<ARTN, true>), ag, abk, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dC, M, N, K);
          HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); ibest = std::min(ibest, ms * 100.f);

          for (int it = 0; it < 2; ++it)
            hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false>), mg, mbk, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dC, M, N, K);
          HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
          for (int it = 0; it < 10; ++it)
            hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false>), mg, mbk, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dC, M, N, K);
          HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); mbest = std::min(mbest, ms * 100.f);
        }
        printf("%-9s %-6d %11.0f %10.1f %10.1f %10.1f %8.3fx\n", sh.name, M, wsetMB, abest,
               ibest, mbest, ibest / mbest);
        hipEventDestroy(e0); hipEventDestroy(e1);
      }
      for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
      hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
    }
  }
  return 0;
}
