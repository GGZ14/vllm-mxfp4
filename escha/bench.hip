// escha W2 vs the shipped MXFP4 kernels, one binary, interleaved.
//
// Methodology carried over from the MXFP4/int4 work, where each of these cost a wrong conclusion:
//   * interleave the variants -- the first measured on a shape reads ~17-20% inflated;
//   * rotate over NCOPY weight buffers so the working set clears the 64 MB Infinity Cache, or a
//     44 MB weight sits in L3 and reports >100% of DRAM peak;
//   * report bytes/weight alongside time, because at decode these kernels are bandwidth-bound and
//     the byte count is most of the answer.
//
// escha carries 2.469 bits/weight against MXFP4's 4.25, so at decode it should win roughly in
// proportion. At prefill it should LOSE: both use the fp8 pipe, but escha additionally decodes a
// trellis, and this comparison does not even include the two Hadamard-128 activation passes escha
// needs and MXFP4 does not.
#include "escha_kernels.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
#include <cstdio>

struct Shape { int N, K, bits; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 4;
  const Shape shapes[] = {{17408, 5120, 2, "gate_up"}, {5120, 8704, 3, "down"}};
  constexpr int DWN = 8, BND = DWN * 16;

  printf("=== DECODE (us, min of 3 interleaved passes, %d rotated buffers) ===\n", NCOPY);
  printf("%-9s %-4s %8s %8s %8s %8s %8s   %s\n", "shape", "M",
         "f16", "fp8 KB4", "fp8 KB8", "mxfp4", "fp8/mx", "fp8 vs f16");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, KB = sh.bits, Mmax = 64;
    const int ktiles = K / 16, ntiles = N / 16, words = 256 * KB / 32;
    const size_t codeN = (size_t)ktiles * ntiles * words;
    std::vector<unsigned int *> Wc(NCOPY);
    std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wc[c], codeN * 4));  HIP_CHECK(hipMemset(Wc[c], 0x5A + c, codeN * 4));
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2)); HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    __half *dAh; unsigned char *dA8, *dWs, *dWref; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dAh, (size_t)Mmax * K * 2)); HIP_CHECK(hipMemset(dAh, 0x38, (size_t)Mmax * K * 2));
    HIP_CHECK(hipMalloc(&dA8, (size_t)Mmax * K));     HIP_CHECK(hipMemset(dA8, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)4 * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, (N / BND + 8) * 4));   HIP_CHECK(hipMemset(dCnt, 0, (N / BND + 8) * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    const double bytes = codeN * 4.0;
    for (int M : {8, 40, 64}) {
      const int tm = (M + 15) / 16;
      const dim3 g(N / BND, 1, 1), b(DWN * 32);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float be[3] = {1e30f, 1e30f, 1e30f}, bm = 1e30f, ms;
#define ES(TM_,K_,KB_,P_,A_) hipLaunchKernelGGL((escha_gemm_decode<DWN,1,TM_,K_,KB_,P_,false,A_>), \
        g, b, 0, 0, dAh, Wc[it%NCOPY], dP, dCnt, dC, M, N, K)
#define ESD(K_,KB_,P_,A_) do { if(tm==1) ES(1,K_,KB_,P_,A_); else if(tm==2) ES(2,K_,KB_,P_,A_); \
        else if(tm==3) ES(3,K_,KB_,P_,A_); else ES(4,K_,KB_,P_,A_); } while(0)
#define F8(TM_,K_,KB_) hipLaunchKernelGGL((escha_gemm_decode_fp8<DWN,1,TM_,K_,KB_,8>), \
        g, b, 0, 0, dA8, Wc[it%NCOPY], dAs, dP, dCnt, dC, M, N, K)
#define F8D(K_,KB_) do { if(tm==1) F8(1,K_,KB_); else if(tm==2) F8(2,K_,KB_); \
        else if(tm==3) F8(3,K_,KB_); else F8(4,K_,KB_); } while(0)
#define ESK(K_,i) do { if(i==0) ESD(K_,4,4,0); else if(i==1) F8D(K_,4); \
        else F8D(K_,8); } while(0)
#define MX(TM_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN,128,1,TM_,false>), g, b, 0, 0, \
        dA8, Wm[it%NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define MXD do { if(tm==1) MX(1); else if(tm==2) MX(2); else if(tm==3) MX(3); else MX(4); } while(0)
#define TIME(BEST, RUN) { for (int it=0; it<3; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<20; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*50.f); }
      for (int rep = 0; rep < 3; ++rep) {
        for (int v = 0; v < 3; ++v) {
          if (KB == 2) { TIME(be[v], ESK(2, v)); } else { TIME(be[v], ESK(3, v)); }
        }
        TIME(bm, MXD);
      }
#undef ES
#undef ESD
#undef ESK
#undef F8
#undef F8D
#undef MX
#undef MXD
      const float f8 = std::min(be[1], be[2]);
      printf("%-9s %-4d %8.1f %8.1f %8.1f %8.1f %7.3fx   %+.0f%%\n", sh.name, M,
             be[0], be[1], be[2], bm, f8 / bm, 100.0 * (be[0] - f8) / be[0]);
      (void)bytes;
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wc[c]); hipFree(Wm[c]); }
    hipFree(dAh); hipFree(dA8); hipFree(dWs); hipFree(dWref); hipFree(dAs);
    hipFree(dP); hipFree(dCnt); hipFree(dC);
  }

  printf("\n=== PREFILL (us) ===\n");
  printf("%-9s %-6s %9s %9s %9s %9s %9s  %s\n", "shape", "M", "TM4/WM4", "TM8/WM4", "TM4/WM8", "mxfp4", "best/mx", "TFLOP/s");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, KB = sh.bits, Mmax = 2048;
    const int ktiles = K / 16, ntiles = N / 16, words = 256 * KB / 32;
    const size_t codeN = (size_t)ktiles * ntiles * words;
    unsigned int *Wc; unsigned char *Wm, *dA, *dWs, *dWref; float *dAs; __bf16 *dC;
    HIP_CHECK(hipMalloc(&Wc, codeN * 4));  HIP_CHECK(hipMemset(Wc, 0x5A, codeN * 4));
    HIP_CHECK(hipMalloc(&Wm, (size_t)N * K / 2)); HIP_CHECK(hipMemset(Wm, 0x42, (size_t)N * K / 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));  HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));              HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));         HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    for (int M : {512, 2048}) {
      const int EB = EP_WN * 2 * 16;
      const dim3 eg4((N + EB - 1) / EB, (M + EP_BMF - 1) / EP_BMF), eb(EP_NTHREADS);
      const dim3 eg8((N + EB - 1) / EB, (M + 2 * EP_BMF - 1) / (2 * EP_BMF));
      const dim3 ebw(EP_WM * 2 * EP_WN * 32);
      const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mb(NTHREADS);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float be = 1e30f, be8 = 1e30f, bew = 1e30f, bm = 1e30f, ms;
#define EP(K_) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,4>), eg4, eb, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define EP8(K_) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,8>), eg8, eb, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define EPW(K_) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,4,8>), eg8, ebw, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define MP hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2,false>), mg, mb, 0, 0, dA, Wm, dWs, dWref, dAs, dC, M, N, K)
#define PT(BEST, RUN) { for (int it=0; it<2; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<10; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*100.f); }
      for (int rep = 0; rep < 3; ++rep) {
        if (KB == 2) { PT(be, EP(2)); PT(be8, EP8(2)); PT(bew, EPW(2)); }
        else { PT(be, EP(3)); PT(be8, EP8(3)); PT(bew, EPW(3)); }
        PT(bm, MP);
      }
#undef EP
#undef EP8
#undef EPW
#undef MP
      const float bb = std::min(std::min(be, be8), bew);
      const double tf = 2.0 * M * N * K / (bb * 1e-6) / 1e12;
      printf("%-9s %-6d %9.1f %9.1f %9.1f %9.1f %8.3fx  %.1f\n", sh.name, M, be, be8, bew, bm,
             bb / bm, tf);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    hipFree(Wc); hipFree(Wm); hipFree(dA); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
