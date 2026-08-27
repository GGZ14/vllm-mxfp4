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

  // Launch geometry sweep. The earlier runs pinned DWN=8 / DKS=1, which for N=5120 is
  // 5120/128 = 40 workgroups on a 64-CU part -- 24 CUs idle for the whole GEMM. Neither the block
  // width nor the k-split can be assumed; both are swept per shape and per M.
  printf("=== DECODE (us, interleaved, %d rotated buffers) ===\n", NCOPY);
  printf("%-9s %-4s %5s %5s %5s %8s %8s %8s   %s\n", "shape", "M", "DWN", "DKS", "KB",
         "escha", "mxfp4", "esc/mx", "WGs");
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
    unsigned char *dA8, *dWs, *dWref; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA8, (size_t)Mmax * K));     HIP_CHECK(hipMemset(dA8, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N));                  HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));             HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)8 * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, (N / 32 + 16) * 4));   HIP_CHECK(hipMemset(dCnt, 0, (N / 32 + 16) * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    for (int M : {8, 40, 64}) {
      const int tm = (M + 15) / 16;
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      // arms: (DWN, DKS) x KB, plus one mxfp4 reference arm
      struct Cfg { int dwn, dks, kb; };
      const Cfg cfg[] = {{8,4,4},{8,4,8},{8,8,4},{8,8,8},{8,4,80},{8,8,80},{16,4,8},{16,8,8}};
      const int NC = sizeof(cfg)/sizeof(cfg[0]);
      float be[16]; for (int i=0;i<NC;++i) be[i]=1e30f;
      float bmv[4] = {1e30f, 1e30f, 1e30f, 1e30f}, ms;
// kb of 80 means "KB=8 with the 24-bit multiply decomposition" -- the quarter-rate
      // v_mul_lo_u32 is ~42% of the codec's cycles, so it gets one more honest look now that the
      // kernel around it has changed (fp8 pipe, split-K).
#define F8(DWN_,KS_,TM_,K_,KB_,M24_) \
        hipLaunchKernelGGL((escha_gemm_decode_fp8<DWN_,KS_,TM_,K_,KB_,8,M24_>), \
        dim3(N/(DWN_*16),1,KS_), dim3(DWN_*32), 0, 0, dA8, Wc[it%NCOPY], dAs, dP, dCnt, dC, M, N, K)
#define BYTM(DWN_,KS_,K_,KB_,M24_) do { if(tm==1) F8(DWN_,KS_,1,K_,KB_,M24_); \
        else if(tm==2) F8(DWN_,KS_,2,K_,KB_,M24_); \
        else if(tm==3) F8(DWN_,KS_,3,K_,KB_,M24_); else F8(DWN_,KS_,4,K_,KB_,M24_); } while(0)
#define BYK(DWN_,KS_,KB_,M24_) do { if(KB==2) BYTM(DWN_,KS_,2,KB_,M24_); \
        else BYTM(DWN_,KS_,3,KB_,M24_); } while(0)
#define ARM(i) do { const Cfg &q = cfg[i]; \
        if(q.dwn==8&&q.dks==4&&q.kb==4) BYK(8,4,4,false); \
        else if(q.dwn==8&&q.dks==4&&q.kb==8) BYK(8,4,8,false); \
        else if(q.dwn==8&&q.dks==8&&q.kb==4) BYK(8,8,4,false); \
        else if(q.dwn==8&&q.dks==8&&q.kb==8) BYK(8,8,8,false); \
        else if(q.dwn==8&&q.dks==4) BYK(8,4,8,true); \
        else if(q.dwn==8&&q.dks==8) BYK(8,8,8,true); \
        else if(q.dks==4) BYK(16,4,8,false); else BYK(16,8,8,false); } while(0)
      // MXFP4 gets the SAME sweep, otherwise the comparison is rigged: its production launcher
      // picks split-K from split_k_for() and BK from decode_bk64(), and pinning it to DKS=1 hands
      // it the identical workgroup starvation this sweep exists to fix. Best-of is compared to
      // best-of.
#define MX(BK_,KS_,TM_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<8,BK_,KS_,TM_,false>), \
        dim3(N/128,1,KS_), dim3(256), 0, 0, dA8, Wm[it%NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define MXTM(BK_,KS_) do { if(tm==1) MX(BK_,KS_,1); else if(tm==2) MX(BK_,KS_,2); \
        else if(tm==3) MX(BK_,KS_,3); else MX(BK_,KS_,4); } while(0)
#define MXARM(i) do { if(i==0) MXTM(128,1); else if(i==1) MXTM(64,1); \
        else if(i==2) MXTM(128,2); else MXTM(128,4); } while(0)
#define TIME(BEST, RUN) { for (int it=0; it<3; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<20; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*50.f); }
      for (int rep = 0; rep < 3; ++rep) {
        for (int i = 0; i < NC; ++i) TIME(be[i], ARM(i));
        for (int i = 0; i < 4; ++i) TIME(bmv[i], MXARM(i));
      }
      int bi = 0; for (int i = 1; i < NC; ++i) if (be[i] < be[bi]) bi = i;
      float bm = bmv[0]; int mi = 0;
      for (int i = 1; i < 4; ++i) if (bmv[i] < bm) { bm = bmv[i]; mi = i; }
      static const char *mxn[4] = {"BK128/KS1", "BK64/KS1", "BK128/KS2", "BK128/KS4"};
      for (int i = 0; i < NC; ++i)
        printf("%-9s %-4d %5d %5d %5s %8.1f %8.1f %7.3fx   %d%s\n", i ? "" : sh.name, i ? 0 : M,
               cfg[i].dwn, cfg[i].dks, cfg[i].kb == 80 ? "8m24" : (cfg[i].kb == 4 ? "4" : "8"),
               be[i], bm, be[i] / bm,
               (N / (cfg[i].dwn * 16)) * cfg[i].dks, i == bi ? "  <== best" : "");
      printf("%-9s %-4s %5s %5s %5s %8s %8.1f %7s   mxfp4 = %s\n", "", "", "", "", "", "", bm, "",
             mxn[mi]);
#undef F8
#undef BYTM
#undef BYK
#undef ARM
#undef MX
#undef MXTM
#undef MXARM
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wc[c]); hipFree(Wm[c]); }
    hipFree(dA8); hipFree(dWs); hipFree(dWref); hipFree(dAs);
    hipFree(dP); hipFree(dCnt); hipFree(dC);
  }

  printf("\n=== PREFILL (us) ===\n");
  printf("%-9s %-6s %8s %8s %8s %8s %8s  %s\n", "shape", "M", "TN2/TM4", "TN2/TM8", "TN4/TM4", "mxfp4", "best/mx", "TFLOP/s");
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
      // BNF = EP_WN*TN*16, so TN=4 doubles the N tile to 128 and halves the workgroup count --
      // fewer, fatter blocks that decode each weight once for twice the columns.
      const int EB = EP_WN * 2 * 16, EB4 = EP_WN * 4 * 16;
      const dim3 eg4((N + EB - 1) / EB, (M + EP_BMF - 1) / EP_BMF), eb(EP_NTHREADS);
      const dim3 eg8((N + EB - 1) / EB, (M + 2 * EP_BMF - 1) / (2 * EP_BMF));
      const dim3 eg4b((N + EB4 - 1) / EB4, (M + EP_BMF - 1) / EP_BMF);
      const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mb(NTHREADS);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float be = 1e30f, be8 = 1e30f, bn4 = 1e30f, bm = 1e30f, ms;
#define EP(K_) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,4>), eg4, eb, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define EP8(K_) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,8>), eg8, eb, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define EPN4(K_) hipLaunchKernelGGL((escha_gemm_prefill<4,K_,4>), eg4b, eb, 0, 0, dA, Wc, dAs, dC, M, N, K)
#define MP hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2,false>), mg, mb, 0, 0, dA, Wm, dWs, dWref, dAs, dC, M, N, K)
#define PT(BEST, RUN) { for (int it=0; it<2; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<10; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*100.f); }
      for (int rep = 0; rep < 3; ++rep) {
        if (KB == 2) { PT(be, EP(2)); PT(be8, EP8(2)); PT(bn4, EPN4(2)); }
        else { PT(be, EP(3)); PT(be8, EP8(3)); PT(bn4, EPN4(3)); }
        PT(bm, MP);
      }
#undef EP
#undef EP8
#undef EPN4
#undef MP
      const float bb = std::min(std::min(be, be8), bn4);
      const double tf = 2.0 * M * N * K / (bb * 1e-6) / 1e12;
      printf("%-9s %-6d %8.1f %8.1f %8.1f %8.1f %7.3fx  %.1f\n", sh.name, M, be, be8, bn4, bm,
             bb / bm, tf);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    hipFree(Wc); hipFree(Wm); hipFree(dA); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
