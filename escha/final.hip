// Final confirmation: winning escha config vs the best MXFP4 config, nothing else in the binary.
// Only the surviving arms run, so no spilling or off-optimum variant can depress clocks for the
// arm measured after it -- the trap that made two earlier prefill passes disagree by 5%.
// Five interleaved repeats; min of each.
#include "escha_kernels.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
#include <cstdio>

int main() {
  const int NCOPY = 4;
  struct Shape { int N, K, bits; const char *name; };
  const Shape shapes[] = {{17408, 5120, 2, "gate_up"}, {5120, 8704, 3, "down"}};
  printf("=== DECODE (us, min of 5 interleaved repeats) ===\n");
  printf("%-9s %-4s %10s %10s %9s\n", "shape", "M", "escha", "mxfp4", "esc/mx");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, KB = sh.bits, Mmax = 64;
    const size_t codeN = (size_t)(K / 16) * (N / 16) * (256 * KB / 32);
    std::vector<unsigned int *> Wc(NCOPY); std::vector<unsigned char *> Wm(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&Wc[c], codeN * 4)); HIP_CHECK(hipMemset(Wc[c], 0x5A + c, codeN * 4));
      HIP_CHECK(hipMalloc(&Wm[c], (size_t)N * K / 2)); HIP_CHECK(hipMemset(Wm[c], 0x42 + c, (size_t)N * K / 2));
    }
    unsigned char *dA8, *dWs, *dWref; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dA8, (size_t)Mmax * K)); HIP_CHECK(hipMemset(dA8, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N)); HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4)); HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)8 * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, (N / 32 + 16) * 4)); HIP_CHECK(hipMemset(dCnt, 0, (N / 32 + 16) * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    const int KS = escha_decode_split_k(N / 128, K / 16);
    for (int M : {8, 40, 64}) {
      const int tm = (M + 15) / 16;
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float be = 1e30f, bm = 1e30f, ms;
#define F8(KS_,TM_,K_) hipLaunchKernelGGL((escha_gemm_decode_fp8<8,KS_,TM_,K_,4,8,false>), \
        dim3(N/128,1,KS_), dim3(256), 0, 0, dA8, Wc[it%NCOPY], dAs, dP, dCnt, dC, M, N, K)
#define ETM(KS_,K_) do { if(tm==1) F8(KS_,1,K_); else if(tm==2) F8(KS_,2,K_); \
        else if(tm==3) F8(KS_,3,K_); else F8(KS_,4,K_); } while(0)
#define EA do { if(KS==4){ if(KB==2) ETM(4,2); else ETM(4,3);} else { if(KB==2) ETM(8,2); else ETM(8,3);} } while(0)
#define MX(BK_,KS_,TM_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<8,BK_,KS_,TM_,false>), \
        dim3(N/128,1,KS_), dim3(256), 0, 0, dA8, Wm[it%NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define MTM(BK_,KS_) do { if(tm==1) MX(BK_,KS_,1); else if(tm==2) MX(BK_,KS_,2); \
        else if(tm==3) MX(BK_,KS_,3); else MX(BK_,KS_,4); } while(0)
#define MA do { if(N>=17408){ if(tm==4) MTM(64,1); else MTM(128,1);} else MTM(128,4); } while(0)
#define T(BEST, RUN) { for (int it=0; it<3; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<20; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*50.f); }
      for (int r = 0; r < 5; ++r) { T(be, EA); T(bm, MA); }
      printf("%-9s %-4d %10.1f %10.1f %8.3fx\n", sh.name, M, be, bm, be / bm);
#undef F8
#undef ETM
#undef EA
#undef MX
#undef MTM
#undef MA
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wc[c]); hipFree(Wm[c]); }
    hipFree(dA8); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dP); hipFree(dCnt); hipFree(dC);
  }

  printf("\n=== PREFILL (us, min of 5 interleaved repeats) ===\n");
  printf("%-9s %-6s %10s %10s %9s %10s\n", "shape", "M", "escha", "mxfp4", "esc/mx", "TFLOP/s");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, KB = sh.bits, Mmax = 2048;
    const size_t codeN = (size_t)(K / 16) * (N / 16) * (256 * KB / 32);
    unsigned int *Wc; unsigned char *Wm, *dA, *dWs, *dWref; float *dAs; __bf16 *dC;
    HIP_CHECK(hipMalloc(&Wc, codeN * 4)); HIP_CHECK(hipMemset(Wc, 0x5A, codeN * 4));
    HIP_CHECK(hipMalloc(&Wm, (size_t)N * K / 2)); HIP_CHECK(hipMemset(Wm, 0x42, (size_t)N * K / 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K)); HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dWs, (size_t)(K / 32) * N)); HIP_CHECK(hipMemset(dWs, 127, (size_t)(K / 32) * N));
    HIP_CHECK(hipMalloc(&dWref, N)); HIP_CHECK(hipMemset(dWref, 127, N));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4)); HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    for (int M : {512, 2048}) {
      const int EB = EP_WN * 2 * 16;
      // TM picks the row block: 8 (512 rows) unless that would starve the grid, which happens at
      // down/M=512 -- one row block x 80 column blocks is 80 workgroups for 64 CUs.
      const int TMs = ((N / EB) * ((M + 511) / 512) >= 128) ? 8 : 4;
      const dim3 g8((N + EB - 1) / EB, (M + 511) / 512), g4((N + EB - 1) / EB, (M + 255) / 256);
      const dim3 eb(EP_NTHREADS);
      const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mb(NTHREADS);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float be = 1e30f, bm = 1e30f, ms;
#define EP(K_) do { if (TMs == 8) hipLaunchKernelGGL((escha_gemm_prefill<2,K_,8>), g8, eb, 0, 0, \
          dA, Wc, dAs, dC, M, N, K); else hipLaunchKernelGGL((escha_gemm_prefill<2,K_,4>), g4, eb, \
          0, 0, dA, Wc, dAs, dC, M, N, K); } while(0)
#define MP hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2,false>), mg, mb, 0, 0, dA, Wm, dWs, dWref, dAs, dC, M, N, K)
#define P(BEST, RUN) { for (int it=0; it<2; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<10; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*100.f); }
      for (int r = 0; r < 5; ++r) { if (KB == 2) { P(be, EP(2)); } else { P(be, EP(3)); } P(bm, MP); }
      printf("%-9s %-6d %10.1f %10.1f %8.3fx %9.1f\n", sh.name, M, be, bm, be / bm,
             2.0 * M * N * K / (be * 1e-6) / 1e12);
#undef EP
#undef MP
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    hipFree(Wc); hipFree(Wm); hipFree(dA); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
