// Fit a general split-K rule instead of hardcoding the two shapes that were benchmarked.
// MXFP4's split_k_for() keys on WORKGROUP COUNT (nblk*ks) against an M-dependent fill target;
// this measures whether the same shape of rule holds for the trellis kernel, across N values that
// include the TP=2 per-GPU shapes, not just the two full-model ones.
#include "escha_kernels.h"
#include <vector>
#include <algorithm>
#include <cstdio>

int main() {
  struct S { int N, K, bits; };
  const S sh[] = {{5120,4352,3},{5120,8704,3},{8704,5120,2},{8704,4352,2},{17408,5120,2},{2560,5120,2}};
  printf("%-6s %-6s %-4s %-4s %6s %6s %6s %6s   %s\n", "N", "K", "b", "M",
         "KS=1", "KS=2", "KS=4", "KS=8", "best (nblk*ks)");
  for (const S &s : sh) {
    const int N = s.N, K = s.K, KB = s.bits, nblk = N / 128;
    const size_t codeN = (size_t)(K / 16) * (N / 16) * (256 * KB / 32);
    unsigned int *W; unsigned char *dA8; float *dAs, *dP; int *dCnt; __bf16 *dC;
    HIP_CHECK(hipMalloc(&W, codeN * 4)); HIP_CHECK(hipMemset(W, 0x5A, codeN * 4));
    HIP_CHECK(hipMalloc(&dA8, (size_t)64 * K)); HIP_CHECK(hipMemset(dA8, 0x38, (size_t)64 * K));
    HIP_CHECK(hipMalloc(&dAs, 64 * 4)); HIP_CHECK(hipMemset(dAs, 0, 64 * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)8 * 64 * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, (nblk + 16) * 4)); HIP_CHECK(hipMemset(dCnt, 0, (nblk + 16) * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)64 * N * 2));
    for (int M : {8, 40, 64}) {
      const int tm = (M + 15) / 16;
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float b[4] = {1e30f,1e30f,1e30f,1e30f}, ms;
#define F8(KS_,TM_,K_) hipLaunchKernelGGL((escha_gemm_decode_fp8<8,KS_,TM_,K_,4,8,false>), \
        dim3(nblk,1,KS_), dim3(256), 0, 0, dA8, W, dAs, dP, dCnt, dC, M, N, K)
#define TMS(KS_,K_) do { if(tm==1) F8(KS_,1,K_); else if(tm==2) F8(KS_,2,K_); \
        else if(tm==3) F8(KS_,3,K_); else F8(KS_,4,K_); } while(0)
#define ARM(v) do { if(KB==2){ if(v==0)TMS(1,2); else if(v==1)TMS(2,2); else if(v==2)TMS(4,2); else TMS(8,2);} \
        else { if(v==0)TMS(1,3); else if(v==1)TMS(2,3); else if(v==2)TMS(4,3); else TMS(8,3);} } while(0)
#define T(BEST, RUN) { for (int it=0; it<3; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it=0; it<20; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); BEST = std::min(BEST, ms*50.f); }
      for (int r = 0; r < 3; ++r) for (int v = 0; v < 4; ++v) T(b[v], ARM(v));
      int bi = 0; for (int i = 1; i < 4; ++i) if (b[i] < b[bi]) bi = i;
      printf("%-6d %-6d %-4d %-4d %6.1f %6.1f %6.1f %6.1f   KS=%d (%d WGs)\n", N, K, KB, M,
             b[0], b[1], b[2], b[3], 1 << bi, nblk << bi);
#undef F8
#undef TMS
#undef ARM
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    hipFree(W); hipFree(dA8); hipFree(dAs); hipFree(dP); hipFree(dCnt); hipFree(dC);
  }
  return 0;
}
