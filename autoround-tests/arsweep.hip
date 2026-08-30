// Split-K sweep for the int4 decode kernel across the 16-concurrent band (M to 128), on the
// SHARED header the serving module compiles -- decopt.hip holds a drifting copy and is not
// trusted for policy. Bench-only; correctness for the same M range is gated by ar_harness.
#include "../radiance_autoround_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1);} } while (0)

struct Shape { int N, K; const char *name; };

int main() {
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {8192, 5120, "n8192"}, {7168, 5120, "n7168"},
                          {5120, 8704, "down"}, {5120, 5120, "out"}};
  constexpr int DWN = 8, BND = DWN * 16, MMAX = 128;
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP;
    const int nblk = (N + BND - 1) / BND;
    const size_t wbytes = (size_t)N * kw * 4;
    const int NCOPY = (int)std::min<size_t>(12, std::max<size_t>(1, (128ull << 20) / wbytes + 1));
    std::vector<unsigned int *> W(NCOPY); std::vector<__half *> S(NCOPY);
    for (int c = 0; c < NCOPY; ++c) {
      HIP_CHECK(hipMalloc(&W[c], wbytes));            HIP_CHECK(hipMemset(W[c], 0x91 + c, wbytes));
      HIP_CHECK(hipMalloc(&S[c], (size_t)G * N * 2)); HIP_CHECK(hipMemset(S[c], 0x11, (size_t)G * N * 2));
    }
    unsigned char *A; float *As, *P; int *cnt; __bf16 *C;
    HIP_CHECK(hipMalloc(&A, (size_t)MMAX * K));  HIP_CHECK(hipMemset(A, 0x38, (size_t)MMAX * K));
    HIP_CHECK(hipMalloc(&As, MMAX * 4));         HIP_CHECK(hipMemset(As, 0, MMAX * 4));
    HIP_CHECK(hipMalloc(&P, (size_t)4 * MMAX * N * 4));
    HIP_CHECK(hipMalloc(&cnt, nblk * 4));        HIP_CHECK(hipMemset(cnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&C, (size_t)MMAX * N * 2));
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    printf("sweep %-8s N=%d K=%d nblk=%d ncopy=%d\n"
           "%-6s %10s %10s %10s %10s %10s %10s\n",
           sh.name, N, K, nblk, NCOPY, "M",
           "w8k1", "w8k2", "w8k4", "w4k1", "w4k2", "w4k4");
    for (int M : {5, 8, 16, 40, 64, 72, 80, 96, 112, 128}) {
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      printf("%-6d", M);
      for (int dwnsel : {8, 4})
      for (int ks : {1, 2, 4}) {
        const dim3 grid(nblk, 1, ks), blk(DWN * 32);
        (void)grid; (void)blk;
#define L(KS_, TM_) hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWNSEL, KS_, TM_>), \
                        dim3((N + DWNSEL * 16 - 1) / (DWNSEL * 16), 1, KS_), dim3(DWNSEL * 32), \
                        0, 0, A, W[it % NCOPY], S[it % NCOPY], As, P, cnt, C, M, N, K)
#define LT(KS_) do { if (tm==1) L(KS_,1); else if (tm==2) L(KS_,2); else if (tm==3) L(KS_,3); \
                     else if (tm==4) L(KS_,4); else if (tm==5) L(KS_,5); else if (tm==6) L(KS_,6); \
                     else if (tm==7) L(KS_,7); else L(KS_,8); } while (0)
        float best = 1e30f, ms;
#define LD(KS_) do { if (dwnsel == 8) { constexpr int DWNSEL = 8; LT(KS_); } \
                     else             { constexpr int DWNSEL = 4; LT(KS_); } } while (0)
        for (int rep = 0; rep < 3; ++rep) {
          for (int it = 0; it < 3; ++it) { if (ks==1) LD(1); else if (ks==2) LD(2); else LD(4); }
          HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
          for (int it = 0; it < 15; ++it) { if (ks==1) LD(1); else if (ks==2) LD(2); else LD(4); }
          HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
          best = std::min(best, ms / 15.f);
        }
#undef LD
        printf(" %10.1f", best * 1000.f);
#undef LT
#undef L
      }
      printf("\n");
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    for (int c = 0; c < NCOPY; ++c) { hipFree(W[c]); hipFree(S[c]); }
    hipFree(A); hipFree(As); hipFree(P); hipFree(cnt); hipFree(C);
  }
  return 0;
}
