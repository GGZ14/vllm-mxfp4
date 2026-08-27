// How much could any unpack or rescale optimisation possibly buy?
//
// Two ablations, each of which makes the kernel WRONG but leaves memory traffic and WMMA count
// untouched, so the delta is an upper bound on the corresponding optimisation:
//   -unpack : write raw code bytes instead of running the 16-entry int4->e4m3 LUT
//   -scale  : skip the per-group scale load and the fold FMAs
// Anything the ablations do not explain is memory or launch, and is not worth chasing in the ALU.
#include "../radiance_autoround_kernels.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>
#include <string>

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 3;
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
  constexpr int DWN = 8, DKS = 4, BND = DWN * 16, ARTN = 2, AR_BNF = AR_WN * ARTN * 16;

  printf("decode (us, min of 3x20, DRAM-resident, %d buffers)\n", NCOPY);
  printf("%-9s %-4s %9s %9s %9s %9s %9s\n", "shape", "M", "full", "-unpack", "-scale", "-LDS",
         "mxfp4");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 64;
    const int nblk = (N + BND - 1) / BND;
    std::vector<unsigned int *> Wa(NCOPY);
    std::vector<unsigned char *> Wm(NCOPY);
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
    HIP_CHECK(hipMalloc(&dP, (size_t)DKS * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));            HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));

    for (int M : {8, 40, 64}) {
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      const dim3 grid(nblk, 1, DKS), block(DWN * 32);
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float best[5] = {1e30f, 1e30f, 1e30f, 1e30f, 1e30f}, ms;
#define TIME(SLOT, LAUNCH)                                                            \
      for (int it = 0; it < 3; ++it) { LAUNCH; }                                       \
      HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));                \
      for (int it = 0; it < 20; ++it) { LAUNCH; }                                      \
      HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));               \
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); best[SLOT] = std::min(best[SLOT], ms * 50.f);
#define ARL(AB, TM_) hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, TM_, true, AB>), grid, \
        block, 0, 0, dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
#define BY_TM(AB) do { if (tm==1) ARL(AB,1); else if (tm==2) ARL(AB,2); else if (tm==3) ARL(AB,3); else ARL(AB,4); } while(0)
#define MXL(TM_) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, 128, DKS, TM_, false>), \
        grid, block, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define MX_BY_TM do { if (tm==1) MXL(1); else if (tm==2) MXL(2); else if (tm==3) MXL(3); else MXL(4); } while(0)
      for (int rep = 0; rep < 3; ++rep) {
        TIME(0, BY_TM(0)); TIME(1, BY_TM(1)); TIME(2, BY_TM(2)); TIME(3, BY_TM(4));
        TIME(4, MX_BY_TM);
      }
      printf("%-9s %-4d %9.1f %9.1f %9.1f %9.1f %9.1f\n", sh.name, M, best[0], best[1], best[2],
             best[3], best[4]);
      hipEventDestroy(e0); hipEventDestroy(e1);
#undef ARL
#undef BY_TM
#undef MXL
#undef MX_BY_TM
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs);
    hipFree(dP); hipFree(dCnt); hipFree(dC);
  }

  printf("\nprefill (us)\n");
  printf("%-9s %-6s %9s %9s %9s %9s %9s\n", "shape", "M", "full", "-unpack", "-scale", "-both",
         "mxfp4");
  for (const Shape &sh : shapes) {
    const int N = sh.N, K = sh.K, kw = K / 8, G = K / AR_GROUP, Mmax = 2048;
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
    const int M = 2048;
    const dim3 ag((N + AR_BNF - 1) / AR_BNF, (M + AR_BMF - 1) / AR_BMF), abk(AR_NTHREADS);
    const dim3 mg((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF), mbk(NTHREADS);
    hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    float best[5] = {1e30f, 1e30f, 1e30f, 1e30f, 1e30f}, ms;
#define PTIME(SLOT, LAUNCH)                                                           \
    for (int it = 0; it < 2; ++it) { LAUNCH; }                                         \
    HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));                  \
    for (int it = 0; it < 10; ++it) { LAUNCH; }                                        \
    HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));                 \
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); best[SLOT] = std::min(best[SLOT], ms * 100.f);
#define PAR(AB) hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<ARTN, true, AB>), ag, abk, 0, 0, dA, \
        Wa[it % NCOPY], dS, dAs, dC, M, N, K)
#define PMX hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false>), mg, mbk, 0, 0, dA, \
        Wm[it % NCOPY], dWs, dWref, dAs, dC, M, N, K)
    for (int rep = 0; rep < 3; ++rep) {
      PTIME(0, PAR(0)); PTIME(1, PAR(1)); PTIME(2, PAR(2)); PTIME(3, PAR(3));
      PTIME(4, PMX);
    }
    printf("%-9s %-6d %9.1f %9.1f %9.1f %9.1f %9.1f\n", sh.name, M, best[0], best[1], best[2],
           best[3], best[4]);
    hipEventDestroy(e0); hipEventDestroy(e1);
#undef PAR
#undef PMX
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs); hipFree(dC);
  }
  return 0;
}
