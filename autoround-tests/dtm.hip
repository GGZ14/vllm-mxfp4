// Two follow-ups from the ablation.
//
// 1. M=40 is 1.54x MXFP4 and the ablation says unpack+scale explain almost none of it (fully
//    ablated it is still 1.32x). M=40 takes DTM=3, the only non-power-of-two tile. Is the launcher's
//    "smallest tile that covers M" rule wrong here -- does DTM=4 beat DTM=3 at M=33..48?
// 2. The scale is loaded from global inside the slab loop. Each lane needs exactly ONE fp16 per
//    slab and its N column is fixed for the whole kernel, so the entire per-lane scale vector
//    (spb entries, 10-17 for production shapes) can be preloaded into registers before the loop.
#include "../radiance_autoround_kernels.h"
#include "mxfp4_kernels.h"
#include <vector>
#include <algorithm>

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const int NCOPY = argc > 1 ? atoi(argv[1]) : 3;
  const Shape shapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
  constexpr int DWN = 8, DKS = 4, BND = DWN * 16;
  printf("DTM choice at M in the 17..64 band (us). launcher currently picks ceil(M/16).\n");
  printf("%-9s %-4s %8s %8s %8s %8s %10s\n", "shape", "M", "DTM=1", "DTM=2", "DTM=3", "DTM=4",
         "mxfp4(auto)");
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
    const dim3 grid(nblk, 1, DKS), block(DWN * 32);
    for (int M : {24, 40, 48, 56}) {
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float b[5] = {1e30f, 1e30f, 1e30f, 1e30f, 1e30f}, ms;
#define T(SLOT, L) for (int it = 0; it < 3; ++it) { L; } \
      HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0)); \
      for (int it = 0; it < 20; ++it) { L; } \
      HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); b[SLOT] = std::min(b[SLOT], ms * 50.f);
#define A(TM_) hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, TM_>), grid, block, 0, 0, \
        dA, Wa[it % NCOPY], dS, dAs, dP, dCnt, dC, M, N, K)
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
#define MX hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, 128, DKS, 3, false>), grid, \
        block, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
#define MX4 hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, 128, DKS, 4, false>), grid, \
        block, 0, 0, dA, Wm[it % NCOPY], dWs, dWref, dAs, dP, dCnt, dC, M, N, K)
      for (int rep = 0; rep < 3; ++rep) {
        if (tm <= 1) { T(0, A(1)); }
        if (tm <= 2) { T(1, A(2)); }
        if (tm <= 3) { T(2, A(3)); }
        T(3, A(4));
        if (tm == 3) { T(4, MX); } else { T(4, MX4); }
      }
      printf("%-9s %-4d", sh.name, M);
      for (int i = 0; i < 4; ++i)
        if (b[i] < 1e29f) printf(" %8.1f", b[i]); else printf(" %8s", "-");
      printf(" %10.1f   (launcher picks DTM=%d)\n", b[4], tm);
      hipEventDestroy(e0); hipEventDestroy(e1);
#undef A
#undef MX
#undef MX4
#undef T
    }
    for (int c = 0; c < NCOPY; ++c) { hipFree(Wa[c]); hipFree(Wm[c]); }
    hipFree(dA); hipFree(dS); hipFree(dWs); hipFree(dWref); hipFree(dAs);
    hipFree(dP); hipFree(dCnt); hipFree(dC);
  }
  return 0;
}
