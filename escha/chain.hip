// Where does the escha LAYER's prefill time actually go?
//
// The GEMM microbenchmark says escha prefill is at parity with MXFP4 (1.046x), but served prefill
// is 0.59x. That gap has to live in the rest of the layer, and "integration overhead" is not a
// measurement. This times the pieces of one real layer at real prefill shapes:
//
//   pre_amax -> pre_quant -> GEMM -> post_rot
//
// and prices the merged-layer case honestly: gate_up is TWO shards (gate K=2, up K=3, different
// rin), so it pays the rotation chain twice for one logical layer.
#include "escha_kernels.h"
#include "escha_act.h"
#include <vector>
#include <algorithm>
#include <cstdio>

int main() {
  struct S { int N, K, bits; const char *name; int shards; };
  // per-GPU shapes at TP=2, as served
  const S sh[] = {{8704, 5120, 2, "gate_up(1 shard)", 1},
                  {5120, 8704, 3, "down", 1},
                  {5120, 5120, 2, "in_proj_qkv", 1}};
  printf("%-18s %-6s %9s %9s %9s %9s %9s   %s\n", "shape", "M", "pre_amax", "pre_quant",
         "GEMM", "post_rot", "TOTAL", "rot % of total");
  for (const S &s : sh) {
    const int N = s.N, K = s.K, KB = s.bits, Mmax = 8192;
    const size_t codeN = (size_t)(K / 16) * (N / 16) * (256 * KB / 32);
    unsigned int *W; unsigned char *dA; __bf16 *dX, *dC, *dY;
    __half *dRin, *dRout; float *dSin, *dSout, *dAs, *dAmax;
    HIP_CHECK(hipMalloc(&W, codeN * 4)); HIP_CHECK(hipMemset(W, 0x5A, codeN * 4));
    HIP_CHECK(hipMalloc(&dX, (size_t)Mmax * K * 2)); HIP_CHECK(hipMemset(dX, 0x38, (size_t)Mmax * K * 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dRin, K * 2));  HIP_CHECK(hipMemset(dRin, 0x3C, K * 2));
    HIP_CHECK(hipMalloc(&dRout, N * 2)); HIP_CHECK(hipMemset(dRout, 0x3C, N * 2));
    HIP_CHECK(hipMalloc(&dSin, K * 4));  HIP_CHECK(hipMemset(dSin, 0, K * 4));
    HIP_CHECK(hipMalloc(&dSout, N * 4)); HIP_CHECK(hipMemset(dSout, 0, N * 4));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4)); HIP_CHECK(hipMalloc(&dAmax, Mmax * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    HIP_CHECK(hipMalloc(&dY, (size_t)Mmax * N * 2));
    for (int M : {2048, 8192}) {
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      float b[4] = {1e30f, 1e30f, 1e30f, 1e30f}, ms;
      const dim3 preg((K + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M), blk(ESCHA_ACT_THREADS);
      const dim3 postg((N + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M);
      const int EB = EP_WN * 2 * 16, ncol = (N + EB - 1) / EB;
      const bool wide = (size_t)ncol * ((M + 511) / 512) >= 128;
      const dim3 gg(ncol, wide ? (M + 511) / 512 : (M + 255) / 256), gb(EP_NTHREADS);
#define T(BEST, RUN) { for (int it = 0; it < 2; ++it) { RUN; } HIP_CHECK(hipDeviceSynchronize()); \
        HIP_CHECK(hipEventRecord(e0)); for (int it = 0; it < 10; ++it) { RUN; } \
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1)); \
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); BEST = std::min(BEST, ms * 100.f); }
#define GEMM do { if (wide) { if (KB==2) hipLaunchKernelGGL((escha_gemm_prefill<2,2,8>), gg, gb, 0, 0, dA, W, dAs, dC, M, N, K); \
          else hipLaunchKernelGGL((escha_gemm_prefill<2,3,8>), gg, gb, 0, 0, dA, W, dAs, dC, M, N, K); } \
        else { if (KB==2) hipLaunchKernelGGL((escha_gemm_prefill<2,2,4>), gg, gb, 0, 0, dA, W, dAs, dC, M, N, K); \
          else hipLaunchKernelGGL((escha_gemm_prefill<2,3,4>), gg, gb, 0, 0, dA, W, dAs, dC, M, N, K); } } while (0)
      for (int r = 0; r < 3; ++r) {
        T(b[0], hipLaunchKernelGGL(escha_pre_amax, preg, blk, 0, 0, dX, dSin, dRin, dAmax, M, K));
        T(b[1], hipLaunchKernelGGL(escha_pre_quant, preg, blk, 0, 0, dX, dSin, dRin, dAmax, dA, dAs, M, K));
        T(b[2], GEMM);
        T(b[3], hipLaunchKernelGGL(escha_post_rot, postg, blk, 0, 0, dC, dRout, dSout, dY, M, N, N, 0));
      }
#undef T
#undef GEMM
      const float tot = b[0] + b[1] + b[2] + b[3], rot = b[0] + b[1] + b[3];
      printf("%-18s %-6d %9.1f %9.1f %9.1f %9.1f %9.1f   %.0f%%\n", s.name, M,
             b[0], b[1], b[2], b[3], tot, 100.0 * rot / tot);
      hipEventDestroy(e0); hipEventDestroy(e1);
    }
    hipFree(W); hipFree(dX); hipFree(dA); hipFree(dRin); hipFree(dRout);
    hipFree(dSin); hipFree(dSout); hipFree(dAs); hipFree(dAmax); hipFree(dC); hipFree(dY);
  }
  return 0;
}
