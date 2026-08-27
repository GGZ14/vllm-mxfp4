// Gate the fp8 prefill GEMM against the reference.
//
// M values are chosen to exercise the epilogue's two paths: 256 and 512 tile the 256-row block
// exactly (all blocks take the branch-free full path), while 300 and 1024 leave a ragged last row
// block that must take the predicated fallback. A wrong `full` predicate shows up only there.
#include "escha_kernels.h"
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
  FILE *f = fopen("prefill_vectors.bin", "rb");
  if (!f) { printf("prefill_vectors.bin missing\n"); return 1; }
  unsigned int magic, ncase;
  if (fread(&magic, 4, 1, f) != 1 || magic != 0xE5C7A002u) { printf("bad magic\n"); return 1; }
  if (fread(&ncase, 4, 1, f) != 1) return 1;
  int bad = 0;
  printf("%-4s %-6s %-6s %-6s %-4s %10s  %s\n", "K", "M", "N", "Kdim", "TN", "rel", "verdict");
  for (unsigned int c = 0; c < ncase; ++c) {
    int K, M, N, Kdim;
    if (fread(&K, 4, 1, f) != 1) break;
    if (fread(&M, 4, 1, f) != 1 || fread(&N, 4, 1, f) != 1 || fread(&Kdim, 4, 1, f) != 1) break;
    const int words = (Kdim / 16) * (N / 16) * (256 * K / 32);
    std::vector<unsigned int> hw(words);
    std::vector<unsigned char> hA((size_t)M * Kdim);
    std::vector<float> hAs(M), hC((size_t)M * N);
    if (fread(hw.data(), 4, words, f) != (size_t)words) break;
    if (fread(hA.data(), 1, hA.size(), f) != hA.size()) break;
    if (fread(hAs.data(), 4, M, f) != (size_t)M) break;
    if (fread(hC.data(), 4, hC.size(), f) != hC.size()) break;

    unsigned int *dW; unsigned char *dA; float *dAs; __bf16 *dC;
    HIP_CHECK(hipMalloc(&dW, (size_t)words * 4));
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dAs, M * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * 2));
    HIP_CHECK(hipMemcpy(dW, hw.data(), (size_t)words * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));

    for (int TN : {2, 4}) {
      const int BNF = EP_WN * TN * 16;
      if (N % BNF) continue;
      HIP_CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
      const dim3 g((N + BNF - 1) / BNF, (M + EP_BMF - 1) / EP_BMF), b(EP_NTHREADS);
#define L(TN_, K_) hipLaunchKernelGGL((escha_gemm_prefill<TN_, K_>), g, b, 0, 0, dA, dW, dAs, dC, M, N, Kdim)
      if (K == 2) { if (TN == 2) L(2, 2); else L(4, 2); }
      else        { if (TN == 2) L(2, 3); else L(4, 3); }
#undef L
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned short> got((size_t)M * N);
      HIP_CHECK(hipMemcpy(got.data(), dC, got.size() * 2, hipMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        unsigned int u = (unsigned int)got[i] << 16; float v;
        __builtin_memcpy(&v, &u, 4);
        num += (v - hC[i]) * (v - hC[i]); den += hC[i] * hC[i];
      }
      const double rel = sqrt(num / (den > 0 ? den : 1));
      const bool ok = rel < 5e-3;
      bad += !ok;
      printf("%-4d %-6d %-6d %-6d %-4d %10.2e  %s\n", K, M, N, Kdim, TN, rel, ok ? "OK" : "FAIL");
    }
    hipFree(dW); hipFree(dA); hipFree(dAs); hipFree(dC);
  }
  fclose(f);
  printf("\nprefill gate: %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
