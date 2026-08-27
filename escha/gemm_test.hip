// Gate the decode GEMM against the NumPy reference (real escha weights, both bit rates).
//
// Tolerance, not bit-exactness, and deliberately so: the reference accumulates in float64 while
// the kernel accumulates in fp32 through the WMMA and rounds the result to bf16, so a few 1e-3
// relative is the floor and anything tighter would be measuring the reference's precision. The
// DECODE itself is gated bit-exactly, separately, in decode_test.hip -- that split matters,
// because a tolerance on the decode would hide exactly the ordering bugs this format invites.
#include "escha_kernels.h"
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
  FILE *f = fopen("gemm_vectors.bin", "rb");
  if (!f) { printf("gemm_vectors.bin missing (run gen_gemm_vectors.py)\n"); return 1; }
  unsigned int magic, ncase;
  if (fread(&magic, 4, 1, f) != 1 || magic != 0xE5C7A001u) { printf("bad magic\n"); return 1; }
  if (fread(&ncase, 4, 1, f) != 1) return 1;

  constexpr int DWN = 8, BND = DWN * 16;
  int bad = 0;
  printf("%-6s %-5s %-6s %-7s %-5s %10s  %s\n", "K", "M", "N", "Kdim", "KS", "rel", "verdict");
  for (unsigned int c = 0; c < ncase; ++c) {
    int K, M, N, Kdim;
    if (fread(&K, 4, 1, f) != 1) break;
    if (fread(&M, 4, 1, f) != 1 || fread(&N, 4, 1, f) != 1 || fread(&Kdim, 4, 1, f) != 1) break;
    const int words = (Kdim / 16) * (N / 16) * (256 * K / 32);
    std::vector<unsigned int> hw(words);
    std::vector<unsigned short> hA((size_t)M * Kdim);
    std::vector<float> hC((size_t)M * N);
    if (fread(hw.data(), 4, words, f) != (size_t)words) break;
    if (fread(hA.data(), 2, hA.size(), f) != hA.size()) break;
    if (fread(hC.data(), 4, hC.size(), f) != hC.size()) break;

    unsigned int *dW; __half *dA; float *dP; int *dCnt; __bf16 *dC;
    const int nblk = (N + BND - 1) / BND;
    HIP_CHECK(hipMalloc(&dW, (size_t)words * 4));
    HIP_CHECK(hipMalloc(&dA, hA.size() * 2));
    HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * 2));
    HIP_CHECK(hipMemcpy(dW, hw.data(), (size_t)words * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));

    for (int KS : {1, 4})
     for (int KBLK : {1, 2, 4, 8}) {
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
      HIP_CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
      const int tm = (M + 15) / 16;
      const dim3 g(nblk, 1, KS), b(DWN * 32);
// KB=1 keeps PAD=8 (the original layout); KB>=2 uses PAD=4, which makes the shared stride
// 2 mod 4 dwords and the 16 fragment lanes bank-conflict-free.
#define L(KS_, TM_, K_, KB_, PAD_) hipLaunchKernelGGL( \
        (escha_gemm_decode<DWN, KS_, TM_, K_, KB_, PAD_>), g, b, 0, 0, \
        dA, dW, dP, dCnt, dC, M, N, Kdim)
#define BY_TM(KS_, K_, KB_, PAD_) do { if (tm==1) L(KS_,1,K_,KB_,PAD_); \
        else if (tm==2) L(KS_,2,K_,KB_,PAD_); else if (tm==3) L(KS_,3,K_,KB_,PAD_); \
        else L(KS_,4,K_,KB_,PAD_); } while (0)
#define BY_KB(KS_, K_) do { if (KBLK==1) BY_TM(KS_,K_,1,8); else if (KBLK==2) BY_TM(KS_,K_,2,4); \
        else if (KBLK==4) BY_TM(KS_,K_,4,4); else BY_TM(KS_,K_,8,4); } while (0)
      if (K == 2) { if (KS == 1) BY_KB(1, 2); else BY_KB(4, 2); }
      else        { if (KS == 1) BY_KB(1, 3); else BY_KB(4, 3); }
#undef L
#undef BY_TM
#undef BY_KB
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
      printf("%-6d %-5d %-6d %-7d %-5d %-4d %10.2e  %s\n", K, M, N, Kdim, KS, KBLK, rel, ok ? "OK" : "FAIL");
    }
    hipFree(dW); hipFree(dA); hipFree(dP); hipFree(dCnt); hipFree(dC);
  }
  fclose(f);
  printf("\ngemm gate: %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
