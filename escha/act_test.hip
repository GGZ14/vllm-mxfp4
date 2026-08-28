// Gate the FULL escha linear layer: rotate+quantize -> GEMM -> rotate, against the reference
// runtime's chain. The GEMM gates prove the codec; this proves everything wrapped around it,
// which is where an integration goes silently wrong.
#include "escha_kernels.h"
#include "escha_act.h"
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
  FILE *f = fopen("act_vectors.bin", "rb");
  if (!f) { printf("act_vectors.bin missing\n"); return 1; }
  unsigned int magic, ncase;
  if (fread(&magic, 4, 1, f) != 1 || magic != 0xE5C7A003u) { printf("bad magic\n"); return 1; }
  if (fread(&ncase, 4, 1, f) != 1) return 1;
  int bad = 0;
  printf("%-4s %-6s %-6s %-6s %-9s %10s  %s\n", "K", "M", "IC", "OC", "path", "rel", "verdict");
  for (unsigned int c = 0; c < ncase; ++c) {
    int K, M, IC, OC;
    if (fread(&K, 4, 1, f) != 1) break;
    if (fread(&M, 4, 1, f) != 1 || fread(&IC, 4, 1, f) != 1 || fread(&OC, 4, 1, f) != 1) break;
    const int nwords = (IC / 16) * (OC / 16) * (256 * K / 32);
    std::vector<unsigned int> hw(nwords);
    std::vector<unsigned short> hx((size_t)M * IC), hrin(IC), hrout(OC);
    std::vector<float> hsin(IC), hsout(OC), hy((size_t)M * OC);
    if (fread(hw.data(), 4, nwords, f) != (size_t)nwords) break;
    if (fread(hx.data(), 2, hx.size(), f) != hx.size()) break;
    if (fread(hsin.data(), 4, IC, f) != (size_t)IC) break;
    if (fread(hrin.data(), 2, IC, f) != (size_t)IC) break;
    if (fread(hrout.data(), 2, OC, f) != (size_t)OC) break;
    if (fread(hsout.data(), 4, OC, f) != (size_t)OC) break;
    if (fread(hy.data(), 4, hy.size(), f) != hy.size()) break;

    unsigned int *dW; __bf16 *dX, *dC, *dY; __half *dRin, *dRout;
    float *dSin, *dSout, *dAs, *dP; unsigned char *dA; int *dCnt;
    const int nblk = (OC + 127) / 128;
    HIP_CHECK(hipMalloc(&dW, (size_t)nwords * 4));
    HIP_CHECK(hipMalloc(&dX, (size_t)M * IC * 2));
    HIP_CHECK(hipMalloc(&dRin, IC * 2));   HIP_CHECK(hipMalloc(&dRout, OC * 2));
    HIP_CHECK(hipMalloc(&dSin, IC * 4));   HIP_CHECK(hipMalloc(&dSout, OC * 4));
    HIP_CHECK(hipMalloc(&dA, (size_t)M * IC)); HIP_CHECK(hipMalloc(&dAs, M * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)8 * M * OC * 4));
    HIP_CHECK(hipMalloc(&dCnt, (nblk + 16) * 4)); HIP_CHECK(hipMemset(dCnt, 0, (nblk + 16) * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * OC * 2)); HIP_CHECK(hipMalloc(&dY, (size_t)M * OC * 2));
    HIP_CHECK(hipMemcpy(dW, hw.data(), (size_t)nwords * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dRin, hrin.data(), IC * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dRout, hrout.data(), OC * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dSin, hsin.data(), IC * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dSout, hsout.data(), OC * 4, hipMemcpyHostToDevice));

    float *dAmax; HIP_CHECK(hipMalloc(&dAmax, M * 4)); HIP_CHECK(hipMemset(dAmax, 0, M * 4));
    hipLaunchKernelGGL(escha_pre_amax, dim3((IC + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M),
                       dim3(ESCHA_ACT_THREADS), 0, 0, dX, dSin, dRin, dAmax, M, IC);
    hipLaunchKernelGGL(escha_pre_quant, dim3((IC + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M),
                       dim3(ESCHA_ACT_THREADS), 0, 0, dX, dSin, dRin, dAmax, dA, dAs, M, IC);
    const bool dec = M <= 64;
    if (dec) {
      const int tm = (M + 15) / 16, KS = escha_decode_split_k(nblk, IC / 16);
#define D(KS_, TM_, K_) hipLaunchKernelGGL((escha_gemm_decode_fp8<8, KS_, TM_, K_, 4, 8, false>), \
        dim3(nblk, 1, KS_), dim3(256), 0, 0, dA, dW, dAs, dP, dCnt, dC, M, OC, IC)
#define DTM(KS_, K_) do { if (tm==1) D(KS_,1,K_); else if (tm==2) D(KS_,2,K_); \
        else if (tm==3) D(KS_,3,K_); else D(KS_,4,K_); } while (0)
      if (KS == 8) { if (K == 2) DTM(8, 2); else DTM(8, 3); }
      else         { if (K == 2) DTM(4, 2); else DTM(4, 3); }
#undef D
#undef DTM
    } else {
      const int EB = EP_WN * 2 * 16;
      const dim3 g((OC + EB - 1) / EB, (M + 2 * EP_BMF - 1) / (2 * EP_BMF)), b(EP_NTHREADS);
      if (K == 2) hipLaunchKernelGGL((escha_gemm_prefill<2, 2, 8>), g, b, 0, 0, dA, dW, dAs, dC, M, OC, IC);
      else        hipLaunchKernelGGL((escha_gemm_prefill<2, 3, 8>), g, b, 0, 0, dA, dW, dAs, dC, M, OC, IC);
    }
    hipLaunchKernelGGL(escha_post_rot, dim3((OC + ESCHA_HAD * ESCHA_ACT_WAVES - 1) / (ESCHA_HAD * ESCHA_ACT_WAVES), M),
                       dim3(ESCHA_ACT_THREADS), 0, 0, dC, dRout, dSout, dY, M, OC, OC, 0);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned short> got((size_t)M * OC);
    HIP_CHECK(hipMemcpy(got.data(), dY, got.size() * 2, hipMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      unsigned int u = (unsigned int)got[i] << 16; float v;
      __builtin_memcpy(&v, &u, 4);
      num += (v - hy[i]) * (v - hy[i]); den += hy[i] * hy[i];
    }
    const double rel = sqrt(num / (den > 0 ? den : 1));
    const bool ok = rel < 2e-2;      // bf16 round trip on x, C and y, plus a different sum order
    bad += !ok;
    printf("%-4d %-6d %-6d %-6d %-9s %10.2e  %s\n", K, M, IC, OC, dec ? "decode" : "prefill",
           rel, ok ? "OK" : "FAIL");
    hipFree(dW); hipFree(dX); hipFree(dRin); hipFree(dRout); hipFree(dSin); hipFree(dSout);
    hipFree(dA); hipFree(dAs); hipFree(dAmax); hipFree(dP); hipFree(dCnt); hipFree(dC); hipFree(dY);
  }
  fclose(f);
  printf("\nact gate: %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
