// Correctness gate and benchmark for the AutoRound int4 W4A8 kernels.
// The kernels live in ar_kernels.h so that this harness and the shipped pybind module
// (radiance_autoround.hip) compile the SAME source -- a kernel that is tested here and
// then re-typed into the module is a kernel that has not been tested.
#include "../radiance_autoround_kernels.h"

// ---------------------------------------------------------------------------------------------
// Streaming roofline probe: reads exactly the bytes the GEMM reads, does nothing else. Reported
// alongside every GEMM number so the result is a bandwidth FRACTION rather than a raw microsecond
// count -- that makes the measurement self-normalising, and a low roofline reading is also the
// detector for the clock-depression trap (a variant that spills poisons everything after it).
__global__ __launch_bounds__(256) void ar_stream_probe(const unsigned int *__restrict__ W,
                                                       size_t nwords, float *__restrict__ sink) {
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * 256;
  unsigned int acc = 0;
  for (; i < nwords; i += stride) acc ^= W[i];
  if (acc == 0xFFFFFFFFu) sink[0] = 1.f;   // never true; keeps the loads live
}

static float e4m3_decode(unsigned char b) {
  const float s = (b >> 7) ? -1.f : 1.f;
  const int E = (b >> 3) & 0xF, m = b & 7;
  return E == 0 ? s * (float)m * 0.001953125f / 256.f * 4.f   // m * 2^-9
                : s * (1.f + m / 8.f) * powf(2.f, (float)(E - 7));
}

// Nearest-even e4m3 encode over the normal range; the harness only feeds it modest values.
static unsigned char e4m3_encode(float v) {
  if (v == 0.f || !std::isfinite(v)) return 0;
  unsigned char best = 0;
  float bd = 1e30f;
  for (int b = 0; b < 256; ++b) {
    if (((b >> 3) & 0xF) == 0xF) continue;            // NaN/inf slots
    const float d = fabsf(e4m3_decode((unsigned char)b) - v);
    if (d < bd) { bd = d; best = (unsigned char)b; }
  }
  return best;
}

struct Shape { int N, K; const char *name; };

int main(int argc, char **argv) {
  const bool bench = (argc > 1 && std::string(argv[1]) == "--bench");
  const bool imajor = getenv("AR_IMAJOR") != nullptr;
  printf("prefill variant: %s\n", imajor ? "IMAJOR (TN temp tiles)" : "group temp (TM*TN tiles)");
  const bool diag = (argc > 1 && std::string(argv[1]) == "--diag");
  if (diag) {
    // One-hot activation at k0 makes the output read back the weight table directly:
    //   C[0][n] = (code[n][k0] - 8) * s[n][g0]
    // so a wrong nibble order shows up as the wrong k, and a wrong row mapping as the wrong n.
    // A uniform-weight test cannot see either.
    const int M = 1, N = 32, K = 256, G = K / AR_GROUP, kw = K / 8;
    constexpr int DWN = 8, DKS = 4, BND = DWN * 16;
    const int nblk = (N + BND - 1) / BND;
    for (int mode = 0; mode < 3; ++mode) {
      // mode 0: code depends only on k   -> probes the k / nibble order
      // mode 1: code depends only on n   -> probes the row mapping
      std::vector<unsigned int> hW((size_t)N * kw, 0u);
      for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
          const int code = mode == 0 ? (k % 16) : mode == 1 ? (n % 16) : ((n * 7 + k * 3) % 16);
          hW[(size_t)n * kw + k / 8] |= ((unsigned int)code) << (4 * (k % 8));
        }
      std::vector<__half> hS((size_t)G * N, __float2half(1.f));
      std::vector<float> hAs(M, 1.f);
      for (int k0 = 0; k0 < 18; ++k0) {
        std::vector<unsigned char> hA((size_t)M * K, 0x00);
        hA[k0] = 0x38;   // e4m3 1.0
        unsigned char *dA; unsigned int *dW; __half *dS; float *dAs, *dP; int *dCnt; __bf16 *dC;
        HIP_CHECK(hipMalloc(&dA, (size_t)M * K));
        HIP_CHECK(hipMalloc(&dW, hW.size() * 4));
        HIP_CHECK(hipMalloc(&dS, hS.size() * 2));
        HIP_CHECK(hipMalloc(&dAs, M * 4));
        HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * N * 4));   // widest split
        HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
        HIP_CHECK(hipMalloc(&dC, (size_t)M * N * 2));
        HIP_CHECK(hipMemcpy(dA, hA.data(), (size_t)M * K, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
        HIP_CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
        hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, 1>), dim3(nblk, 1, DKS),
                           dim3(DWN * 32), 0, 0, dA, dW, dS, dAs, dP, dCnt, dC, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned short> hC((size_t)M * N);
        HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost));
        int nbad = 0;
        for (int n = 0; n < N; ++n) {
          const int wantv = (mode == 0 ? (k0 % 16) : mode == 1 ? (n % 16)
                                                   : ((n * 7 + k0 * 3) % 16)) - 8;
          unsigned int u = (unsigned int)hC[n] << 16; float gv; memcpy(&gv, &u, 4);
          if (gv != (float)wantv) ++nbad;
        }
        printf("  mode=%d k0=%-3d mismatched-n=%-3d %s\n", mode, k0, nbad,
               nbad ? "FAIL" : "ok");
        hipFree(dA); hipFree(dW); hipFree(dS); hipFree(dAs);
        hipFree(dP); hipFree(dCnt); hipFree(dC);
      }
      printf("\n");
    }
    return 0;
  }

  // ---------------- correctness ----------------
  // Sweep the real speculative M values plus the dispatch boundaries, N not a multiple of BND,
  // and K not a multiple of DKS*DBK.
  const int Ms[] = {1, 2, 3, 5, 8, 9, 13, 16, 17, 32, 40, 64, 72, 96, 127, 128};
  const Shape cshapes[] = {{128, 512, "small"}, {48, 640, "narrow-N"},
                           {5120, 8704, "down"}, {17408, 5120, "gate_up"},
                           {200, 384, "ragged"}};
  int failures = 0;
  for (const Shape &sh : cshapes) {
    const int N = sh.N, K = sh.K;
    if (K % AR_GROUP) { printf("skip %s: K not a multiple of 128\n", sh.name); continue; }
    const int G = K / AR_GROUP, kw = K / 8;

    std::vector<unsigned int> hW((size_t)N * kw);
    std::vector<__half> hS((size_t)G * N);
    for (size_t i = 0; i < hW.size(); ++i) {
      unsigned int w = 0;
      for (int j = 0; j < 8; ++j) w |= ((unsigned int)(rand() & 0xF)) << (4 * j);
      hW[i] = w;
    }
    const bool scale1 = getenv("AR_SCALE1") != nullptr;
    const bool code1 = getenv("AR_CODE1") != nullptr;
    for (size_t i = 0; i < hS.size(); ++i)
      hS[i] = __float2half(scale1 ? 1.f : ((rand() % 2000) - 1000) / 40000.f);
    if (code1) for (size_t i = 0; i < hW.size(); ++i) hW[i] = 0x99999999u;

    for (int M : Ms) {
      if (M > DEC_MTILE * 4) continue;
      if (getenv("AR_ONE") && (M != 5 || N != 128)) continue;
      std::vector<unsigned char> hA((size_t)M * K);
      std::vector<float> hAs(M);
      for (int m = 0; m < M; ++m) {
        hAs[m] = 0.5f + (rand() % 100) / 100.f;
        const bool act1 = getenv("AR_ACT1") != nullptr;
        for (int k = 0; k < K; ++k)
          hA[(size_t)m * K + k] = act1 ? 0x38 : e4m3_encode(((rand() % 2000) - 1000) / 500.f);
      }

      // Exact reference in double, matching the kernel's association order:
      //   C[m][n] = As[m] * sum_g s[n][g] * sum_{k in g} a_fp8[m][k] * (code - 8)
      std::vector<double> ref((size_t)M * N);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double tot = 0.0;
          for (int g = 0; g < G; ++g) {
            double gs = 0.0;
            for (int t = 0; t < AR_GROUP; ++t) {
              const int k = g * AR_GROUP + t;
              const unsigned int word = hW[(size_t)n * kw + k / 8];
              const int code = (int)((word >> (4 * (k % 8))) & 0xF);
              gs += (double)e4m3_decode(hA[(size_t)m * K + k]) * (double)(code - 8);
            }
            tot += (double)__half2float(hS[(size_t)g * N + n]) * gs;
          }
          ref[(size_t)m * N + n] = tot * hAs[m];
        }

      unsigned char *dA;
      unsigned int *dW;
      __half *dS;
      float *dAs, *dP;
      int *dCnt;
      __bf16 *dC;
      constexpr int DWN = 8, BND = DWN * 16;
      const int nblk = (N + BND - 1) / BND;
      // Poison the output and one extra row: an M-padded kernel writing row 15 of a 5-row output
      // is an out-of-bounds write that a relative-error check would happily pass.
      const int Cpad = M + 4;
      HIP_CHECK(hipMalloc(&dA, (size_t)M * K));
      HIP_CHECK(hipMalloc(&dW, hW.size() * 4));
      HIP_CHECK(hipMalloc(&dS, hS.size() * 2));
      HIP_CHECK(hipMalloc(&dAs, M * 4));
      HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * N * 4));   // widest split of the KS sweep
      HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
      HIP_CHECK(hipMalloc(&dC, (size_t)Cpad * N * 2));
      HIP_CHECK(hipMemcpy(dA, hA.data(), (size_t)M * K, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
      std::vector<unsigned short> poison((size_t)Cpad * N, 0x7FC0);   // bf16 NaN
      HIP_CHECK(hipMemcpy(dC, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));

      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      // Sweep every split-K width split_k_for() can return. DKS==1 takes a separate early-return
      // path that writes C directly and never touches the partial buffer, the threadfence or the
      // atomic; it is the width the launcher picks for gate_up (136 n-blocks), so leaving it
      // ungated would mean the shipped decode path is the one path never checked.
      for (int KS : {1, 2, 4}) {
      const dim3 grid(nblk, 1, KS), block(DWN * 32);
#define AR_LAUNCH(KS_, TM_)                                                                \
      hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, KS_, TM_>), grid, block, 0, 0, dA,   \
                         dW, dS, dAs, dP, dCnt, dC, M, N, K)
#define AR_BY_TM(KS_)                                                                      \
      do { if (tm == 1) AR_LAUNCH(KS_, 1); else if (tm == 2) AR_LAUNCH(KS_, 2);             \
           else if (tm == 3) AR_LAUNCH(KS_, 3); else if (tm == 4) AR_LAUNCH(KS_, 4);        \
           else if (tm == 5) AR_LAUNCH(KS_, 5); else if (tm == 6) AR_LAUNCH(KS_, 6);        \
           else if (tm == 7) AR_LAUNCH(KS_, 7); else AR_LAUNCH(KS_, 8); } while (0)
      if (KS == 1) AR_BY_TM(1); else if (KS == 2) AR_BY_TM(2); else AR_BY_TM(4);
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<unsigned short> hC((size_t)Cpad * N);
      HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost));

      // Every row, not an 8x8 corner: a corner cannot see rows M..15 of a padded tile.
      double num = 0, den = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          unsigned int u = (unsigned int)hC[(size_t)m * N + n] << 16;
          float got;
          memcpy(&got, &u, 4);
          const double r = ref[(size_t)m * N + n];
          num += (got - r) * (got - r);
          den += r * r;
        }
      const double rel = sqrt(num / (den > 0 ? den : 1));
      int touched = 0;
      for (int m = M; m < Cpad; ++m)
        for (int n = 0; n < N; ++n)
          if (hC[(size_t)m * N + n] != 0x7FC0) ++touched;

      const bool pass = rel < 5e-3 && touched == 0;
      if (!pass) ++failures;
      printf("  %-9s N=%-6d K=%-6d M=%-3d KS=%d rel=%.2e  rows-past-M-touched=%-6d %s\n", sh.name,
             N, K, M, KS, rel, touched, pass ? "OK" : "FAIL");
      HIP_CHECK(hipMemcpy(dC, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
      }   // KS

      hipFree(dA); hipFree(dW); hipFree(dS); hipFree(dAs);
      hipFree(dP); hipFree(dCnt); hipFree(dC);
    }
  }

  // ---------------- prefill correctness ----------------
  // Small shapes get a full fp64 CPU reference at large M. Production shapes are too big for that
  // (M*N*K in double), so they are cross-checked against the DECODE kernel at M=64, which the
  // sweep above has already gated against the CPU reference. Agreement there pins the prefill
  // kernel's unpack, group-fold and epilogue to a known-good path at a real shape.
  printf("\nprefill vs fp64 reference\n");
  {
    const int PMs[] = {1, 3, 16, 17, 64, 127, 256, 257, 512};
    const Shape pshapes[] = {{256, 512, "small"}, {64, 384, "narrow"}, {200, 640, "ragged"}};
    for (const Shape &sh : pshapes) {
      const int N = sh.N, K = sh.K, G = K / AR_GROUP, kw = K / 8;
      std::vector<unsigned int> hW((size_t)N * kw);
      std::vector<__half> hS((size_t)G * N);
      for (size_t i = 0; i < hW.size(); ++i) {
        unsigned int w = 0;
        for (int j = 0; j < 8; ++j) w |= ((unsigned int)(rand() & 0xF)) << (4 * j);
        hW[i] = w;
      }
      for (size_t i = 0; i < hS.size(); ++i)
        hS[i] = __float2half(((rand() % 2000) - 1000) / 40000.f);
      for (int M : PMs) {
        std::vector<unsigned char> hA((size_t)M * K);
        std::vector<float> hAs(M);
        for (int m = 0; m < M; ++m) {
          hAs[m] = 0.5f + (rand() % 100) / 100.f;
          for (int k = 0; k < K; ++k)
            hA[(size_t)m * K + k] = e4m3_encode(((rand() % 2000) - 1000) / 500.f);
        }
        std::vector<double> ref((size_t)M * N);
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) {
            double tot = 0.0;
            for (int g = 0; g < G; ++g) {
              double gs = 0.0;
              for (int t = 0; t < AR_GROUP; ++t) {
                const int k = g * AR_GROUP + t;
                const unsigned int word = hW[(size_t)n * kw + k / 8];
                const int code = (int)((word >> (4 * (k % 8))) & 0xF);
                gs += (double)e4m3_decode(hA[(size_t)m * K + k]) * (double)(code - 8);
              }
              tot += (double)__half2float(hS[(size_t)g * N + n]) * gs;
            }
            ref[(size_t)m * N + n] = tot * hAs[m];
          }
        unsigned char *dA; unsigned int *dW; __half *dS; float *dAs; __bf16 *dC;
        const int Cpad = M + 4;
        HIP_CHECK(hipMalloc(&dA, (size_t)M * K));
        HIP_CHECK(hipMalloc(&dW, hW.size() * 4));
        HIP_CHECK(hipMalloc(&dS, hS.size() * 2));
        HIP_CHECK(hipMalloc(&dAs, M * 4));
        HIP_CHECK(hipMalloc(&dC, (size_t)Cpad * N * 2));
        HIP_CHECK(hipMemcpy(dA, hA.data(), (size_t)M * K, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
        std::vector<unsigned short> poison((size_t)Cpad * N, 0x7FC0);
        HIP_CHECK(hipMemcpy(dC, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));
        // Sweep both N-tile widths: the launcher picks TN=4 above AR_TN4_MIN_M, so TN=4 is a
        // shipped path and must be gated, not just benchmarked.
        for (int TNv : {2, 4}) {
        const int BNF_T = AR_WN * TNv * 16;
        const dim3 g((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), b(AR_NTHREADS);
        if (TNv == 4) {
          if (imajor)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<4, true>), g, b, 0, 0, dA, dW, dS, dAs, dC, M, N, K);
          else
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<4, false>), g, b, 0, 0, dA, dW, dS, dAs, dC, M, N, K);
        } else if (imajor)
          hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<2, true>), g, b, 0, 0, dA, dW, dS, dAs, dC, M, N, K);
        else
          hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<2, false>), g, b, 0, 0, dA, dW, dS, dAs, dC, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned short> hC((size_t)Cpad * N);
        HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) {
            unsigned int u = (unsigned int)hC[(size_t)m * N + n] << 16; float got;
            memcpy(&got, &u, 4);
            const double r = ref[(size_t)m * N + n];
            num += (got - r) * (got - r); den += r * r;
          }
        const double rel = sqrt(num / (den > 0 ? den : 1));
        int touched = 0;
        for (int m = M; m < Cpad; ++m)
          for (int n = 0; n < N; ++n)
            if (hC[(size_t)m * N + n] != 0x7FC0) ++touched;
        const bool pass = rel < 5e-3 && touched == 0;
        if (!pass) ++failures;
        printf("  %-8s N=%-5d K=%-5d M=%-4d TN=%d rel=%.2e  rows-past-M-touched=%-5d %s\n",
               sh.name, N, K, M, TNv, rel, touched, pass ? "OK" : "FAIL");
        HIP_CHECK(hipMemcpy(dC, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));
        }   // TNv
        hipFree(dA); hipFree(dW); hipFree(dS); hipFree(dAs); hipFree(dC);
      }
    }
  }

  // ---------------- prefill vs decode on production shapes ----------------
  printf("\nprefill vs decode, M=64, production shapes\n");
  {
    const Shape xshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}, {5120, 5120, "out"}};
    const int M = 64;
    for (const Shape &sh : xshapes) {
      const int N = sh.N, K = sh.K, G = K / AR_GROUP, kw = K / 8;
      std::vector<unsigned int> hW((size_t)N * kw);
      std::vector<__half> hS((size_t)G * N);
      for (size_t i = 0; i < hW.size(); ++i) {
        unsigned int w = 0;
        for (int j = 0; j < 8; ++j) w |= ((unsigned int)(rand() & 0xF)) << (4 * j);
        hW[i] = w;
      }
      for (size_t i = 0; i < hS.size(); ++i)
        hS[i] = __float2half(((rand() % 2000) - 1000) / 40000.f);
      std::vector<unsigned char> hA((size_t)M * K);
      std::vector<float> hAs(M);
      for (int m = 0; m < M; ++m) {
        hAs[m] = 0.5f + (rand() % 100) / 100.f;
        for (int k = 0; k < K; ++k)
          hA[(size_t)m * K + k] = e4m3_encode(((rand() % 2000) - 1000) / 500.f);
      }
      unsigned char *dA; unsigned int *dW; __half *dS; float *dAs, *dP; int *dCnt;
      __bf16 *dCp, *dCd;
      constexpr int DWN = 8, DKS = 4, BND = DWN * 16, TN = 2, BNF_T = AR_WN * TN * 16;
      const int nblk = (N + BND - 1) / BND;
      HIP_CHECK(hipMalloc(&dA, (size_t)M * K));
      HIP_CHECK(hipMalloc(&dW, hW.size() * 4));
      HIP_CHECK(hipMalloc(&dS, hS.size() * 2));
      HIP_CHECK(hipMalloc(&dAs, M * 4));
      HIP_CHECK(hipMalloc(&dP, (size_t)DKS * M * N * 4));
      HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
      HIP_CHECK(hipMalloc(&dCp, (size_t)M * N * 2));
      HIP_CHECK(hipMalloc(&dCd, (size_t)M * N * 2));
      HIP_CHECK(hipMemcpy(dA, hA.data(), (size_t)M * K, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dS, hS.data(), hS.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dAs, hAs.data(), M * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
      if (imajor)
        hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, true>),
                           dim3((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
                           dim3(AR_NTHREADS), 0, 0, dA, dW, dS, dAs, dCp, M, N, K);
      else
        hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, false>),
                           dim3((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
                           dim3(AR_NTHREADS), 0, 0, dA, dW, dS, dAs, dCp, M, N, K);
      hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, 4>), dim3(nblk, 1, DKS),
                         dim3(DWN * 32), 0, 0, dA, dW, dS, dAs, dP, dCnt, dCd, M, N, K);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned short> hCp((size_t)M * N), hCd((size_t)M * N);
      HIP_CHECK(hipMemcpy(hCp.data(), dCp, hCp.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hCd.data(), dCd, hCd.size() * 2, hipMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (size_t i = 0; i < hCp.size(); ++i) {
        unsigned int up = (unsigned int)hCp[i] << 16, ud = (unsigned int)hCd[i] << 16;
        float fp, fd; memcpy(&fp, &up, 4); memcpy(&fd, &ud, 4);
        num += (double)(fp - fd) * (fp - fd); den += (double)fd * fd;
      }
      const double rel = sqrt(num / (den > 0 ? den : 1));
      const bool pass = rel < 5e-3;
      if (!pass) ++failures;
      printf("  %-8s N=%-6d K=%-6d rel(prefill,decode)=%.2e  %s\n", sh.name, N, K, rel,
             pass ? "OK" : "FAIL");
      hipFree(dA); hipFree(dW); hipFree(dS); hipFree(dAs);
      hipFree(dP); hipFree(dCnt); hipFree(dCp); hipFree(dCd);
    }
  }

  printf("\ncorrectness: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  if (failures || !bench) return failures ? 1 : 0;

  // ---------------- benchmark ----------------
  // Interleave the GEMM and the roofline probe and repeat: the first variant measured on a shape
  // reads ~17-20% inflated, so min-of-N within one position is not enough.
  printf("\n%-10s %-5s %-6s %-7s %10s %10s %10s %8s\n", "shape", "M", "N", "K", "gemm us",
         "roof us", "GB/s", "%roof");
  const Shape bshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}, {5120, 5120, "out"}};
  const int bMs[] = {5, 8, 16, 40, 64};
  for (const Shape &sh : bshapes) {
    const int N = sh.N, K = sh.K, G = K / AR_GROUP, kw = K / 8;
    std::vector<unsigned int> hW((size_t)N * kw);
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = (unsigned int)rand();
    unsigned int *dW;
    __half *dS;
    float *dAs, *dP, *dSink;
    unsigned char *dA;
    int *dCnt;
    __bf16 *dC;
    constexpr int DWN = 8, DKS = 4, BND = DWN * 16;
    const int nblk = (N + BND - 1) / BND, Mmax = 64;
    HIP_CHECK(hipMalloc(&dW, hW.size() * 4));
    HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));
    HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));
    HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
    HIP_CHECK(hipMalloc(&dAs, Mmax * 4));
    HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
    HIP_CHECK(hipMalloc(&dP, (size_t)DKS * Mmax * N * 4));
    HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
    HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
    HIP_CHECK(hipMalloc(&dSink, 4));

    // bytes the GEMM must stream: 4-bit codes + fp16 group scales
    const double wbytes = (double)N * K / 2.0 + (double)G * N * 2.0;
    const size_t nwords = hW.size();

    for (int M : bMs) {
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
      const dim3 grid(nblk, 1, DKS), block(DWN * 32);
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      float gbest = 1e30f, rbest = 1e30f;
      // The bench fixes KS at the widest split; split_k_for()'s shape-dependent choice is
      // measured separately in decopt/tilebench, not here.
#define AR_BLAUNCH(TM_)                                                                    \
      hipLaunchKernelGGL((ar_int4_fp8_gemm_decode<DWN, DKS, TM_>), grid, block, 0, 0, dA,   \
                         dW, dS, dAs, dP, dCnt, dC, M, N, K)
#define AR_BBY_TM                                                                          \
      do { if (tm == 1) AR_BLAUNCH(1); else if (tm == 2) AR_BLAUNCH(2);                     \
           else if (tm == 3) AR_BLAUNCH(3); else AR_BLAUNCH(4); } while (0)
      for (int rep = 0; rep < 3; ++rep) {
        for (int w = 0; w < 3; ++w) AR_BBY_TM;
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(e0));
        for (int it = 0; it < 20; ++it) AR_BBY_TM;
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        gbest = std::min(gbest, ms * 1000.f / 20.f);

        for (int w = 0; w < 3; ++w)
          hipLaunchKernelGGL(ar_stream_probe, dim3(2048), dim3(256), 0, 0, dW, nwords, dSink);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(e0));
        for (int it = 0; it < 20; ++it)
          hipLaunchKernelGGL(ar_stream_probe, dim3(2048), dim3(256), 0, 0, dW, nwords, dSink);
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        rbest = std::min(rbest, ms * 1000.f / 20.f);
      }
      const double gbs = wbytes / (gbest * 1e-6) / 1e9;
      const double roofgbs = (double)nwords * 4.0 / (rbest * 1e-6) / 1e9;
      printf("%-10s %-5d %-6d %-7d %10.1f %10.1f %10.1f %7.1f%%\n", sh.name, M, N, K, gbest, rbest,
             gbs, 100.0 * gbs / roofgbs);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }
    hipFree(dW); hipFree(dS); hipFree(dA); hipFree(dAs);
    hipFree(dP); hipFree(dCnt); hipFree(dC); hipFree(dSink);
  }
#undef AR_BLAUNCH
#undef AR_BBY_TM

  // ---------------- prefill benchmark ----------------
  // Prefill is compute-bound, so the figure of merit is TFLOP/s against the fp8 WMMA ceiling
  // (412 TF/s peak, ~355 TF/s measured register-resident), not a bandwidth fraction.
  printf("\n%-10s %-6s %-6s %-7s %10s %10s %8s\n", "shape", "M", "N", "K", "us", "TFLOP/s",
         "%of355");
  {
    const Shape pshapes[] = {{17408, 5120, "gate_up"}, {5120, 8704, "down"}};
    const int pMs[] = {512, 1024, 2048, 4096};
    for (const Shape &sh : pshapes) {
      const int N = sh.N, K = sh.K, G = K / AR_GROUP, kw = K / 8;
      constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
      unsigned int *dW; __half *dS; unsigned char *dA; float *dAs; __bf16 *dC;
      const int Mmax = pMs[sizeof(pMs) / sizeof(pMs[0]) - 1];
      HIP_CHECK(hipMalloc(&dW, (size_t)N * kw * 4));
      HIP_CHECK(hipMemset(dW, 0x99, (size_t)N * kw * 4));
      HIP_CHECK(hipMalloc(&dS, (size_t)G * N * 2));
      HIP_CHECK(hipMemset(dS, 0x11, (size_t)G * N * 2));
      HIP_CHECK(hipMalloc(&dA, (size_t)Mmax * K));
      HIP_CHECK(hipMemset(dA, 0x38, (size_t)Mmax * K));
      HIP_CHECK(hipMalloc(&dAs, Mmax * 4));
      HIP_CHECK(hipMemset(dAs, 0, Mmax * 4));
      HIP_CHECK(hipMalloc(&dC, (size_t)Mmax * N * 2));
      for (int M : pMs) {
        const dim3 g((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), b(AR_NTHREADS);
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        float best = 1e30f;
        // Repeat and take the min: the first variant measured on a shape reads ~17-20% inflated.
        for (int rep = 0; rep < 3; ++rep) {
          for (int w = 0; w < 2; ++w)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, false>), g, b, 0, 0, dA, dW, dS, dAs, dC, M,
                               N, K);
          HIP_CHECK(hipDeviceSynchronize());
          HIP_CHECK(hipEventRecord(e0));
          for (int it = 0; it < 10; ++it)
            hipLaunchKernelGGL((ar_int4_fp8_gemm_prefill<TN, false>), g, b, 0, 0, dA, dW, dS, dAs, dC, M,
                               N, K);
          HIP_CHECK(hipEventRecord(e1));
          HIP_CHECK(hipEventSynchronize(e1));
          float ms;
          HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
          best = std::min(best, ms * 1000.f / 10.f);
        }
        const double tf = 2.0 * M * N * K / (best * 1e-6) / 1e12;
        printf("%-10s %-6d %-6d %-7d %10.1f %10.1f %7.1f%%\n", sh.name, M, N, K, best, tf,
               100.0 * tf / 355.0);
        hipEventDestroy(e0);
        hipEventDestroy(e1);
      }
      hipFree(dW); hipFree(dS); hipFree(dA); hipFree(dAs); hipFree(dC);
    }
  }
  return 0;
}
