// Correctness gate and benchmark for the ParoQuant W4A8 kernels.
// Compiles par_kernels.h UNCHANGED -- same source the serving module (radiance_paroquant.hip)
// compiles. Gates, in dependency order:
//   1. e4m3 encode/decode: 256-code round-trip identity and nearest-value optimality of encode.
//   2. pq_rotate_quant vs a CPU reference that mirrors the kernel's float op order exactly.
//   3. decode + prefill GEMM vs a CPU double reference over the SAME codes, with random
//      asymmetric zeros and multi-partition boundaries; rows past M must be untouched.
//   4. end-to-end (X -> prologue -> GEMM) vs a pure-math reference, looser tolerance (fp8 noise).
// --bench times the real per-rank model shapes against the streaming roofline.
#include "par_kernels.h"

#include <vector>
#include <random>
#include <string>
#include <algorithm>

__global__ __launch_bounds__(256) void pq_stream_probe(const unsigned int *__restrict__ W,
                                                       size_t nwords, float *__restrict__ sink) {
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * 256;
  unsigned int acc = 0;
  for (; i < nwords; i += stride) acc ^= W[i];
  if (acc == 0xFFFFFFFFu) sink[0] = 1.f;
}

static float lut_val(int code) { return (float)(code - 8); }   // what ar_lut4 feeds the WMMA

struct Shape {
  int N, K, P, pb1, pb2;
  const char *name;
};

// Per-rank (TP2) shapes of Qwen3.8-27B-PARO, plus ragged edge cases.
static const Shape SHAPES[] = {
    {7168, 5120, 3, 6144, 6656, "qkv"},          // q 6144 | k 512 | v 512
    {5120, 3072, 1, 1 << 30, 1 << 30, "o_proj"},
    {17408, 5120, 2, 8704, 1 << 30, "gate_up"},
    {5120, 8704, 1, 1 << 30, 1 << 30, "down"},
    {8192, 5120, 2, 5120, 1 << 30, "in_proj"},   // qkv 5120 | z 3072
    {200, 384, 1, 1 << 30, 1 << 30, "ragged"},
    {640, 512, 2, 512, 1 << 30, "tiny2p"},
};

struct Buffers {
  int M, N, K, P, G;
  std::vector<unsigned int> W;       // [N, K/8]
  std::vector<__half> SZ;           // [G, N, 2]
  std::vector<int> ZP;              // [G, N] raw zero points (reference only)
  std::vector<unsigned char> A;     // [P, M, K]
  std::vector<float> ASG, RS;       // [P, M, G]
  std::vector<float> AS;            // [P, M] (per-token mode)
};

static void fill_weights(Buffers &b, std::mt19937 &rng) {
  std::uniform_int_distribution<int> code(0, 15);
  std::uniform_real_distribution<float> scl(0.002f, 0.05f);
  const int kw = b.K / 8;
  b.W.assign((size_t)b.N * kw, 0u);
  for (int n = 0; n < b.N; ++n)
    for (int k = 0; k < b.K; ++k)
      b.W[(size_t)n * kw + k / 8] |= ((unsigned int)code(rng)) << (4 * (k % 8));
  b.SZ.assign((size_t)b.G * b.N * 2, __float2half(0.f));
  b.ZP.assign((size_t)b.G * b.N, 0);
  for (int g = 0; g < b.G; ++g)
    for (int n = 0; n < b.N; ++n) {
      const float s = scl(rng);
      const int zp = code(rng);
      b.ZP[(size_t)g * b.N + n] = zp;
      b.SZ[((size_t)g * b.N + n) * 2 + 0] = __float2half(s);
      b.SZ[((size_t)g * b.N + n) * 2 + 1] = __float2half(s * (float)(zp - 8));
    }
}

static void fill_acts(Buffers &b, std::mt19937 &rng, bool subnormal_heavy = false,
                      bool ptok = false) {
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::uniform_real_distribution<float> mag(0.05f, 4.f);
  b.A.assign((size_t)b.P * b.M * b.K, 0);
  b.ASG.assign((size_t)b.P * b.M * b.G, 0.f);
  b.RS.assign((size_t)b.P * b.M * b.G, 0.f);
  b.AS.assign((size_t)b.P * b.M, 0.f);
  for (int p = 0; p < b.P; ++p)
    for (int m = 0; m < b.M; ++m) {
      std::vector<float> v((size_t)b.G * PQ_GROUP);
      std::vector<float> gscale(b.G);
      for (int g = 0; g < b.G; ++g) {
        float amax = 0.f;
        const float span = mag(rng);
        for (int c = 0; c < PQ_GROUP; ++c) {
          float &x = v[(size_t)g * PQ_GROUP + c];
          x = gauss(rng) * span;
          // Force real subnormal CODES (|code| < 2^-6): values below amax/448/64. This is the
          // probe for hardware-WMMA-vs-pq_e4m3_decode subnormal agreement.
          if (subnormal_heavy && (c % 5) == 0) x *= 1e-6f;
          amax = std::max(amax, fabsf(x));
        }
        gscale[g] = std::max(amax / 448.f, 1e-10f);
      }
      float astok = 1e-10f;
      for (int g = 0; g < b.G; ++g) astok = std::max(astok, gscale[g]);
      b.AS[(size_t)p * b.M + m] = astok;
      for (int g = 0; g < b.G; ++g) {
        const float scale = ptok ? astok : gscale[g];
        float rs = 0.f;
        for (int c = 0; c < PQ_GROUP; ++c) {
          const unsigned char code = pq_e4m3_encode(v[(size_t)g * PQ_GROUP + c] / scale);
          b.A[(((size_t)p * b.M + m) * b.K) + g * PQ_GROUP + c] = code;
          rs += pq_e4m3_decode(code);
        }
        b.ASG[((size_t)p * b.M + m) * b.G + g] = gscale[g];
        // per-group mode: RS = rowsum*asg; per-token mode: RS = plain rowsum
        b.RS[((size_t)p * b.M + m) * b.G + g] = ptok ? rs : rs * gscale[g];
      }
    }
}

// Reference over the same codes, kernel algebra (lut + zscale), double accumulate.
static void ref_gemm2(const Buffers &b, int pb1, int pb2, std::vector<float> &C,
                      bool ptok = false) {
  C.assign((size_t)b.M * b.N, 0.f);
  const int kw = b.K / 8;
  for (int m = 0; m < b.M; ++m)
    for (int n = 0; n < b.N; ++n) {
      const int p = (n >= pb1 ? 1 : 0) + (n >= pb2 ? 1 : 0);
      double acc = 0.0;
      for (int g = 0; g < b.G; ++g) {
        const float sc = __half2float(b.SZ[((size_t)g * b.N + n) * 2 + 0]);
        const float zsc = __half2float(b.SZ[((size_t)g * b.N + n) * 2 + 1]);
        double dot = 0.0;
        for (int k = g * PQ_GROUP; k < (g + 1) * PQ_GROUP; ++k) {
          const int code = (b.W[(size_t)n * kw + k / 8] >> (4 * (k % 8))) & 0xF;
          const float a = pq_e4m3_decode(b.A[(((size_t)p * b.M + m) * b.K) + k]);
          dot += (double)a * (double)lut_val(code);
        }
        const double rsv = b.RS[((size_t)p * b.M + m) * b.G + g];
        if (ptok) acc += sc * dot - zsc * rsv;                     // As applied after the loop
        else acc += (double)b.ASG[((size_t)p * b.M + m) * b.G + g] * sc * dot - zsc * rsv;
      }
      if (ptok) acc *= (double)b.AS[(size_t)p * b.M + m];
      C[(size_t)m * b.N + n] = (float)acc;
    }
}

static float bf16_to_f(unsigned short u) {
  unsigned int w = (unsigned int)u << 16;
  float f;
  memcpy(&f, &w, 4);
  return f;
}

int main(int argc, char **argv) {
  const bool bench = (argc > 1 && std::string(argv[1]) == "--bench");
  std::mt19937 rng(20260831);

  // ---------------- 1. e4m3 encode/decode ----------------
  {
    int bad = 0;
    for (int c = 0; c < 256; ++c) {
      if (((c >> 3) & 0xF) == 0xF && (c & 7) == 7) continue;   // NaN slots
      const float v = pq_e4m3_decode((unsigned char)c);
      if (pq_e4m3_encode(v) != (unsigned char)c && !(v == 0.f && (c == 0x80 || c == 0)))
        ++bad;
    }
    // Encode optimality: encode(v) must be at least as close to v as any code.
    std::uniform_real_distribution<float> u(-500.f, 500.f);
    int subopt = 0;
    for (int i = 0; i < 200000; ++i) {
      const float v = u(rng);
      const unsigned char e = pq_e4m3_encode(v);
      const float de = fabsf(pq_e4m3_decode(e) - v);
      for (int c = 0; c < 256; ++c) {
        if (((c >> 3) & 0xF) == 0xF && (c & 7) == 7) continue;
        if (fabsf(pq_e4m3_decode((unsigned char)c) - v) < de - 1e-7f * fabsf(v)) {
          ++subopt;
          break;
        }
      }
    }
    printf("e4m3: round-trip bad=%d  suboptimal-encodes=%d  %s\n", bad, subopt,
           (bad || subopt) ? "FAIL" : "ok");
    if (bad || subopt) return 1;
  }

  int failures = 0;

  // ---------------- 2. rotate+quant prologue ----------------
  for (const int M : {1, 3, 8, 64, 300}) {
    const int K = 640, P = 2, krot = 8, G = K / PQ_GROUP;
    std::uniform_real_distribution<float> uang(-1.5f, 1.5f);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> ucs(0.5f, 2.f);

    // Build records: per (p, r, g), a random disjoint pairing of the 128 channels.
    std::vector<unsigned short> T((size_t)P * krot * (K / 2) * 4, 0);
    for (int p = 0; p < P; ++p)
      for (int r = 0; r < krot; ++r)
        for (int g = 0; g < G; ++g) {
          std::vector<int> perm(PQ_GROUP);
          for (int i = 0; i < PQ_GROUP; ++i) perm[i] = i;
          std::shuffle(perm.begin(), perm.end(), rng);
          for (int t = 0; t < 64; ++t) {
            const int i = perm[2 * t], j = perm[2 * t + 1];
            const float th = uang(rng);
            unsigned short *rec =
                &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
            rec[0] = (unsigned short)(i | (j << 8));
            const __half c = __float2half(cosf(th)), s = __float2half(sinf(th));
            memcpy(rec + 1, &c, 2);
            memcpy(rec + 2, &s, 2);
          }
        }
    std::vector<__half> CS((size_t)P * K);
    for (auto &v : CS) v = __float2half(ucs(rng));
    std::vector<unsigned short> X((size_t)M * K);   // bf16 bits
    for (auto &v : X) {
      const float f = gauss(rng);
      unsigned int w;
      memcpy(&w, &f, 4);
      v = (unsigned short)(w >> 16);
    }

    // CPU reference, mirroring kernel float op order.
    std::vector<unsigned char> refA((size_t)P * M * K);
    std::vector<float> refASG((size_t)P * M * G), refRS((size_t)P * M * G);
    for (int p = 0; p < P; ++p)
      for (int m = 0; m < M; ++m)
        for (int g = 0; g < G; ++g) {
          float v[PQ_GROUP];
          for (int c = 0; c < PQ_GROUP; ++c)
            v[c] = bf16_to_f(X[(size_t)m * K + g * PQ_GROUP + c]) *
                   __half2float(CS[(size_t)p * K + g * PQ_GROUP + c]);
          for (int r = 0; r < krot; ++r) {
            float nv[PQ_GROUP];
            memcpy(nv, v, sizeof(v));
            for (int t = 0; t < 64; ++t) {
              const unsigned short *rec =
                  &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
              const int i = rec[0] & 0xFF, j = rec[0] >> 8;
              __half ch, sh;
              memcpy(&ch, rec + 1, 2);
              memcpy(&sh, rec + 2, 2);
              const float c = __half2float(ch), s = __half2float(sh);
              nv[i] = fmaf(c, v[i], s * v[j]);
              nv[j] = fmaf(c, v[j], -s * v[i]);
            }
            memcpy(v, nv, sizeof(v));
          }
          float amax = 0.f;
          for (int c = 0; c < PQ_GROUP; ++c) amax = std::max(amax, fabsf(v[c]));
          // amax * (1/448) exactly as the kernel computes it -- amax/448 rounds differently
          // and flips rare encode ties.
          const float scale = std::max(amax * (1.f / 448.f), 1e-10f);
          const float inv = 1.f / scale;
          float rs = 0.f;
          for (int c = 0; c < PQ_GROUP; ++c) {
            const unsigned char b = pq_e4m3_encode(v[c] * inv);
            refA[(((size_t)p * M + m) * K) + g * PQ_GROUP + c] = b;
            rs += pq_e4m3_decode(b);
          }
          refASG[((size_t)p * M + m) * G + g] = scale;
          refRS[((size_t)p * M + m) * G + g] = rs * scale;
        }

    unsigned short *dX;
    unsigned short *dT;
    __half *dCS;
    unsigned char *dA;
    float *dASG, *dRS;
    HIP_CHECK(hipMalloc(&dX, X.size() * 2));
    HIP_CHECK(hipMalloc(&dT, T.size() * 2));
    HIP_CHECK(hipMalloc(&dCS, CS.size() * 2));
    HIP_CHECK(hipMalloc(&dA, refA.size()));
    HIP_CHECK(hipMalloc(&dASG, refASG.size() * 4));
    HIP_CHECK(hipMalloc(&dRS, refRS.size() * 4));
    HIP_CHECK(hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCS, CS.data(), CS.size() * 2, hipMemcpyHostToDevice));
    const dim3 grid(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P);
    hipLaunchKernelGGL(pq_rotate_quant, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                       (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<unsigned char> gA(refA.size());
    std::vector<float> gASG(refASG.size()), gRS(refRS.size());
    HIP_CHECK(hipMemcpy(gA.data(), dA, gA.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gASG.data(), dASG, gASG.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gRS.data(), dRS, gRS.size() * 4, hipMemcpyDeviceToHost));
    size_t codebad = 0;
    for (size_t i = 0; i < gA.size(); ++i)
      if (gA[i] != refA[i]) ++codebad;
    float srel = 0.f, rrel = 0.f;
    for (size_t i = 0; i < gASG.size(); ++i) {
      srel = std::max(srel, fabsf(gASG[i] - refASG[i]) / std::max(fabsf(refASG[i]), 1e-12f));
      rrel = std::max(rrel, fabsf(gRS[i] - refRS[i]) / std::max(fabsf(refRS[i]), 1e-3f));
    }
    const bool pass = codebad * 500 < gA.size() && srel < 1e-5f && rrel < 1e-4f;
    if (!pass) ++failures;
    printf("rotquant M=%-4d code-mismatch=%zu/%zu asg-rel=%.2e rs-rel=%.2e %s\n", M, codebad,
           gA.size(), srel, rrel, pass ? "OK" : "FAIL");
    hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dA); hipFree(dASG); hipFree(dRS);
  }

  // ---------------- 3. GEMM decode + prefill vs reference codes ----------------
  const int Ms[] = {1, 2, 3, 5, 8, 13, 16, 17, 32, 40, 64, 72, 128, 300};
  for (const Shape &sh : SHAPES) {
    Buffers b;
    b.N = sh.N; b.K = sh.K; b.P = sh.P; b.G = sh.K / PQ_GROUP;
    for (const int M : Ms) {
      b.M = M;
      fill_weights(b, rng);
      fill_acts(b, rng, /*subnormal_heavy=*/(M == 13));
      std::vector<float> ref;
      ref_gemm2(b, sh.pb1, sh.pb2, ref);

      unsigned char *dA;
      unsigned int *dW;
      __half *dSZ;
      float *dASG, *dRS, *dP;
      int *dCnt;
      __bf16 *dC;
      const int kw = b.K / 8;
      constexpr int DWN = 8, BND = DWN * 16;
      const int nblk = (b.N + BND - 1) / BND;
      HIP_CHECK(hipMalloc(&dA, b.A.size()));
      HIP_CHECK(hipMalloc(&dW, b.W.size() * 4));
      HIP_CHECK(hipMalloc(&dSZ, b.SZ.size() * 2));
      HIP_CHECK(hipMalloc(&dASG, b.ASG.size() * 4));
      HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4));
      HIP_CHECK(hipMalloc(&dP, (size_t)4 * std::max(M, 1) * b.N * 4));
      HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
      const int Mpad = M + 8;   // rows past M must stay untouched
      HIP_CHECK(hipMalloc(&dC, (size_t)Mpad * b.N * 2));
      HIP_CHECK(hipMemcpy(dA, b.A.data(), b.A.size(), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dW, b.W.data(), b.W.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dSZ, b.SZ.data(), b.SZ.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dASG, b.ASG.data(), b.ASG.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));

      for (int variant = 0; variant < 2; ++variant) {   // 0 = decode band, 1 = prefill
        if (variant == 0 && M > 64) continue;
        HIP_CHECK(hipMemset(dC, 0x7B, (size_t)Mpad * b.N * 2));   // poison
        if (variant == 0) {
          const int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : 4;
          const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
          const dim3 grid(nblk, 1, ks), block(DWN * 32);
#define H_DEC(KS_, TM_)                                                                     \
          hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<DWN, KS_, TM_>), grid, block, 0, 0,    \
                             dA, dW, dSZ, dASG, dRS, dP, dCnt, dC, M, b.N, b.K, sh.pb1,      \
                             sh.pb2)
#define H_DEC_T(KS_)                                                                        \
          do { if (tm == 1) H_DEC(KS_, 1); else if (tm == 2) H_DEC(KS_, 2);                  \
               else if (tm == 3) H_DEC(KS_, 3); else H_DEC(KS_, 4); } while (0)
          if (ks == 1) H_DEC_T(1);
          else if (ks == 2) H_DEC_T(2);
          else H_DEC_T(4);
#undef H_DEC_T
#undef H_DEC
        } else {
          constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
          const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
              block(AR_NTHREADS);
          hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true>), grid, block, 0, 0, dA, dW,
                             dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
        }
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned short> hC((size_t)Mpad * b.N);
        HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < b.N; ++n) {
            const double d = bf16_to_f(hC[(size_t)m * b.N + n]) - ref[(size_t)m * b.N + n];
            num += d * d;
            den += (double)ref[(size_t)m * b.N + n] * ref[(size_t)m * b.N + n];
          }
        int touched = 0;
        for (int m = M; m < Mpad; ++m)
          for (int n = 0; n < b.N; ++n)
            if (hC[(size_t)m * b.N + n] != 0x7B7B) ++touched;
        const double rel = sqrt(num / (den > 0 ? den : 1));
        const bool pass = rel < 5e-3 && touched == 0;
        if (!pass) ++failures;
        printf("  %-8s %-7s N=%-6d K=%-5d P=%d M=%-4d rel=%.2e rows-past-M=%-4d %s\n",
               sh.name, variant ? "prefill" : "decode", b.N, b.K, b.P, M, rel, touched,
               pass ? "OK" : "FAIL");
      }
      // ---- per-token (PTOK) prefill kernel, the serving path for M > decode band ----
      {
        fill_acts(b, rng, /*subnormal_heavy=*/(M == 13), /*ptok=*/true);
        std::vector<float> refp;
        ref_gemm2(b, sh.pb1, sh.pb2, refp, /*ptok=*/true);
        HIP_CHECK(hipMemcpy(dA, b.A.data(), b.A.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dASG, b.AS.data(), b.AS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dC, 0x7B, (size_t)Mpad * b.N * 2));
        constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
        const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
            block(AR_NTHREADS);
        hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, true>), grid, block, 0, 0, dA,
                           dW, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned short> hC((size_t)Mpad * b.N);
        HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < b.N; ++n) {
            const double d = bf16_to_f(hC[(size_t)m * b.N + n]) - refp[(size_t)m * b.N + n];
            num += d * d;
            den += (double)refp[(size_t)m * b.N + n] * refp[(size_t)m * b.N + n];
          }
        int touched = 0;
        for (int m = M; m < Mpad; ++m)
          for (int n = 0; n < b.N; ++n)
            if (hC[(size_t)m * b.N + n] != 0x7B7B) ++touched;
        const double rel = sqrt(num / (den > 0 ? den : 1));
        const bool pass = rel < 5e-3 && touched == 0;
        if (!pass) ++failures;
        printf("  %-8s %-7s N=%-6d K=%-5d P=%d M=%-4d rel=%.2e rows-past-M=%-4d %s\n",
               sh.name, "ptok", b.N, b.K, b.P, M, rel, touched, pass ? "OK" : "FAIL");
      }
      hipFree(dA); hipFree(dW); hipFree(dSZ); hipFree(dASG); hipFree(dRS);
      hipFree(dP); hipFree(dCnt); hipFree(dC);
    }
    if (bench) break;   // bench mode reuses the loop below instead
  }

  // ---------------- 2b. prefill prologue pipeline (pass A + pass C) ----------------
  for (const int M : {1, 9, 200}) {
    const int K = 640, P = 2, krot = 8, G = K / PQ_GROUP;
    std::uniform_real_distribution<float> uang(-1.5f, 1.5f);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> ucs(0.5f, 2.f);
    std::vector<unsigned short> T((size_t)P * krot * (K / 2) * 4, 0);
    for (int p = 0; p < P; ++p)
      for (int r = 0; r < krot; ++r)
        for (int g = 0; g < G; ++g) {
          std::vector<int> perm(PQ_GROUP);
          for (int i = 0; i < PQ_GROUP; ++i) perm[i] = i;
          std::shuffle(perm.begin(), perm.end(), rng);
          for (int t = 0; t < 64; ++t) {
            const int i = perm[2 * t], j = perm[2 * t + 1];
            const float th = uang(rng);
            unsigned short *rec =
                &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
            rec[0] = (unsigned short)(i | (j << 8));
            const __half c = __float2half(cosf(th)), sh_ = __float2half(sinf(th));
            memcpy(rec + 1, &c, 2);
            memcpy(rec + 2, &sh_, 2);
          }
        }
    std::vector<__half> CS((size_t)P * K);
    for (auto &vv : CS) vv = __float2half(ucs(rng));
    std::vector<unsigned short> X((size_t)M * K);
    for (auto &vv : X) {
      const float f = gauss(rng);
      unsigned int w;
      memcpy(&w, &f, 4);
      vv = (unsigned short)(w >> 16);
    }
    // CPU reference, mirroring kernel float op order end-to-end through pass A + C.
    std::vector<unsigned char> refA((size_t)P * M * K);
    std::vector<float> refAS((size_t)P * M), refRS((size_t)P * M * G);
    for (int p = 0; p < P; ++p)
      for (int m = 0; m < M; ++m) {
        std::vector<float> vrow((size_t)K);
        std::vector<float> gsc(G);
        for (int g = 0; g < G; ++g) {
          float v[PQ_GROUP];
          for (int c = 0; c < PQ_GROUP; ++c)
            v[c] = bf16_to_f(X[(size_t)m * K + g * PQ_GROUP + c]) *
                   __half2float(CS[(size_t)p * K + g * PQ_GROUP + c]);
          for (int r = 0; r < krot; ++r) {
            float nv[PQ_GROUP];
            memcpy(nv, v, sizeof(v));
            for (int t = 0; t < 64; ++t) {
              const unsigned short *rec =
                  &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
              const int i = rec[0] & 0xFF, j = rec[0] >> 8;
              __half ch, sh_;
              memcpy(&ch, rec + 1, 2);
              memcpy(&sh_, rec + 2, 2);
              const float c = __half2float(ch), s_ = __half2float(sh_);
              nv[i] = fmaf(c, v[i], s_ * v[j]);
              nv[j] = fmaf(c, v[j], -s_ * v[i]);
            }
            memcpy(v, nv, sizeof(v));
          }
          float amax = 0.f;
          for (int c = 0; c < PQ_GROUP; ++c) amax = std::max(amax, fabsf(v[c]));
          gsc[g] = std::max(amax * (1.f / 448.f), 1e-10f);
          // pass A rounds through bf16 scratch; mirror that
          for (int c = 0; c < PQ_GROUP; ++c) {
            unsigned int w;
            memcpy(&w, &v[c], 4);
            // bf16 RNE truncation
            const unsigned int lsb = (w >> 16) & 1u;
            w += 0x7FFFu + lsb;
            unsigned short hb = (unsigned short)(w >> 16);
            vrow[(size_t)g * PQ_GROUP + c] = bf16_to_f(hb);
          }
        }
        float astok = 1e-10f;
        for (int g = 0; g < G; ++g) astok = std::max(astok, gsc[g]);
        refAS[(size_t)p * M + m] = astok;
        const float inv = 1.f / astok;
        for (int g = 0; g < G; ++g) {
          float rsum = 0.f;
          for (int c = 0; c < PQ_GROUP; ++c) {
            const unsigned char code = pq_e4m3_encode(vrow[(size_t)g * PQ_GROUP + c] * inv);
            refA[(((size_t)p * M + m) * K) + g * PQ_GROUP + c] = code;
            rsum += pq_e4m3_decode(code);
          }
          refRS[((size_t)p * M + m) * G + g] = rsum;
        }
      }
    unsigned short *dX, *dT;
    __half *dCS;
    unsigned char *dXR, *dA;
    float *dASG, *dAS, *dRS;
    HIP_CHECK(hipMalloc(&dX, X.size() * 2));
    HIP_CHECK(hipMalloc(&dT, T.size() * 2));
    HIP_CHECK(hipMalloc(&dCS, CS.size() * 2));
    HIP_CHECK(hipMalloc(&dXR, (size_t)P * M * K * 2));
    HIP_CHECK(hipMalloc(&dA, refA.size()));
    HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4));
    HIP_CHECK(hipMalloc(&dAS, refAS.size() * 4));
    HIP_CHECK(hipMalloc(&dRS, refRS.size() * 4));
    HIP_CHECK(hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCS, CS.data(), CS.size() * 2, hipMemcpyHostToDevice));
    const dim3 grid(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P);
    hipLaunchKernelGGL(pq_rotate_quant<true>, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                       (const __bf16 *)dX, dT, dCS, dXR, dASG, dRS, M, K, krot);
    const dim3 grid2((M + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, P);
    hipLaunchKernelGGL(pq_token_quant, grid2, dim3(PQ_ROT_WAVES * 32), 0, 0,
                       (const __bf16 *)dXR, dASG, dA, dAS, dRS, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<unsigned char> gA(refA.size());
    std::vector<float> gAS(refAS.size()), gRS(refRS.size());
    HIP_CHECK(hipMemcpy(gA.data(), dA, gA.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gAS.data(), dAS, gAS.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gRS.data(), dRS, gRS.size() * 4, hipMemcpyDeviceToHost));
    size_t codebad = 0;
    for (size_t i = 0; i < gA.size(); ++i)
      if (gA[i] != refA[i]) ++codebad;
    float srel = 0.f, rrel = 0.f;
    for (size_t i = 0; i < gAS.size(); ++i)
      srel = std::max(srel, fabsf(gAS[i] - refAS[i]) / std::max(fabsf(refAS[i]), 1e-12f));
    for (size_t i = 0; i < gRS.size(); ++i)
      rrel = std::max(rrel, fabsf(gRS[i] - refRS[i]) / std::max(fabsf(refRS[i]), 1e-3f));
    const bool pass = codebad * 500 < gA.size() && srel < 1e-5f && rrel < 1e-4f;
    if (!pass) ++failures;
    printf("ptok-prologue M=%-4d code-mismatch=%zu/%zu as-rel=%.2e rs-rel=%.2e %s\n", M,
           codebad, gA.size(), srel, rrel, pass ? "OK" : "FAIL");
    hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dXR); hipFree(dA);
    hipFree(dASG); hipFree(dAS); hipFree(dRS);
  }

  printf("\ncorrectness: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  if (failures) return 1;

  if (!bench) return 0;

  // ---------------- bench: rotate+quant prologue ----------------
  for (const int M : {8, 64, 2048, 8192}) {
    for (const int P : {1, 3}) {
      const int K = 5120, krot = 8, G = K / PQ_GROUP;
      unsigned short *dX, *dT;
      __half *dCS;
      unsigned char *dA;
      float *dASG, *dRS;
      HIP_CHECK(hipMalloc(&dX, (size_t)M * K * 2));
      HIP_CHECK(hipMalloc(&dT, (size_t)P * krot * (K / 2) * 4 * 2));
      HIP_CHECK(hipMalloc(&dCS, (size_t)P * K * 2));
      HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K));
      HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4));
      HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
      HIP_CHECK(hipMemset(dX, 0x3C, (size_t)M * K * 2));
      HIP_CHECK(hipMemset(dT, 0, (size_t)P * krot * (K / 2) * 4 * 2));   // theta 0 = identity
      HIP_CHECK(hipMemset(dCS, 0x3C, (size_t)P * K * 2));
      const dim3 grid(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P);
      const int iters = M > 256 ? 50 : 200;
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      hipLaunchKernelGGL(pq_rotate_quant, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                         (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipEventRecord(e0));
      for (int i = 0; i < iters; ++i)
        hipLaunchKernelGGL(pq_rotate_quant, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                           (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms;
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      printf("bench rotquant M=%-5d P=%d K=%d  %8.2f us\n", M, P, ms * 1000.0 / iters);
      hipEventDestroy(e0); hipEventDestroy(e1);
      hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dA); hipFree(dASG); hipFree(dRS);
    }
  }

  // ---------------- bench: model shapes, decode Ms + prefill, vs stream roofline ----------------
  for (const Shape &sh : SHAPES) {
    if (std::string(sh.name) == "ragged" || std::string(sh.name) == "tiny2p") continue;
    Buffers b;
    b.N = sh.N; b.K = sh.K; b.P = sh.P; b.G = sh.K / PQ_GROUP;
    const int kw = b.K / 8;
    const int Mb[] = {1, 8, 16, 40, 64, 2048};
    for (const int M : Mb) {
      b.M = M;
      fill_weights(b, rng);
      const char *ptk_fill = getenv("PQ_BENCH_PTOK");
      fill_acts(b, rng, false, ptk_fill && *ptk_fill);
      if (ptk_fill && *ptk_fill)
        b.ASG.assign(b.AS.begin(), b.AS.end());   // dASG carries As [P, M] in ptok mode
      unsigned char *dA; unsigned int *dW; __half *dSZ; float *dASG, *dRS, *dP; int *dCnt;
      __bf16 *dC; float *dSink;
      constexpr int DWN = 8, BND = DWN * 16;
      const int nblk = (b.N + BND - 1) / BND;
      HIP_CHECK(hipMalloc(&dA, b.A.size()));
      HIP_CHECK(hipMalloc(&dW, b.W.size() * 4));
      HIP_CHECK(hipMalloc(&dSZ, b.SZ.size() * 2));
      HIP_CHECK(hipMalloc(&dASG, b.ASG.size() * 4));
      HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4));
      HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * b.N * 4));
      HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
      HIP_CHECK(hipMalloc(&dC, (size_t)M * b.N * 2));
      HIP_CHECK(hipMalloc(&dSink, 4));
      HIP_CHECK(hipMemcpy(dA, b.A.data(), b.A.size(), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dW, b.W.data(), b.W.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dSZ, b.SZ.data(), b.SZ.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dASG, b.ASG.data(), b.ASG.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));

      const int iters = M > 256 ? 20 : 200;
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      auto run = [&](bool prefill) {
        if (!prefill) {
          const int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : 4;
          const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
          const dim3 grid(nblk, 1, ks), block(DWN * 32);
#define B_DEC(KS_, TM_)                                                                     \
          hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<DWN, KS_, TM_>), grid, block, 0, 0,    \
                             dA, dW, dSZ, dASG, dRS, dP, dCnt, dC, M, b.N, b.K, sh.pb1,      \
                             sh.pb2)
          if (ks == 1) { if (tm == 1) B_DEC(1, 1); else if (tm == 2) B_DEC(1, 2);
                         else if (tm == 3) B_DEC(1, 3); else B_DEC(1, 4); }
          else if (ks == 2) { if (tm == 1) B_DEC(2, 1); else if (tm == 2) B_DEC(2, 2);
                              else if (tm == 3) B_DEC(2, 3); else B_DEC(2, 4); }
          else { if (tm == 1) B_DEC(4, 1); else if (tm == 2) B_DEC(4, 2);
                 else if (tm == 3) B_DEC(4, 3); else B_DEC(4, 4); }
#undef B_DEC
        } else {
          constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
          const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
              block(AR_NTHREADS);
          const char *abf = getenv("PQ_ABLATE_FOLD");
          const char *ptk = getenv("PQ_BENCH_PTOK");
          if (abf && *abf)
            hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 4>), grid, block, 0, 0, dA,
                               dW, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
          else if (ptk && *ptk)
            hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, true>), grid, block, 0, 0,
                               dA, dW, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
          else
            hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true>), grid, block, 0, 0, dA, dW,
                               dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
        }
      };
      const bool prefill = M > 64;
      run(prefill);   // warm
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipEventRecord(e0));
      for (int i = 0; i < iters; ++i) run(prefill);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms;
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double us = ms * 1000.0 / iters;

      // Stream probe over the weight bytes (the decode-band floor).
      const size_t nwords = b.W.size() + b.SZ.size() / 2;
      HIP_CHECK(hipEventRecord(e0));
      for (int i = 0; i < iters; ++i)
        hipLaunchKernelGGL(pq_stream_probe, dim3(1024), dim3(256), 0, 0, dW, b.W.size(), dSink);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double sus = ms * 1000.0 / iters;
      const double gbs = (double)b.W.size() * 4 * iters / (ms / 1000.0) / 1e9;
      printf("bench %-8s %-7s M=%-5d  %8.2f us  (stream %8.2f us, %.0f GB/s)  ratio %.2f\n",
             sh.name, prefill ? "prefill" : "decode", M, us, sus, gbs, us / sus);
      hipEventDestroy(e0); hipEventDestroy(e1);
      hipFree(dA); hipFree(dW); hipFree(dSZ); hipFree(dASG); hipFree(dRS);
      hipFree(dP); hipFree(dCnt); hipFree(dC); hipFree(dSink);
    }
  }
  return 0;
}
