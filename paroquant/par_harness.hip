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


// ---- layout helpers (host) ----
// Fragment-tiled activation: [P, M, K] -> [P, Mt*16, K]; pad rows filled with 0x7F (a large
// finite code) so a kernel that let them leak would fail the gate loudly.
static std::vector<unsigned char> tile_a(const std::vector<unsigned char> &A, int P, int M,
                                         int K) {
  const int Mt = (M + 15) / 16, ks = K / 16;
  std::vector<unsigned char> T((size_t)P * Mt * 16 * K, 0x7F);
  for (int p = 0; p < P; ++p)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k) {
        const int mt = m / 16, lrow = m % 16, kst = k / 16, half = (k / 8) % 2, b = k % 8;
        T[(size_t)p * Mt * 16 * K + ((size_t)(mt * ks + kst) * 32 + lrow + 16 * half) * 8 + b] =
            A[((size_t)p * M + m) * K + k];
      }
  return T;
}
// Fragment-order weight: [N, K/8] u32 (row layout) -> one uint32 per (n-tile, k-step, lane).
static std::vector<unsigned int> permute_w(const std::vector<unsigned int> &W, int N, int K) {
  if (N % 16 || K % 16) return W;     // fragment order needs whole tiles (loader-asserted)
  const int kw = K / 8, ks = K / 16;
  std::vector<unsigned int> P((size_t)N * kw, 0u);
  for (int n = 0; n < N; ++n)
    for (int st = 0; st < ks; ++st)
      for (int half = 0; half < 2; ++half) {
        const int lane = half * 16 + (n % 16);
        P[((size_t)(n / 16) * ks + st) * 32 + lane] = W[(size_t)n * kw + st * 2 + half];
      }
  return P;
}

// ZPE operands: fp16 row-sums in fragment order [P][Mt][Gp/16][32 lanes x 8], fp16 zero-scales
// [N][Gp] (k-contiguous). Gp = G rounded up to 16, zero padded.
static std::vector<__half> build_rsh(const std::vector<float> &RS, int P, int M, int G) {
  const int Mt = (M + 15) / 16, Gp = (G + 15) & ~15, gsteps = Gp / 16;
  std::vector<__half> out((size_t)P * Mt * Gp * 16, __float2half(0.f));
  for (int p = 0; p < P; ++p)
    for (int m = 0; m < M; ++m)
      for (int g = 0; g < G; ++g) {
        const int mt = m / 16, lrow = m % 16, gs = g / 16, half = (g / 8) % 2, b = g % 8;
        out[(size_t)p * Mt * Gp * 16 + ((size_t)(mt * gsteps + gs) * 32 + lrow + 16 * half) * 8 + b] =
            __float2half(RS[((size_t)p * M + m) * G + g]);
      }
  return out;
}
static std::vector<__half> build_zsh(const std::vector<__half> &SZ, int N, int G) {
  const int Gp = (G + 15) & ~15;
  std::vector<__half> out((size_t)N * Gp, __float2half(0.f));
  for (int g = 0; g < G; ++g)
    for (int n = 0; n < N; ++n) out[(size_t)n * Gp + g] = SZ[((size_t)g * N + n) * 2 + 1];
  return out;
}

// ---- bench2: DRAM-fed (rotated weight copies), ABAB-interleaved kernel A/B ----
// The original --bench keeps ONE weight buffer, which on gate_up (44.6 MB) sits inside the
// 64 MB Infinity Cache and reads at ~1170 GB/s -- twice the DRAM stream, hiding exactly the
// staging questions these variants ask. NCOPY copies rotate per launch so every weight read is
// a DRAM read, as in a real step.
struct DevW { std::vector<unsigned int *> row, perm; size_t words; };
static DevW upload_w_copies(const Buffers &b, int ncopy) {
  DevW d; d.words = b.W.size();
  const std::vector<unsigned int> Wp = permute_w(b.W, b.N, b.K);
  for (int c = 0; c < ncopy; ++c) {
    unsigned int *r, *pp;
    HIP_CHECK(hipMalloc(&r, d.words * 4));
    HIP_CHECK(hipMalloc(&pp, d.words * 4));
    HIP_CHECK(hipMemcpy(r, b.W.data(), d.words * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pp, Wp.data(), d.words * 4, hipMemcpyHostToDevice));
    d.row.push_back(r); d.perm.push_back(pp);
  }
  return d;
}
static void free_w_copies(DevW &d) {
  for (auto *r : d.row) hipFree(r);
  for (auto *pp : d.perm) hipFree(pp);
}

template <typename F>
static double time_us(F &&launch, int iters, hipEvent_t e0, hipEvent_t e1) {
  launch(0);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(e0));
  for (int i = 0; i < iters; ++i) launch(i);
  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  return ms * 1000.0 / iters;
}

static void bench2(std::mt19937 &rng, const std::string &what) {
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  const int REPS = 4;
  const char *only = getenv("PQ_SHAPE");
  for (const Shape &sh : SHAPES) {
    if (std::string(sh.name) == "ragged" || std::string(sh.name) == "tiny2p") continue;
    if (only && *only && std::string(sh.name) != only) continue;
    Buffers b;
    b.N = sh.N; b.K = sh.K; b.P = sh.P; b.G = sh.K / PQ_GROUP;
    fill_weights(b, rng);
    const size_t wbytes = b.W.size() * 4;
    const int ncopy = std::min<int>(12, std::max<int>(2, (int)((256u << 20) / wbytes) + 1));
    DevW dw = upload_w_copies(b, ncopy);
    __half *dSZ;
    HIP_CHECK(hipMalloc(&dSZ, b.SZ.size() * 2));
    HIP_CHECK(hipMemcpy(dSZ, b.SZ.data(), b.SZ.size() * 2, hipMemcpyHostToDevice));

    if (what == "pre" || what == "all") {
      const char *mlist = getenv("PQ_MS");
      std::vector<int> Ms = {512, 1024, 2048, 4096};
      if (mlist && *mlist) { Ms.clear(); for (const char *q = mlist; *q;) { Ms.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; } }
      for (const int M : Ms) {
        b.M = M;
        fill_acts(b, rng, false, /*ptok=*/true);
        const std::vector<unsigned char> At = tile_a(b.A, b.P, M, b.K);
        unsigned char *dA, *dAT; float *dAS, *dRS; __bf16 *dC;
        HIP_CHECK(hipMalloc(&dA, b.A.size()));
        HIP_CHECK(hipMalloc(&dAT, At.size()));
        HIP_CHECK(hipMalloc(&dAS, b.AS.size() * 4));
        HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4));
        HIP_CHECK(hipMalloc(&dC, (size_t)M * b.N * 2));
        HIP_CHECK(hipMemcpy(dA, b.A.data(), b.A.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAT, At.data(), At.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAS, b.AS.data(), b.AS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
        constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
        const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
        const int iters = M >= 4096 ? 10 : 20;
        struct V { const char *name; double best; };
        std::vector<V> vs = {{"ptok-row(base)", 1e30}, {"at64 hoist", 1e30}, {"at128 hoist", 1e30},
                             {"at64 nohoist", 1e30}, {"at128 nohoist", 1e30},
                             {"at64 hoist wp1", 1e30}, {"at128 hoist wp1", 1e30},
                             {"ptok-row wp1", 1e30}};
        for (int rep = 0; rep < REPS; ++rep)
          for (size_t v = 0; v < vs.size(); ++v) {
            auto L = [&](int it) {
              const unsigned int *W = dw.row[it % ncopy], *Wp = dw.perm[it % ncopy];
              switch (v) {
                case 0: hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, true, false>), grid, block, 0, 0, dA, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 1: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 64, true>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 2: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, true>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 3: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 64, false>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 4: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, false>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 5: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 64, true>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 6: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 7: hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, true, true>), grid, block, 0, 0, dA, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              }
            };
            vs[v].best = std::min(vs[v].best, time_us(L, iters, e0, e1));
          }
        const double flops = 2.0 * M * b.N * b.K;
        printf("prefill %-8s N=%-6d K=%-5d P=%d M=%-5d ncopy=%d  (best of %d, us / ratio vs base / TF)\n",
               sh.name, b.N, b.K, b.P, M, ncopy, REPS);
        for (auto &v : vs)
          printf("   %-18s %9.1f   %.3f   %6.1f\n", v.name, v.best, v.best / vs[0].best,
                 flops / v.best / 1e6);
        hipFree(dA); hipFree(dAT); hipFree(dAS); hipFree(dRS); hipFree(dC);
      }
    }

    if (what == "dec" || what == "all") {
      constexpr int DWN = 8, BND = DWN * 16;
      const int nblk = (b.N + BND - 1) / BND;
      for (const int M : {1, 5, 8, 16, 40, 48, 64}) {
        b.M = M;
        fill_acts(b, rng, false, false);
        unsigned char *dA; float *dASG, *dRS, *dP; int *dCnt; __bf16 *dC;
        HIP_CHECK(hipMalloc(&dA, b.A.size()));
        HIP_CHECK(hipMalloc(&dASG, b.ASG.size() * 4));
        HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4));
        HIP_CHECK(hipMalloc(&dP, (size_t)4 * M * b.N * 4));
        HIP_CHECK(hipMalloc(&dCnt, nblk * 4));
        HIP_CHECK(hipMalloc(&dC, (size_t)M * b.N * 2));
        HIP_CHECK(hipMemcpy(dA, b.A.data(), b.A.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dASG, b.ASG.data(), b.ASG.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dCnt, 0, nblk * 4));
        // the launcher's rule (radiance_paroquant.hip split_k_for, M <= 64 band)
        int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : 4;
        if (M > 48 && ks > 1) ks >>= 1;
        const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
        const dim3 grid(nblk, 1, ks), block(DWN * 32);
        const int iters = 100;
        struct V { const char *name; double best; };
        std::vector<V> vs = {{"row(base)", 1e30}, {"wp1", 1e30}, {"wp1+nt", 1e30}};
        for (int rep = 0; rep < REPS; ++rep)
          for (size_t v = 0; v < vs.size(); ++v) {
            auto L = [&](int it) {
              const unsigned int *W = v == 0 ? dw.row[it % ncopy] : dw.perm[it % ncopy];
#define B2_DEC(KS_, TM_, WP_, NT_)                                                              \
              hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<DWN, KS_, TM_, true, 0, WP_, NT_>), grid, \
                                 block, 0, 0, dA, W, dSZ, dASG, dRS, dP, dCnt, dC, M, b.N, b.K,   \
                                 sh.pb1, sh.pb2)
#define B2_DEC_TM(KS_, WP_, NT_)                                                                \
              do { if (tm == 1) B2_DEC(KS_, 1, WP_, NT_); else if (tm == 2) B2_DEC(KS_, 2, WP_, NT_); \
                   else if (tm == 3) B2_DEC(KS_, 3, WP_, NT_); else B2_DEC(KS_, 4, WP_, NT_); } while (0)
#define B2_DEC_KS(WP_, NT_)                                                                     \
              do { if (ks == 1) B2_DEC_TM(1, WP_, NT_); else if (ks == 2) B2_DEC_TM(2, WP_, NT_);  \
                   else B2_DEC_TM(4, WP_, NT_); } while (0)
              if (v == 0) B2_DEC_KS(false, false);
              else if (v == 1) B2_DEC_KS(true, false);
              else B2_DEC_KS(true, true);
#undef B2_DEC_KS
#undef B2_DEC_TM
#undef B2_DEC
            };
            vs[v].best = std::min(vs[v].best, time_us(L, iters, e0, e1));
          }
        const double gbs = (double)(wbytes + b.SZ.size() * 2) / vs[0].best / 1e3;
        printf("decode  %-8s N=%-6d K=%-5d M=%-3d ks=%d tm=%d ncopy=%d  (best of %d, us / ratio / GB/s)\n",
               sh.name, b.N, b.K, M, ks, tm, ncopy, REPS);
        for (auto &v : vs)
          printf("   %-12s %8.2f   %.3f   %6.0f\n", v.name, v.best, v.best / vs[0].best,
                 (double)(wbytes + b.SZ.size() * 2) / v.best / 1e3);
        (void)gbs;
        hipFree(dA); hipFree(dASG); hipFree(dRS); hipFree(dP); hipFree(dCnt); hipFree(dC);
      }
    }
    hipFree(dSZ);
    free_w_copies(dw);
  }

  if (what == "wp") {
    for (const Shape &sh : SHAPES) {
      if (std::string(sh.name) != "gate_up" && std::string(sh.name) != "qkv" &&
          std::string(sh.name) != "down") continue;
      Buffers b;
      b.N = sh.N; b.K = sh.K; b.P = sh.P; b.G = sh.K / PQ_GROUP;
      fill_weights(b, rng);
      const size_t wbytes = b.W.size() * 4;
      const int ncopy = std::min<int>(12, std::max<int>(2, (int)((256u << 20) / wbytes) + 1));
      DevW dw = upload_w_copies(b, ncopy);
      __half *dSZ;
      HIP_CHECK(hipMalloc(&dSZ, b.SZ.size() * 2));
      HIP_CHECK(hipMemcpy(dSZ, b.SZ.data(), b.SZ.size() * 2, hipMemcpyHostToDevice));
      for (const int M : {1024, 2048, 4096}) {
        b.M = M;
        fill_acts(b, rng, false, true);
        const std::vector<unsigned char> At = tile_a(b.A, b.P, M, b.K);
        unsigned char *dAT; float *dAS, *dRS; __bf16 *dC;
        HIP_CHECK(hipMalloc(&dAT, At.size()));
        HIP_CHECK(hipMalloc(&dAS, b.AS.size() * 4));
        HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4));
        HIP_CHECK(hipMalloc(&dC, (size_t)M * b.N * 2));
        HIP_CHECK(hipMemcpy(dAT, At.data(), At.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAS, b.AS.data(), b.AS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
        constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
        const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
        const int iters = M >= 4096 ? 10 : 20;
        struct V { const char *name; double best; };
        std::vector<V> vs = {{"at128 row(base)", 1e30}, {"at128 wp1 ws4", 1e30}, {"at128 wp1 ws2", 1e30},
                             {"at64 row", 1e30}, {"at64 wp1 ws2", 1e30}};
        for (int rep = 0; rep < REPS; ++rep)
          for (size_t v = 0; v < vs.size(); ++v) {
            auto L = [&](int it) {
              const unsigned int *W = dw.row[it % ncopy], *Wp = dw.perm[it % ncopy];
              switch (v) {
                case 0: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, true, 0>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 1: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 4>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 2: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 2>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 3: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 64, true, 0>), grid, block, 0, 0, dAT, W, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 4: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 64, true, 2>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              }
            };
            vs[v].best = std::min(vs[v].best, time_us(L, iters, e0, e1));
          }
        printf("wpexp %-8s M=%-5d ncopy=%d\n", sh.name, M, ncopy);
        for (auto &v : vs) printf("   %-18s %9.1f   %.3f\n", v.name, v.best, v.best / vs[0].best);
        hipFree(dAT); hipFree(dAS); hipFree(dRS); hipFree(dC);
      }
      hipFree(dSZ); free_w_copies(dw);
    }
  }

  if (what == "abl") {
    for (const Shape &sh : SHAPES) {
      if (std::string(sh.name) != "gate_up" && std::string(sh.name) != "qkv" && std::string(sh.name) != "in_proj") continue;
      if (only && *only && std::string(sh.name) != only) continue;
      Buffers b;
      b.N = sh.N; b.K = sh.K; b.P = sh.P; b.G = sh.K / PQ_GROUP;
      fill_weights(b, rng);
      const size_t wbytes = b.W.size() * 4;
      const int ncopy = std::min<int>(12, std::max<int>(2, (int)((256u << 20) / wbytes) + 1));
      DevW dw = upload_w_copies(b, ncopy);
      __half *dSZ;
      HIP_CHECK(hipMalloc(&dSZ, b.SZ.size() * 2));
      HIP_CHECK(hipMemcpy(dSZ, b.SZ.data(), b.SZ.size() * 2, hipMemcpyHostToDevice));
      for (const int M : {2048}) {
        b.M = M;
        fill_acts(b, rng, false, true);
        const std::vector<unsigned char> At = tile_a(b.A, b.P, M, b.K);
        unsigned char *dAT; float *dAS, *dRS; __bf16 *dC;
        HIP_CHECK(hipMalloc(&dAT, At.size())); HIP_CHECK(hipMalloc(&dAS, b.AS.size() * 4));
        HIP_CHECK(hipMalloc(&dRS, b.RS.size() * 4)); HIP_CHECK(hipMalloc(&dC, (size_t)M * b.N * 2));
        HIP_CHECK(hipMemcpy(dAT, At.data(), At.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dAS, b.AS.data(), b.AS.size() * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRS, b.RS.data(), b.RS.size() * 4, hipMemcpyHostToDevice));
        const std::vector<__half> RSH = build_rsh(b.RS, b.P, M, b.G), ZSH = build_zsh(b.SZ, b.N, b.G);
        __half *dRSH, *dZSH;
        HIP_CHECK(hipMalloc(&dRSH, RSH.size() * 2)); HIP_CHECK(hipMalloc(&dZSH, ZSH.size() * 2));
        HIP_CHECK(hipMemcpy(dRSH, RSH.data(), RSH.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dZSH, ZSH.data(), ZSH.size() * 2, hipMemcpyHostToDevice));
        constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
        const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
        const int iters = M >= 4096 ? 10 : 20;
        struct V { const char *name; double best; };
        // Variant order is rotated per rep (a fixed order let the last variant measure a
        // power-capped card on the long gate_up runs) and best-of-REPS is taken.
        std::vector<V> vs = {{"at128 (shipped)", 1e30}, {"no fold (temp acc)", 1e30}, {"direct acc (mxfp4-shaped)", 1e30},
                             {"at128 + epilogue zp (ZPE)", 1e30}, {"ZPE loop, no epilogue", 1e30}};
        for (int rep = 0; rep < REPS; ++rep)
          for (size_t vv = 0; vv < vs.size(); ++vv) {
            const size_t v = (vv + rep) % vs.size();
            auto L = [&](int it) {
              const unsigned int *Wp = dw.perm[it % ncopy];
              switch (v) {
                case 0: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 0>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 1: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 1>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 2: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 2>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
                case 3: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 4>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2, dRSH, dZSH); break;
                case 4: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 12>), grid, block, 0, 0, dAT, Wp, dSZ, dAS, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2, dRSH, dZSH); break;
              }
            };
            vs[v].best = std::min(vs[v].best, time_us(L, iters, e0, e1));
          }
        const double flops = 2.0 * M * b.N * b.K;
        printf("abl %-8s M=%-5d (us / ratio / TF)\n", sh.name, M);
        for (auto &v : vs) printf("   %-26s %9.1f   %.3f   %6.1f\n", v.name, v.best, v.best / vs[0].best, flops / v.best / 1e6);
        hipFree(dAT); hipFree(dAS); hipFree(dRS); hipFree(dC); hipFree(dRSH); hipFree(dZSH);
      }
      hipFree(dSZ); free_w_copies(dw);
    }
  }

  if (what == "rot" || what == "all") {
    for (const int M : {5, 8, 40, 64, 2048, 8192}) {
      for (const int P : {1, 3}) {
        const int K = 5120, krot = 8, G = K / PQ_GROUP;
        unsigned short *dX, *dT; __half *dCS; unsigned char *dA; float *dASG, *dRS;
        HIP_CHECK(hipMalloc(&dX, (size_t)M * K * 2));
        HIP_CHECK(hipMalloc(&dT, (size_t)P * krot * (K / 2) * 4 * 2));
        HIP_CHECK(hipMalloc(&dCS, (size_t)P * K * 2));
        HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K * 2));
        HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4));
        HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
        HIP_CHECK(hipMemset(dX, 0x3C, (size_t)M * K * 2));
        // real-looking records: pair t = (2t, 2t+1), theta 0.3
        {
          std::vector<unsigned short> T((size_t)P * krot * (K / 2) * 4, 0);
          const __half c = __float2half(cosf(0.3f)), sn = __float2half(sinf(0.3f));
          for (size_t e = 0; e < (size_t)P * krot * (K / 2); ++e) {
            const int t = (int)(e % 64);
            T[e * 4] = (unsigned short)((2 * t) | ((2 * t + 1) << 8));
            memcpy(&T[e * 4 + 1], &c, 2); memcpy(&T[e * 4 + 2], &sn, 2);
          }
          HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipMemset(dCS, 0x3C, (size_t)P * K * 2));
        const dim3 grid(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P), blk(PQ_ROT_WAVES * 32);
        const int iters = M > 256 ? 50 : 200;
        double t1 = 1e30, t2 = 1e30, t1r = 1e30, t2r = 1e30;
        for (int rep = 0; rep < REPS; ++rep) {
          t1 = std::min(t1, time_us([&](int) { hipLaunchKernelGGL(pq_rotate_quant<false>, grid, blk, 0, 0, (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot); }, iters, e0, e1));
          t2 = std::min(t2, time_us([&](int) { hipLaunchKernelGGL(pq_rotate_quant2<false>, grid, blk, 0, 0, (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot); }, iters, e0, e1));
          t1r = std::min(t1r, time_us([&](int) { hipLaunchKernelGGL(pq_rotate_quant<true>, grid, blk, 0, 0, (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot); }, iters, e0, e1));
          t2r = std::min(t2r, time_us([&](int) { hipLaunchKernelGGL(pq_rotate_quant2<true>, grid, blk, 0, 0, (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot); }, iters, e0, e1));
        }
        printf("rotquant M=%-5d P=%d  fused: v1 %8.2f v2 %8.2f (%.3f)   passA: v1 %8.2f v2 %8.2f (%.3f)\n",
               M, P, t1, t2, t2 / t1, t1r, t2r, t2r / t1r);
        hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dA); hipFree(dASG); hipFree(dRS);
      }
    }
  }

  if (what == "fuse" || what == "all") {
    // fused add+rms+rot vs (plain norm kernel + rotate_quant2) vs rotate_quant2 alone
    for (const int M : {1, 5, 8, 16, 40, 64}) {
      for (const int P : {1, 2, 3}) {
        const int K = 5120, krot = 8, G = K / PQ_GROUP;
        unsigned short *dY, *dR, *dW, *dT, *dHS, *dRO; __half *dCS; unsigned char *dA; float *dASG, *dRS;
        HIP_CHECK(hipMalloc(&dY, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dR, (size_t)M * K * 2));
        HIP_CHECK(hipMalloc(&dW, (size_t)K * 2)); HIP_CHECK(hipMalloc(&dT, (size_t)P * krot * (K / 2) * 8));
        HIP_CHECK(hipMalloc(&dCS, (size_t)P * K * 2)); HIP_CHECK(hipMalloc(&dHS, (size_t)M * K * 2));
        HIP_CHECK(hipMalloc(&dRO, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K));
        HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4)); HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
        HIP_CHECK(hipMemset(dY, 0x3C, (size_t)M * K * 2)); HIP_CHECK(hipMemset(dR, 0x3C, (size_t)M * K * 2));
        HIP_CHECK(hipMemset(dW, 0, (size_t)K * 2)); HIP_CHECK(hipMemset(dT, 0, (size_t)P * krot * (K / 2) * 8));
        HIP_CHECK(hipMemset(dCS, 0x3C, (size_t)P * K * 2));
        const dim3 blk(PQ_ROT_WAVES * 32);
        const int iters = 200;
        double tf[3] = {1e30, 1e30, 1e30}, tn = 1e30, tr = 1e30;
        const int gpws[3] = {1, 2, 4};
        for (int rep = 0; rep < REPS; ++rep) {
          for (int gi = 0; gi < 3; ++gi) {
            const int gpw = gpws[gi];
            tf[gi] = std::min(tf[gi], time_us([&](int) { hipLaunchKernelGGL(pq_add_rms_rot<true>, dim3(M, (G + 8 * gpw - 1) / (8 * gpw), P), blk, 0, 0, (const __bf16 *)dY, (const __bf16 *)dR, (const __bf16 *)dW, 1e-6f, dT, dCS, (__bf16 *)dHS, (__bf16 *)dRO, dA, dASG, dRS, M, K, krot, gpw); }, iters, e0, e1));
          }
          tn = std::min(tn, time_us([&](int) { hipLaunchKernelGGL(pq_add_rms_rot<false>, dim3(M, 1, 1), blk, 0, 0, (const __bf16 *)dY, (const __bf16 *)dR, (const __bf16 *)dW, 1e-6f, dT, dCS, (__bf16 *)dHS, (__bf16 *)dRO, dA, dASG, dRS, M, K, krot, 1); }, iters, e0, e1));
          tr = std::min(tr, time_us([&](int) { hipLaunchKernelGGL(pq_rotate_quant2<false>, dim3(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P), blk, 0, 0, (const __bf16 *)dHS, dT, dCS, dA, dASG, dRS, M, K, krot); }, iters, e0, e1));
        }
        printf("fuse    M=%-3d P=%d  fused gpw1 %7.2f gpw2 %7.2f gpw4 %7.2f us   norm-only %7.2f + rotquant2 %7.2f = %7.2f   (best %.3f)\n",
               M, P, tf[0], tf[1], tf[2], tn, tr, tn + tr, std::min(tf[0], std::min(tf[1], tf[2])) / (tn + tr));
        hipFree(dY); hipFree(dR); hipFree(dW); hipFree(dT); hipFree(dCS); hipFree(dHS); hipFree(dRO); hipFree(dA); hipFree(dASG); hipFree(dRS);
      }
    }
  }

  if (what == "passc" || what == "all") {
    for (const int M : {64, 2048, 8192}) {
      for (const int P : {1, 2, 3}) {
        const int K = 5120, G = K / PQ_GROUP, Mt = (M + 15) / 16;
        unsigned char *dXR, *dA, *dAT; float *dASG, *dAS, *dRS;
        HIP_CHECK(hipMalloc(&dXR, (size_t)P * M * K * 2));
        HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K));
        HIP_CHECK(hipMalloc(&dAT, (size_t)P * Mt * 16 * K));
        HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4));
        HIP_CHECK(hipMalloc(&dAS, (size_t)P * M * 4));
        HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
        HIP_CHECK(hipMemset(dXR, 0x3C, (size_t)P * M * K * 2));
        HIP_CHECK(hipMemset(dASG, 0x3C, (size_t)P * M * G * 4));
        const int iters = M > 256 ? 50 : 200;
        const dim3 g1((M + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, P), g2(Mt, P, pq_tq_zsplit(M)), blk(PQ_ROT_WAVES * 32);
        double t_row = 1e30, t_til = 1e30;
        for (int rep = 0; rep < REPS; ++rep) {
          t_row = std::min(t_row, time_us([&](int) {
            hipLaunchKernelGGL(pq_token_quant, g1, blk, 0, 0, (const __bf16 *)dXR, dASG, dA, dAS, dRS, M, K);
          }, iters, e0, e1));
          t_til = std::min(t_til, time_us([&](int) {
            hipLaunchKernelGGL(pq_token_quant_tiled, g2, blk, 0, 0, (const __bf16 *)dXR, dASG, dAT, dAS, dRS, M, K);
          }, iters, e0, e1));
        }
        printf("passC   M=%-5d P=%d K=%d   row %8.2f us   tiled %8.2f us   ratio %.3f\n", M, P, K,
               t_row, t_til, t_til / t_row);
        hipFree(dXR); hipFree(dA); hipFree(dAT); hipFree(dASG); hipFree(dAS); hipFree(dRS);
      }
    }
  }
  hipEventDestroy(e0); hipEventDestroy(e1);
}

int main(int argc, char **argv) {
  const bool bench = (argc > 1 && std::string(argv[1]) == "--bench");
  std::mt19937 rng(20260831);
  if (argc > 1 && std::string(argv[1]) == "--bench2") {
    bench2(rng, argc > 2 ? argv[2] : "all");
    return 0;
  }

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
  const bool quick = (argc > 1 && std::string(argv[1]) == "--quick");

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
    {
      // v2 (register records) must be BIT-EXACT against v1: codes, scales and row-sums.
      HIP_CHECK(hipMemset(dA, 0, refA.size()));
      HIP_CHECK(hipMemset(dASG, 0, refASG.size() * 4));
      HIP_CHECK(hipMemset(dRS, 0, refRS.size() * 4));
      hipLaunchKernelGGL(pq_rotate_quant2<false>, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                         (const __bf16 *)dX, dT, dCS, dA, dASG, dRS, M, K, krot);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned char> g2A(refA.size());
      std::vector<float> g2ASG(refASG.size()), g2RS(refRS.size());
      HIP_CHECK(hipMemcpy(g2A.data(), dA, g2A.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(g2ASG.data(), dASG, g2ASG.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(g2RS.data(), dRS, g2RS.size() * 4, hipMemcpyDeviceToHost));
      size_t d = 0;
      for (size_t i = 0; i < gA.size(); ++i) if (g2A[i] != gA[i]) ++d;
      for (size_t i = 0; i < gASG.size(); ++i) if (memcmp(&g2ASG[i], &gASG[i], 4)) ++d;
      for (size_t i = 0; i < gRS.size(); ++i) if (memcmp(&g2RS[i], &gRS[i], 4)) ++d;
      if (d) ++failures;
      printf("rotquant2 M=%-4d diffs-vs-v1=%zu %s\n", M, d, d ? "FAIL" : "OK");
    }
    hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dA); hipFree(dASG); hipFree(dRS);
  }

  // ---------------- 3. GEMM decode + prefill vs reference codes ----------------
  const int Ms[] = {1, 2, 3, 5, 8, 13, 16, 17, 32, 40, 64, 72, 128, 300};
  for (const Shape &sh : SHAPES) {
    if (quick) break;
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
      // ---- fragment-order weight (WPERM): decode + per-group prefill must be BIT-IDENTICAL to
      // the row layout (same sW tile, same arithmetic) ----
      {
        const std::vector<unsigned int> Wp = permute_w(b.W, b.N, b.K);
        unsigned int *dWp;
        HIP_CHECK(hipMalloc(&dWp, Wp.size() * 4));
        HIP_CHECK(hipMemcpy(dWp, Wp.data(), Wp.size() * 4, hipMemcpyHostToDevice));
        std::vector<unsigned short> hRow((size_t)Mpad * b.N), hPerm((size_t)Mpad * b.N);
        const bool wp_ok_shape = (b.N % 16 == 0) && (b.K % 16 == 0);
        for (int variant = 0; variant < 3 && wp_ok_shape; ++variant) {
          if (variant < 2 && M > 64) continue;   // 0 = decode wp1, 1 = decode wp1+nt, 2 = prefill wp1
          for (int layout = 0; layout < 2; ++layout) {   // 0 = row, 1 = perm
            HIP_CHECK(hipMemset(dC, 0x7B, (size_t)Mpad * b.N * 2));
            const unsigned int *Wsel = layout ? dWp : dW;
            if (variant < 2) {
              const int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : 4;
              const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
              const dim3 grid(nblk, 1, ks), block(DWN * 32);
#define H_DECW(KS_, TM_, WP_, NT_)                                                                              hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<DWN, KS_, TM_, true, 0, WP_, NT_>), grid,                                  block, 0, 0, dA, Wsel, dSZ, dASG, dRS, dP, dCnt, dC, M, b.N,                                      b.K, sh.pb1, sh.pb2)
#define H_DECW_T(KS_, WP_, NT_)                                                                                 do { if (tm == 1) H_DECW(KS_, 1, WP_, NT_); else if (tm == 2) H_DECW(KS_, 2, WP_, NT_);                    else if (tm == 3) H_DECW(KS_, 3, WP_, NT_); else H_DECW(KS_, 4, WP_, NT_); } while (0)
#define H_DECW_K(WP_, NT_)                                                                                      do { if (ks == 1) H_DECW_T(1, WP_, NT_); else if (ks == 2) H_DECW_T(2, WP_, NT_);                       else H_DECW_T(4, WP_, NT_); } while (0)
              if (layout == 0) H_DECW_K(false, false);
              else if (variant == 0) H_DECW_K(true, false);
              else H_DECW_K(true, true);
#undef H_DECW_K
#undef H_DECW_T
#undef H_DECW
            } else {
              constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
              const dim3 grid((b.N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF),
                  block(AR_NTHREADS);
              if (layout == 0)
                hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, false, false>), grid, block,
                                   0, 0, dA, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
              else
                hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, false, true>), grid, block,
                                   0, 0, dA, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2);
            }
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(layout ? hPerm.data() : hRow.data(), dC, hRow.size() * 2,
                                hipMemcpyDeviceToHost));
          }
          size_t diff = 0;
          for (size_t i = 0; i < hRow.size(); ++i) if (hRow[i] != hPerm[i]) ++diff;
          const bool pass = diff == 0;
          if (!pass) ++failures;
          printf("  %-8s %-7s N=%-6d K=%-5d P=%d M=%-4d byte-diff=%zu %s\n", sh.name,
                 variant == 0 ? "dec-wp" : variant == 1 ? "dec-wpnt" : "pre-wp", b.N, b.K, b.P, M,
                 diff, pass ? "OK" : "FAIL");
        }
        hipFree(dWp);
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

        // ---- A-tiled per-token kernel, every (LBK, WHOIST, WPERM) instantiation, same refp ----
        {
          const std::vector<unsigned char> At = tile_a(b.A, b.P, M, b.K);
          const std::vector<unsigned int> Wp = permute_w(b.W, b.N, b.K);
          unsigned char *dAT; unsigned int *dWp;
          HIP_CHECK(hipMalloc(&dAT, At.size()));
          HIP_CHECK(hipMalloc(&dWp, Wp.size() * 4));
          HIP_CHECK(hipMemcpy(dAT, At.data(), At.size(), hipMemcpyHostToDevice));
          HIP_CHECK(hipMemcpy(dWp, Wp.data(), Wp.size() * 4, hipMemcpyHostToDevice));
          const bool wp_ok_shape = (b.N % 16 == 0) && (b.K % 16 == 0);
          const char *vn[8] = {"at64h", "at128h", "at64", "at128", "at64h-wp", "at128h-wp",
                               "at64-wp", "at128-wp"};
          for (int v = 0; v < 8; ++v) {
            const bool wp = v >= 4;
            if (wp && !wp_ok_shape) continue;
            if ((b.K % 128) != 0) continue;
            HIP_CHECK(hipMemset(dC, 0x7B, (size_t)Mpad * b.N * 2));
            const unsigned int *Wsel = wp ? dWp : dW;
            switch (v) {
              case 0: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 64, true>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 1: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, true>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 2: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 64, false>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 3: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, false>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 4: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 64, true>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 5: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 6: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 64, false>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
              case 7: hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, false>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2); break;
            }
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<unsigned short> hT((size_t)Mpad * b.N);
            HIP_CHECK(hipMemcpy(hT.data(), dC, hT.size() * 2, hipMemcpyDeviceToHost));
            double num2 = 0, den2 = 0;
            for (int m = 0; m < M; ++m)
              for (int n = 0; n < b.N; ++n) {
                const double d = bf16_to_f(hT[(size_t)m * b.N + n]) - refp[(size_t)m * b.N + n];
                num2 += d * d;
                den2 += (double)refp[(size_t)m * b.N + n] * refp[(size_t)m * b.N + n];
              }
            int touched2 = 0;
            for (int m = M; m < Mpad; ++m)
              for (int n = 0; n < b.N; ++n)
                if (hT[(size_t)m * b.N + n] != 0x7B7B) ++touched2;
            const double rel2 = sqrt(num2 / (den2 > 0 ? den2 : 1));
            const bool pass2 = rel2 < 5e-3 && touched2 == 0;
            if (!pass2) ++failures;
            printf("  %-8s %-9s N=%-6d K=%-5d P=%d M=%-4d rel=%.2e rows-past-M=%-4d %s\n",
                   sh.name, vn[v], b.N, b.K, b.P, M, rel2, touched2, pass2 ? "OK" : "FAIL");
          }
          // ---- epilogue zero-point (ZPE) variant, row + fragment-order layouts ----
          const std::vector<__half> RSH = build_rsh(b.RS, b.P, M, b.G), ZSH = build_zsh(b.SZ, b.N, b.G);
          __half *dRSH, *dZSH;
          HIP_CHECK(hipMalloc(&dRSH, RSH.size() * 2)); HIP_CHECK(hipMalloc(&dZSH, ZSH.size() * 2));
          HIP_CHECK(hipMemcpy(dRSH, RSH.data(), RSH.size() * 2, hipMemcpyHostToDevice));
          HIP_CHECK(hipMemcpy(dZSH, ZSH.data(), ZSH.size() * 2, hipMemcpyHostToDevice));
          for (int v = 0; v < 2; ++v) {
            const bool wp = v == 1;
            if (wp && !wp_ok_shape) continue;
            if ((b.K % 128) != 0) continue;
            HIP_CHECK(hipMemset(dC, 0x7B, (size_t)Mpad * b.N * 2));
            const unsigned int *Wsel = wp ? dWp : dW;
            if (!wp) hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, false, 128, true, 0, 4>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2, dRSH, dZSH);
            else hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, true, 128, true, 0, 4>), grid, block, 0, 0, dAT, Wsel, dSZ, dASG, dRS, dC, M, b.N, b.K, sh.pb1, sh.pb2, dRSH, dZSH);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<unsigned short> hT((size_t)Mpad * b.N);
            HIP_CHECK(hipMemcpy(hT.data(), dC, hT.size() * 2, hipMemcpyDeviceToHost));
            double num2 = 0, den2 = 0;
            for (int m = 0; m < M; ++m)
              for (int n = 0; n < b.N; ++n) {
                const double d = bf16_to_f(hT[(size_t)m * b.N + n]) - refp[(size_t)m * b.N + n];
                num2 += d * d;
                den2 += (double)refp[(size_t)m * b.N + n] * refp[(size_t)m * b.N + n];
              }
            int touched2 = 0;
            for (int m = M; m < Mpad; ++m)
              for (int n = 0; n < b.N; ++n)
                if (hT[(size_t)m * b.N + n] != 0x7B7B) ++touched2;
            const double rel2 = sqrt(num2 / (den2 > 0 ? den2 : 1));
            const bool pass2 = rel2 < 5e-3 && touched2 == 0;
            if (!pass2) ++failures;
            printf("  %-8s %-9s N=%-6d K=%-5d P=%d M=%-4d rel=%.2e rows-past-M=%-4d %s\n",
                   sh.name, wp ? "zpe-wp" : "zpe", b.N, b.K, b.P, M, rel2, touched2, pass2 ? "OK" : "FAIL");
          }
          hipFree(dRSH); hipFree(dZSH);
          hipFree(dAT); hipFree(dWp);
        }
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
    {
      // pass A v2 bit-exact vs v1 (bf16 XR + ASG)
      unsigned char *dXR2; float *dASG2;
      HIP_CHECK(hipMalloc(&dXR2, (size_t)P * M * K * 2));
      HIP_CHECK(hipMalloc(&dASG2, (size_t)P * M * G * 4));
      hipLaunchKernelGGL(pq_rotate_quant2<true>, grid, dim3(PQ_ROT_WAVES * 32), 0, 0,
                         (const __bf16 *)dX, dT, dCS, dXR2, dASG2, dRS, M, K, krot);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned char> h1((size_t)P * M * K * 2), h2((size_t)P * M * K * 2);
      std::vector<float> a1((size_t)P * M * G), a2((size_t)P * M * G);
      HIP_CHECK(hipMemcpy(h1.data(), dXR, h1.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(h2.data(), dXR2, h2.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(a1.data(), dASG, a1.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(a2.data(), dASG2, a2.size() * 4, hipMemcpyDeviceToHost));
      size_t d = 0;
      for (size_t i = 0; i < h1.size(); ++i) if (h1[i] != h2[i]) ++d;
      for (size_t i = 0; i < a1.size(); ++i) if (memcmp(&a1[i], &a2[i], 4)) ++d;
      if (d) ++failures;
      printf("passA-v2      M=%-4d diffs-vs-v1=%zu %s\n", M, d, d ? "FAIL" : "OK");
      hipFree(dXR2); hipFree(dASG2);
    }
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
    {
      // Tiled pass C: codes must equal the row kernel's codes EXACTLY (same inv, same encode),
      // AS exactly, RS to fp32 reassociation.
      const int Mt = (M + 15) / 16;
      unsigned char *dAT; float *dAS2, *dRS2;
      HIP_CHECK(hipMalloc(&dAT, (size_t)P * Mt * 16 * K));
      HIP_CHECK(hipMalloc(&dAS2, refAS.size() * 4));
      HIP_CHECK(hipMalloc(&dRS2, refRS.size() * 4));
      HIP_CHECK(hipMemset(dAT, 0x11, (size_t)P * Mt * 16 * K));
      hipLaunchKernelGGL(pq_token_quant_tiled, dim3(Mt, P, pq_tq_zsplit(M)),
                         dim3(PQ_ROT_WAVES * 32), 0, 0, (const __bf16 *)dXR, dASG, dAT, dAS2,
                         dRS2, M, K);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned char> gAT((size_t)P * Mt * 16 * K);
      std::vector<float> gAS2(refAS.size()), gRS2(refRS.size());
      HIP_CHECK(hipMemcpy(gAT.data(), dAT, gAT.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(gAS2.data(), dAS2, gAS2.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(gRS2.data(), dRS2, gRS2.size() * 4, hipMemcpyDeviceToHost));
      const std::vector<unsigned char> expT = tile_a(gA, P, M, K);   // row kernel's codes, tiled
      size_t cbad = 0;
      for (int p = 0; p < P; ++p)
        for (int m = 0; m < M; ++m)
          for (int k = 0; k < K; ++k) {
            const int ks = K / 16, mt = m / 16, lrow = m % 16, kst = k / 16, half = (k / 8) % 2;
            const size_t idx = (size_t)p * Mt * 16 * K +
                               ((size_t)(mt * ks + kst) * 32 + lrow + 16 * half) * 8 + (k % 8);
            if (gAT[idx] != expT[idx]) ++cbad;
          }
      float asd = 0.f, rsd = 0.f;
      for (size_t i = 0; i < gAS2.size(); ++i) asd = std::max(asd, fabsf(gAS2[i] - gAS[i]));
      for (size_t i = 0; i < gRS2.size(); ++i)
        rsd = std::max(rsd, fabsf(gRS2[i] - gRS[i]) / std::max(fabsf(gRS[i]), 1e-3f));
      const bool pass2 = cbad == 0 && asd == 0.f && rsd < 1e-5f;
      if (!pass2) ++failures;
      printf("ptok-tiled    M=%-4d code-mismatch=%zu as-absdiff=%.1e rs-rel=%.2e %s\n", M, cbad,
             asd, rsd, pass2 ? "OK" : "FAIL");
      hipFree(dAT); hipFree(dAS2); hipFree(dRS2);
    }
    hipFree(dX); hipFree(dT); hipFree(dCS); hipFree(dXR); hipFree(dA);
    hipFree(dASG); hipFree(dAS); hipFree(dRS);
  }

  // ---------------- 2c. fused add + rmsnorm + rotate + quant ----------------
  for (const int M : {1, 5, 8, 40, 64}) {
    const int K = 5120, P = 2, krot = 8, G = K / PQ_GROUP;
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
            unsigned short *rec = &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
            rec[0] = (unsigned short)(i | (j << 8));
            const __half c = __float2half(cosf(th)), sh_ = __float2half(sinf(th));
            memcpy(rec + 1, &c, 2); memcpy(rec + 2, &sh_, 2);
          }
        }
    std::vector<__half> CS((size_t)P * K);
    for (auto &vv : CS) vv = __float2half(ucs(rng));
    auto rbf = [&](float f) { unsigned int w; memcpy(&w, &f, 4); return (unsigned short)(w >> 16); };
    std::vector<unsigned short> Yh((size_t)M * K), Rh((size_t)M * K), Wh(K);
    for (auto &vv : Yh) vv = rbf(gauss(rng));
    for (auto &vv : Rh) vv = rbf(gauss(rng) * 3.f);
    for (auto &vv : Wh) vv = rbf(gauss(rng) * 0.1f);
    const float eps = 1e-6f;
    // CPU reference of the norm (torch op order): v fp32, res bf16(v), var = mean(v^2),
    // hs = bf16((v*rsqrt(var+eps))*(w+1)).
    std::vector<unsigned short> refRO((size_t)M * K), refHS((size_t)M * K);
    for (int m = 0; m < M; ++m) {
      std::vector<float> v(K);
      double ssq = 0;
      for (int k = 0; k < K; ++k) {
        v[k] = bf16_to_f(Yh[(size_t)m * K + k]) + bf16_to_f(Rh[(size_t)m * K + k]);
        ssq += (double)v[k] * v[k];
      }
      const float inv = 1.f / sqrtf((float)(ssq / K) + eps);
      for (int k = 0; k < K; ++k) {
        unsigned int w; float f = v[k]; memcpy(&w, &f, 4);
        // bf16 RNE
        unsigned int lsb = (w >> 16) & 1u; w += 0x7FFFu + lsb; refRO[(size_t)m * K + k] = (unsigned short)(w >> 16);
        float hs = (v[k] * inv) * (bf16_to_f(Wh[k]) + 1.f);
        memcpy(&w, &hs, 4); lsb = (w >> 16) & 1u; w += 0x7FFFu + lsb; refHS[(size_t)m * K + k] = (unsigned short)(w >> 16);
      }
    }
    unsigned short *dY, *dR, *dW, *dT, *dHS, *dRO, *dHS2, *dRO2;
    __half *dCS; unsigned char *dA, *dA2; float *dASG, *dRS, *dASG2, *dRS2;
    HIP_CHECK(hipMalloc(&dY, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dR, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dW, (size_t)K * 2)); HIP_CHECK(hipMalloc(&dT, T.size() * 2));
    HIP_CHECK(hipMalloc(&dCS, CS.size() * 2));
    HIP_CHECK(hipMalloc(&dHS, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dRO, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dHS2, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dRO2, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K)); HIP_CHECK(hipMalloc(&dA2, (size_t)P * M * K));
    HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4)); HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
    HIP_CHECK(hipMalloc(&dASG2, (size_t)P * M * G * 4)); HIP_CHECK(hipMalloc(&dRS2, (size_t)P * M * G * 4));
    HIP_CHECK(hipMemcpy(dY, Yh.data(), Yh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dR, Rh.data(), Rh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, Wh.data(), Wh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCS, CS.data(), CS.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dHS, 0, (size_t)M * K * 2)); HIP_CHECK(hipMemset(dRO, 0, (size_t)M * K * 2));
    // fused
    // gate every gpw the launcher can pick (1, 2, 4) -- the grid.y changes with it
    for (const int gpw : {1, 2, 4}) {
    HIP_CHECK(hipMemset(dA, 0, (size_t)P * M * K));
    HIP_CHECK(hipMemset(dASG, 0, (size_t)P * M * G * 4)); HIP_CHECK(hipMemset(dRS, 0, (size_t)P * M * G * 4));
    HIP_CHECK(hipMemset(dHS, 0, (size_t)M * K * 2)); HIP_CHECK(hipMemset(dRO, 0, (size_t)M * K * 2));
    hipLaunchKernelGGL(pq_add_rms_rot<true>, dim3(M, (G + PQ_ROT_WAVES * gpw - 1) / (PQ_ROT_WAVES * gpw), P),
                       dim3(PQ_ROT_WAVES * 32), 0, 0, (const __bf16 *)dY, (const __bf16 *)dR,
                       (const __bf16 *)dW, eps, dT, dCS, (__bf16 *)dHS, (__bf16 *)dRO, dA, dASG,
                       dRS, M, K, krot, gpw);
    // reference chain on GPU: plain norm kernel + rotate_quant2
    hipLaunchKernelGGL(pq_add_rms_rot<false>, dim3(M, 1, 1), dim3(PQ_ROT_WAVES * 32), 0, 0,
                       (const __bf16 *)dY, (const __bf16 *)dR, (const __bf16 *)dW, eps, dT, dCS,
                       (__bf16 *)dHS2, (__bf16 *)dRO2, dA2, dASG2, dRS2, M, K, krot, 1);
    hipLaunchKernelGGL(pq_rotate_quant2<false>, dim3(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P),
                       dim3(PQ_ROT_WAVES * 32), 0, 0, (const __bf16 *)dHS2, dT, dCS, dA2, dASG2,
                       dRS2, M, K, krot);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<unsigned short> hHS((size_t)M * K), hRO((size_t)M * K), hHS2((size_t)M * K), hRO2((size_t)M * K);
    std::vector<unsigned char> hA((size_t)P * M * K), hA2((size_t)P * M * K);
    std::vector<float> hASG((size_t)P * M * G), hRS((size_t)P * M * G), hASG2((size_t)P * M * G), hRS2((size_t)P * M * G);
    HIP_CHECK(hipMemcpy(hHS.data(), dHS, hHS.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hRO.data(), dRO, hRO.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hHS2.data(), dHS2, hHS2.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hRO2.data(), dRO2, hRO2.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hA.data(), dA, hA.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hA2.data(), dA2, hA2.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hASG.data(), dASG, hASG.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hASG2.data(), dASG2, hASG2.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hRS.data(), dRS, hRS.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hRS2.data(), dRS2, hRS2.size() * 4, hipMemcpyDeviceToHost));
    size_t dro = 0, dhs = 0, dro_ref = 0, dhs_ref = 0, da = 0, dsc = 0;
    for (size_t i = 0; i < hHS.size(); ++i) {
      if (hHS[i] != hHS2[i]) ++dhs;
      if (hRO[i] != hRO2[i]) ++dro;
      if (hHS[i] != refHS[i]) ++dhs_ref;
      if (hRO[i] != refRO[i]) ++dro_ref;
    }
    for (size_t i = 0; i < hA.size(); ++i) if (hA[i] != hA2[i]) {
      if (da < 6) printf("      mismatch idx=%zu (p=%zu m=%zu g=%zu c=%zu) fused=%02x chain=%02x\n", i,
                         i / ((size_t)M * K), (i / K) % M, (i % K) / PQ_GROUP, i % PQ_GROUP, hA[i], hA2[i]);
      ++da;
    }
    for (size_t i = 0; i < hASG.size(); ++i) if (memcmp(&hASG[i], &hASG2[i], 4) || memcmp(&hRS[i], &hRS2[i], 4)) ++dsc;
    // fused == plain-norm + rotate_quant2 must be bit-exact; vs the CPU reference the residual
    // is exact and hs may differ by an ulp flip a few times per million (reduction order).
    const bool pass = dro == 0 && dhs == 0 && da == 0 && dsc == 0 && dro_ref == 0 &&
                      dhs_ref <= std::max<size_t>(2, hHS.size() / 50000);   // <= 20 ppm, floor 2
    if (!pass) ++failures;
    printf("addrmsrot     M=%-4d P=%d gpw=%d fused-vs-chain: hs=%zu ro=%zu codes=%zu sc=%zu | vs-cpu: ro=%zu hs=%zu/%zu %s\n",
           M, P, gpw, dhs, dro, da, dsc, dro_ref, dhs_ref, hHS.size(), pass ? "OK" : "FAIL");
    }
    hipFree(dY); hipFree(dR); hipFree(dW); hipFree(dT); hipFree(dCS); hipFree(dHS); hipFree(dRO);
    hipFree(dHS2); hipFree(dRO2); hipFree(dA); hipFree(dA2); hipFree(dASG); hipFree(dRS);
    hipFree(dASG2); hipFree(dRS2);
  }

  // ---------------- 2d. elementwise / gdn-norm producers + rotate + quant ----------------
  for (const int MODE : {0, 1, 2}) {
    for (const int M : {1, 5, 40, 64}) {
      const int N = 3072, P = 1, krot = 8, G = N / PQ_GROUP;
      std::uniform_real_distribution<float> uang(-1.5f, 1.5f);
      std::normal_distribution<float> gauss(0.f, 1.f);
      std::uniform_real_distribution<float> ucs(0.5f, 2.f);
      std::vector<unsigned short> T((size_t)P * krot * (N / 2) * 4, 0);
      for (int r = 0; r < krot; ++r)
        for (int g = 0; g < G; ++g) {
          std::vector<int> perm(PQ_GROUP);
          for (int i = 0; i < PQ_GROUP; ++i) perm[i] = i;
          std::shuffle(perm.begin(), perm.end(), rng);
          for (int t = 0; t < 64; ++t) {
            const int i = perm[2 * t], j = perm[2 * t + 1];
            const float th = uang(rng);
            unsigned short *rec = &T[(((size_t)r) * (N / 2) + (size_t)g * 64 + t) * 4];
            rec[0] = (unsigned short)(i | (j << 8));
            const __half c = __float2half(cosf(th)), sh_ = __float2half(sinf(th));
            memcpy(rec + 1, &c, 2); memcpy(rec + 2, &sh_, 2);
          }
        }
      std::vector<__half> CS((size_t)N);
      for (auto &vv : CS) vv = __float2half(ucs(rng));
      auto rbf = [&](float f) { unsigned int w; memcpy(&w, &f, 4); return (unsigned short)(w >> 16); };
      auto bfr = [&](float f) { unsigned int w; memcpy(&w, &f, 4); unsigned int lsb = (w >> 16) & 1u; w += 0x7FFFu + lsb; unsigned short hb = (unsigned short)(w >> 16); return bf16_to_f(hb); };
      const int XW = (MODE == 0) ? 2 * N : N;
      std::vector<unsigned short> Xh((size_t)M * XW), Yh((size_t)M * N), Wh(PQ_GROUP);
      for (auto &vv : Xh) vv = rbf(gauss(rng) * 2.f);
      for (auto &vv : Yh) vv = rbf(gauss(rng) * 2.f);
      for (auto &vv : Wh) vv = rbf(gauss(rng) * 0.3f + 1.f);
      const float eps = 1e-6f;
      // CPU reference of the producer (torch op order)
      std::vector<unsigned short> refHS((size_t)M * N);
      for (int m = 0; m < M; ++m)
        for (int g = 0; g < G; ++g) {
          float inv = 0.f;
          if (MODE == 2) {
            double ssq = 0;
            for (int c = 0; c < PQ_GROUP; ++c) { const float x = bf16_to_f(Xh[(size_t)m * N + g * PQ_GROUP + c]); ssq += (double)x * x; }
            inv = 1.f / sqrtf((float)(ssq / PQ_GROUP) + eps);
          }
          for (int c = 0; c < PQ_GROUP; ++c) {
            const int k = g * PQ_GROUP + c;
            float v;
            if (MODE == 0) {
              const float gg = bf16_to_f(Xh[(size_t)m * XW + k]), u = bf16_to_f(Xh[(size_t)m * XW + N + k]);
              const float t = bfr(gg / (1.f + expf(-gg)));
              v = bfr(t * u);
            } else if (MODE == 1) {
              const float x = bf16_to_f(Xh[(size_t)m * N + k]), z = bf16_to_f(Yh[(size_t)m * N + k]);
              const float sg = bfr(1.f / (1.f + expf(-z)));
              v = bfr(x * sg);
            } else {
              const float x = bf16_to_f(Xh[(size_t)m * N + k]), z = bf16_to_f(Yh[(size_t)m * N + k]);
              const float sg = z / (1.f + expf(-z));
              v = bfr(((x * inv) * bf16_to_f(Wh[c])) * sg);
            }
            refHS[(size_t)m * N + k] = rbf(v);
          }
        }
      unsigned short *dX, *dY, *dW, *dT, *dHS, *dHS2;
      __half *dCS; unsigned char *dA, *dA2; float *dASG, *dRS, *dASG2, *dRS2;
      HIP_CHECK(hipMalloc(&dX, Xh.size() * 2)); HIP_CHECK(hipMalloc(&dY, Yh.size() * 2));
      HIP_CHECK(hipMalloc(&dW, Wh.size() * 2)); HIP_CHECK(hipMalloc(&dT, T.size() * 2));
      HIP_CHECK(hipMalloc(&dCS, CS.size() * 2));
      HIP_CHECK(hipMalloc(&dHS, (size_t)M * N * 2)); HIP_CHECK(hipMalloc(&dHS2, (size_t)M * N * 2));
      HIP_CHECK(hipMalloc(&dA, (size_t)M * N)); HIP_CHECK(hipMalloc(&dA2, (size_t)M * N));
      HIP_CHECK(hipMalloc(&dASG, (size_t)M * G * 4)); HIP_CHECK(hipMalloc(&dRS, (size_t)M * G * 4));
      HIP_CHECK(hipMalloc(&dASG2, (size_t)M * G * 4)); HIP_CHECK(hipMalloc(&dRS2, (size_t)M * G * 4));
      HIP_CHECK(hipMemcpy(dX, Xh.data(), Xh.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dY, Yh.data(), Yh.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dW, Wh.data(), Wh.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dCS, CS.data(), CS.size() * 2, hipMemcpyHostToDevice));
      for (const int gpw : {1, 2}) {
        HIP_CHECK(hipMemset(dA, 0, (size_t)M * N)); HIP_CHECK(hipMemset(dASG, 0, (size_t)M * G * 4));
        HIP_CHECK(hipMemset(dRS, 0, (size_t)M * G * 4)); HIP_CHECK(hipMemset(dHS, 0, (size_t)M * N * 2));
        const dim3 gf(M, (G + PQ_ROT_WAVES * gpw - 1) / (PQ_ROT_WAVES * gpw), 1), gp(M, (G + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, 1), blk(PQ_ROT_WAVES * 32);
#define EW(MODE_, ROT_, GRID_, HS_, A_, ASG_, RS_, GPW_)                                                 \
        hipLaunchKernelGGL((pq_ew_rot<MODE_, ROT_>), GRID_, blk, 0, 0, (const __bf16 *)dX,               \
                           (const __bf16 *)dY, (long)N, (const __bf16 *)dW, eps, dT, dCS, (__bf16 *)HS_, \
                           A_, ASG_, RS_, M, N, krot, GPW_)
        if (MODE == 0) { EW(0, true, gf, dHS, dA, dASG, dRS, gpw); EW(0, false, gp, dHS2, dA2, dASG2, dRS2, 1); }
        else if (MODE == 1) { EW(1, true, gf, dHS, dA, dASG, dRS, gpw); EW(1, false, gp, dHS2, dA2, dASG2, dRS2, 1); }
        else { EW(2, true, gf, dHS, dA, dASG, dRS, gpw); EW(2, false, gp, dHS2, dA2, dASG2, dRS2, 1); }
#undef EW
        hipLaunchKernelGGL(pq_rotate_quant2<false>, dim3(G, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, 1),
                           blk, 0, 0, (const __bf16 *)dHS2, dT, dCS, dA2, dASG2, dRS2, M, N, krot);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned short> hHS((size_t)M * N), hHS2((size_t)M * N);
        std::vector<unsigned char> hA((size_t)M * N), hA2((size_t)M * N);
        std::vector<float> hASG((size_t)M * G), hRS((size_t)M * G), hASG2((size_t)M * G), hRS2((size_t)M * G);
        HIP_CHECK(hipMemcpy(hHS.data(), dHS, hHS.size() * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hHS2.data(), dHS2, hHS2.size() * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hA.data(), dA, hA.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hA2.data(), dA2, hA2.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hASG.data(), dASG, hASG.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hASG2.data(), dASG2, hASG2.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hRS.data(), dRS, hRS.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hRS2.data(), dRS2, hRS2.size() * 4, hipMemcpyDeviceToHost));
        size_t dhs = 0, dhs_ref = 0, da = 0, dsc = 0;
        for (size_t i = 0; i < hHS.size(); ++i) { if (hHS[i] != hHS2[i]) ++dhs; if (hHS[i] != refHS[i]) ++dhs_ref; }
        for (size_t i = 0; i < hA.size(); ++i) if (hA[i] != hA2[i]) ++da;
        for (size_t i = 0; i < hASG.size(); ++i) if (memcmp(&hASG[i], &hASG2[i], 4) || memcmp(&hRS[i], &hRS2[i], 4)) ++dsc;
        const bool pass = dhs == 0 && da == 0 && dsc == 0 && dhs_ref <= std::max<size_t>(2, hHS.size() / 50000);
        if (!pass) ++failures;
        printf("ewrot mode=%d  M=%-4d gpw=%d fused-vs-chain: hs=%zu codes=%zu sc=%zu | vs-cpu hs=%zu/%zu %s\n",
               MODE, M, gpw, dhs, da, dsc, dhs_ref, hHS.size(), pass ? "OK" : "FAIL");
      }
      hipFree(dX); hipFree(dY); hipFree(dW); hipFree(dT); hipFree(dCS); hipFree(dHS); hipFree(dHS2);
      hipFree(dA); hipFree(dA2); hipFree(dASG); hipFree(dRS); hipFree(dASG2); hipFree(dRS2);
    }
  }

  // ---------------- 2e. fused all-reduce kernel, single-rank loopback ----------------
  // peer scratch = my scratch and the flags point at myself, so the push lands in my own slot
  // and the flag is satisfied by my own release: the reduce is in + in. Must be BIT-EXACT with
  // pq_add_rms_rot on y = bf16(2 * in) (the plain kernel's bf16-rounded AR output).
  for (const int M : {1, 5, 40, 64}) {
    const int K = 5120, P = 2, krot = 8, G = K / PQ_GROUP;
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
            unsigned short *rec = &T[(((size_t)p * krot + r) * (K / 2) + (size_t)g * 64 + t) * 4];
            rec[0] = (unsigned short)(i | (j << 8));
            const __half c = __float2half(cosf(th)), sh_ = __float2half(sinf(th));
            memcpy(rec + 1, &c, 2); memcpy(rec + 2, &sh_, 2);
          }
        }
    std::vector<__half> CS((size_t)P * K);
    for (auto &vv : CS) vv = __float2half(ucs(rng));
    auto rbf = [&](float f) { unsigned int w; memcpy(&w, &f, 4); return (unsigned short)(w >> 16); };
    auto bfr = [&](float f) { unsigned int w; memcpy(&w, &f, 4); unsigned int lsb = (w >> 16) & 1u; w += 0x7FFFu + lsb; return (unsigned short)(w >> 16); };
    std::vector<unsigned short> Yh((size_t)M * K), Y2h((size_t)M * K), Rh((size_t)M * K), Wh(K);
    for (size_t i = 0; i < Yh.size(); ++i) { const float f = gauss(rng); Yh[i] = rbf(f); Y2h[i] = bfr(bf16_to_f(Yh[i]) * 2.f); }
    for (auto &vv : Rh) vv = rbf(gauss(rng) * 3.f);
    for (auto &vv : Wh) vv = rbf(gauss(rng) * 0.1f);
    const float eps = 1e-6f;
    unsigned short *dY, *dY2, *dR, *dW, *dT, *dHS, *dRO, *dHS2, *dRO2;
    __half *dCS; unsigned char *dA, *dA2; float *dASG, *dRS, *dASG2, *dRS2;
    unsigned int *dScr, *dFl, *dSeq;
    const long slot_bytes = (long)M * K * 2;
    HIP_CHECK(hipMalloc(&dY, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dY2, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dR, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dW, (size_t)K * 2));
    HIP_CHECK(hipMalloc(&dT, T.size() * 2)); HIP_CHECK(hipMalloc(&dCS, CS.size() * 2));
    HIP_CHECK(hipMalloc(&dHS, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dRO, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dHS2, (size_t)M * K * 2)); HIP_CHECK(hipMalloc(&dRO2, (size_t)M * K * 2));
    HIP_CHECK(hipMalloc(&dA, (size_t)P * M * K)); HIP_CHECK(hipMalloc(&dA2, (size_t)P * M * K));
    HIP_CHECK(hipMalloc(&dASG, (size_t)P * M * G * 4)); HIP_CHECK(hipMalloc(&dRS, (size_t)P * M * G * 4));
    HIP_CHECK(hipMalloc(&dASG2, (size_t)P * M * G * 4)); HIP_CHECK(hipMalloc(&dRS2, (size_t)P * M * G * 4));
    HIP_CHECK(hipMalloc(&dScr, 2 * slot_bytes)); HIP_CHECK(hipMalloc(&dFl, 1024 * 4)); HIP_CHECK(hipMalloc(&dSeq, 1024 * 4));
    HIP_CHECK(hipMemset(dFl, 0, 1024 * 4)); HIP_CHECK(hipMemset(dSeq, 0, 1024 * 4)); HIP_CHECK(hipMemset(dScr, 0, 2 * slot_bytes));
    HIP_CHECK(hipMemcpy(dY, Yh.data(), Yh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dY2, Y2h.data(), Y2h.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dR, Rh.data(), Rh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, Wh.data(), Wh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dT, T.data(), T.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCS, CS.data(), CS.size() * 2, hipMemcpyHostToDevice));
    size_t worst = 0;
    for (int rep = 0; rep < 3; ++rep) {     // three launches: slot parity 1, 0, 1
      HIP_CHECK(hipMemset(dA, 0, (size_t)P * M * K)); HIP_CHECK(hipMemset(dHS, 0, (size_t)M * K * 2)); HIP_CHECK(hipMemset(dRO, 0, (size_t)M * K * 2));
      hipLaunchKernelGGL(pq_ar_add_rms_rot, dim3(M), dim3(PQ_AR_WAVES * 32), 0, 0,
                         (const uint4_t *)dY, (uint4_t *)dScr, (const uint4_t *)dScr, (int)(slot_bytes / 16),
                         dFl, dFl, dSeq, (const __bf16 *)dR, (const __bf16 *)dW, eps, dT, dCS,
                         (__bf16 *)dHS, (__bf16 *)dRO, dA, dASG, dRS, M, K, P, krot, 3, 0);
      hipLaunchKernelGGL(pq_add_rms_rot<true>, dim3(M, (G + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, P),
                         dim3(PQ_ROT_WAVES * 32), 0, 0, (const __bf16 *)dY2, (const __bf16 *)dR,
                         (const __bf16 *)dW, eps, dT, dCS, (__bf16 *)dHS2, (__bf16 *)dRO2, dA2, dASG2,
                         dRS2, M, K, krot, 1);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned short> hHS((size_t)M * K), hRO((size_t)M * K), hHS2((size_t)M * K), hRO2((size_t)M * K);
      std::vector<unsigned char> hA((size_t)P * M * K), hA2((size_t)P * M * K);
      std::vector<float> hASG((size_t)P * M * G), hRS((size_t)P * M * G), hASG2((size_t)P * M * G), hRS2((size_t)P * M * G);
      HIP_CHECK(hipMemcpy(hHS.data(), dHS, hHS.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hRO.data(), dRO, hRO.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hHS2.data(), dHS2, hHS2.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hRO2.data(), dRO2, hRO2.size() * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hA.data(), dA, hA.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hA2.data(), dA2, hA2.size(), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hASG.data(), dASG, hASG.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hASG2.data(), dASG2, hASG2.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hRS.data(), dRS, hRS.size() * 4, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hRS2.data(), dRS2, hRS2.size() * 4, hipMemcpyDeviceToHost));
      size_t dhs = 0, dro = 0, da = 0, dsc = 0;
      for (size_t i = 0; i < hHS.size(); ++i) { if (hHS[i] != hHS2[i]) ++dhs; if (hRO[i] != hRO2[i]) ++dro; }
      for (size_t i = 0; i < hA.size(); ++i) if (hA[i] != hA2[i]) ++da;
      for (size_t i = 0; i < hASG.size(); ++i) if (memcmp(&hASG[i], &hASG2[i], 4) || memcmp(&hRS[i], &hRS2[i], 4)) ++dsc;
      // ro must be exact; hs/codes/scales may flip a few times per million from the 16-wave vs
      // 8-wave variance reduction order (one ulp on rsqrt).
      const bool pass = dro == 0 && dhs <= std::max<size_t>(2, hHS.size() / 50000) && da <= 4 && dsc <= 4;
      worst = std::max(worst, dhs + dro + da + dsc);
      if (!pass) ++failures;
      printf("arfused-loop  M=%-4d rep=%d vs plain(2y): hs=%zu ro=%zu codes=%zu sc=%zu %s\n", M, rep, dhs, dro, da, dsc, pass ? "OK" : "FAIL");
    }
    hipFree(dY); hipFree(dY2); hipFree(dR); hipFree(dW); hipFree(dT); hipFree(dCS); hipFree(dHS); hipFree(dRO);
    hipFree(dHS2); hipFree(dRO2); hipFree(dA); hipFree(dA2); hipFree(dASG); hipFree(dRS); hipFree(dASG2); hipFree(dRS2);
    hipFree(dScr); hipFree(dFl); hipFree(dSeq);
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
