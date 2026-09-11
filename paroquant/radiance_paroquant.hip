// pybind entry for the ParoQuant int4 (g128, asymmetric, pairwise-rotated) x fp8 W4A8 stack on
// gfx1201.
//
// The kernels live in par_kernels.h, compiled unchanged by par_harness.hip -- same discipline as
// the AutoRound module: a kernel gated in a harness and then re-typed into the serving module is a
// kernel that has not been gated.
#include "par_kernels.h"

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <cstdlib>

#define PQ_DEC_MAX_N 32768
#define PQ_DEC_KS 4                     // scratch is still SIZED for the widest split
#define PQ_DEC_MAX_TM 8                 // covers M <= 128 (RADIANCE_PQ_DECODE_MAX_M gates the band)
#define PQ_DEC_DWN 8

static bool env_flag(const char *name, bool dflt) {
  const char *e = getenv(name);
  return (e && *e) ? atoi(e) != 0 : dflt;
}
static int env_int(const char *name, int dflt) {
  const char *e = getenv(name);
  return (e && *e) ? atoi(e) : dflt;
}
// Weight layout, decided once per process by the loader (radiance_paroquant.py WPERM) -- the
// kernels read the flag here and it MUST agree with the tensor handed in, or the weight is read
// as garbage. Fragment order is what lets the decode kernel take streaming (NT) loads.
static bool wperm() { static const bool v = env_flag("RADIANCE_PQ_WPERM", true); return v; }
static bool dec_nt() { static const bool v = env_flag("RADIANCE_PQ_DECODE_NT", true); return v; }
// A-tiled prefill GEMM knobs (harness-measured defaults; see paroquant/RESULTS.md).
static int at_lbk() { static const int v = env_int("RADIANCE_PQ_AT_LBK", 128); return v; }
static bool at_hoist() { static const bool v = env_flag("RADIANCE_PQ_AT_HOIST", true); return v; }
// Rotation prologue v2 (records in registers, no block sync); bit-exact vs v1, harness-gated.
static bool rot_v2() { static const bool v = env_flag("RADIANCE_PQ_ROT_V2", true); return v; }

static float *g_partial = nullptr;
static size_t g_partial_bytes = 0;
static int *g_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_partial = reinterpret_cast<float *>(ptr);
  g_partial_bytes = bytes;
  g_cnt = reinterpret_cast<int *>(cnt);
}

// Split-K policy copied from the AutoRound launcher; the shapes are the same family and the
// partial-buffer economics (linear in M) are unchanged by the PARO deltas.
static int split_k_for(int nblk, int M, int K) {
  if (const char *e = getenv("RADIANCE_PQ_DECODE_KS")) return atoi(e);
  if (M > 64) {
    // The 16-concurrent band, copied from the AutoRound launcher's cell-by-cell sweep
    // (2026-08-29, arsweep.hip): wide shapes never split; the nblk=40 shapes split by 2 until
    // the partial traffic catches up, then diverge by K depth at M=128.
    if (nblk >= 48) return 1;
    if (M <= 112) return 2;
    return (K >= 6144) ? PQ_DEC_KS : 1;
  }
  int ks = (nblk >= 128) ? 1 : (nblk >= 64) ? 2 : PQ_DEC_KS;
  if (M > 48 && ks > 1) ks >>= 1;
  return ks;
}

static int decode_max_m() {
  static const int v = [] {
    const char *e = getenv("RADIANCE_PQ_DECODE_MAX_M");
    return e ? atoi(e) : 64;
  }();
  return v;
}

// Fused rotate + channel-scale + per-group fp8 quantize + code-domain row-sums (mode 0), or
// prefill pass A: rotate + write bf16 XR + per-group scales only (mode 1).
// X [M, K] bf16; T [P, krot, K/2, 4] u16; CS [P, K] f16
// -> A [P, M, K] e4m3 (mode 0) / XR bf16 via the same pointer (mode 1),
//    ASG [P, M, K/128] f32, RS [P, M, K/128] f32 (mode 0 only; carries rowsum*asg)
static void launch_rotate_quant(uintptr_t x, uintptr_t t, uintptr_t cs, uintptr_t a,
                                uintptr_t asg, uintptr_t rs, int M, int K, int P, int krot,
                                int mode, uintptr_t stream, int i8 = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const dim3 grid(K / PQ_GROUP, (M + PQ_ROT_TCHUNK - 1) / PQ_ROT_TCHUNK, P);
  const dim3 block(PQ_ROT_WAVES * 32);
  auto st = reinterpret_cast<hipStream_t>(stream);
#define PQ_ROT(KERN)                                                                            \
  hipLaunchKernelGGL(KERN, grid, block, 0, st, reinterpret_cast<const __bf16 *>(x),             \
                     reinterpret_cast<const unsigned short *>(t),                               \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<unsigned char *>(a), \
                     reinterpret_cast<float *>(asg), reinterpret_cast<float *>(rs), M, K, krot)
  if (i8)        { if (mode == 1) PQ_ROT((pq_rotate_quant2<true, true>)); else PQ_ROT((pq_rotate_quant2<false, true>)); }
  else if (rot_v2()) { if (mode == 1) PQ_ROT(pq_rotate_quant2<true>); else PQ_ROT(pq_rotate_quant2<false>); }
  else          { if (mode == 1) PQ_ROT(pq_rotate_quant<true>);  else PQ_ROT(pq_rotate_quant<false>); }
#undef PQ_ROT
}

// Prefill pass C: per-token scale + encode + plain code row-sums.
// XR [P, M, K] bf16; ASG [P, M, K/128] f32 -> A [P, M, K] e4m3, AS [P, M] f32, RS [P, M, K/128]
// tiled=1 writes A in the fragment-tiled layout the A-tiled GEMM reads ([P, Mt*16, K], see
// pq_token_quant_tiled); the caller sizes A accordingly.
// Waves per row for the per-token kernels. The small-M cost is the serial rotation chain per wave
// (K=5120: 40 groups -> 5 chains at 8 waves, 2 at 32), so single-stream M wants 32; at M >= 40 the
// 1024-thread blocks lose to 16 (par_harness tokstream, 2026-09-08). RADIANCE_PQ_TOK_WAVES=8|16|32
// forces one value for A/Bs. Rule: M<=16 -> 32, M<=64 -> 16, else 8.
static int pq_tok_waves(int M, int K = 0) {
  static int forced = -2;
  if (forced == -2) {
    const char *e = getenv("RADIANCE_PQ_TOK_WAVES");
    forced = (e && *e) ? atoi(e) : -1;
    if (forced != 8 && forced != 16 && forced != 32) forced = -1;
  }
  if (forced > 0) return forced;
  return M <= 16 ? 32 : (M <= 64 ? 16 : (K >= 8192 ? 16 : 8));   // tokqt gate: K=8704 wants 16 at M>=600
}

static void launch_rotate_tokquant(uintptr_t x, uintptr_t t, uintptr_t cs, uintptr_t a,
                                   uintptr_t as, int M, int K, int P, int krot, uintptr_t stream,
                                   int tiled = 0, uintptr_t rs = 0, int i8 = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const size_t smem = (size_t)K * 2;             // the rotated bf16 row
  if (smem > 60u * 1024u) throw std::runtime_error("ParoQuant: rotate_tokquant needs K*2 B of LDS; K too large");
  const dim3 grid(M, P);
  const int W = pq_tok_waves(M, K);
#define PQ_TQ(W_, T_, R_)                                                                        \
  hipLaunchKernelGGL((i8 ? &pq_rotate_tokquant<W_, T_, false, R_, true> : &pq_rotate_tokquant<W_, T_, false, R_, false>), grid, dim3(W_ * 32), smem, \
                     reinterpret_cast<hipStream_t>(stream),                                      \
                     reinterpret_cast<const __bf16 *>(x), reinterpret_cast<const unsigned short *>(t), \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<unsigned char *>(a),  \
                     reinterpret_cast<float *>(as), M, K, krot, reinterpret_cast<float *>(rs))
  if (rs) {   // int4 PTOK prologue: codes + token scale + plain row-sums in one launch
    if (tiled) { if (W == 32) PQ_TQ(32, true, true); else if (W == 16) PQ_TQ(16, true, true); else PQ_TQ(8, true, true); }
    else       { if (W == 32) PQ_TQ(32, false, true); else if (W == 16) PQ_TQ(16, false, true); else PQ_TQ(8, false, true); }
  } else {
    if (tiled) { if (W == 32) PQ_TQ(32, true, false); else if (W == 16) PQ_TQ(16, true, false); else PQ_TQ(8, true, false); }
    else       { if (W == 32) PQ_TQ(32, false, false); else if (W == 16) PQ_TQ(16, false, false); else PQ_TQ(8, false, false); }
  }
#undef PQ_TQ
}

// Per-token stream producers (paroquant_mxfp4 consumer). Out A [P, M, K] codes, AS [P, M].
// Per-group fused producer (PG A-tiled band): A [P, M, K] row or [P, Mt*16, K] tiled, ASG/RS [P, M, K/128].
static void launch_rotate_groupquant(uintptr_t x, uintptr_t t, uintptr_t cs, uintptr_t a, uintptr_t asg,
                                     uintptr_t rs, int M, int K, int P, int krot, uintptr_t stream,
                                     int tiled = 0, int i8 = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const dim3 grid(M, P);
  const int W = pq_tok_waves(M, K);
#define PQ_GQ(W_, T_, I_)                                                                        \
  hipLaunchKernelGGL((pq_rotate_groupquant<W_, T_, I_>), grid, dim3(W_ * 32), 0,                \
                     reinterpret_cast<hipStream_t>(stream),                                      \
                     reinterpret_cast<const __bf16 *>(x), reinterpret_cast<const unsigned short *>(t), \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<unsigned char *>(a),  \
                     reinterpret_cast<float *>(asg), reinterpret_cast<float *>(rs), M, K, krot)
#define PQ_GQ_W(T_, I_) do { if (W == 32) PQ_GQ(32, T_, I_); else if (W == 16) PQ_GQ(16, T_, I_); else PQ_GQ(8, T_, I_); } while (0)
  if (i8) { if (tiled) PQ_GQ_W(true, true); else PQ_GQ_W(false, true); }
  else    { if (tiled) PQ_GQ_W(true, false); else PQ_GQ_W(false, false); }
#undef PQ_GQ_W
#undef PQ_GQ
}

static void launch_add_rms_rot_tok(uintptr_t y, uintptr_t res, uintptr_t w, double eps, uintptr_t t,
                                   uintptr_t cs, uintptr_t hs, uintptr_t ro, uintptr_t a,
                                   uintptr_t as, int M, int K, int P, int krot, uintptr_t stream,
                                   int tiled = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const size_t smem = (size_t)K * 2;
  if (smem > 60u * 1024u) throw std::runtime_error("ParoQuant: add_rms_rot_tok needs K*2 B of LDS; K too large");
  const int W = pq_tok_waves(M, K);
#define PQ_ART(W_, T_)                                                                           \
  hipLaunchKernelGGL((pq_add_rms_rot_tok<W_, T_>), dim3(M, P), dim3(W_ * 32), smem,              \
                     reinterpret_cast<hipStream_t>(stream),                                      \
                     reinterpret_cast<const __bf16 *>(y), reinterpret_cast<const __bf16 *>(res),  \
                     reinterpret_cast<const __bf16 *>(w), (float)eps,                             \
                     reinterpret_cast<const unsigned short *>(t), reinterpret_cast<const __half *>(cs), \
                     reinterpret_cast<__bf16 *>(hs), reinterpret_cast<__bf16 *>(ro),              \
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as), M, K, krot)
  if (tiled) { if (W == 32) PQ_ART(32, true); else if (W == 16) PQ_ART(16, true); else PQ_ART(8, true); }
  else       { if (W == 32) PQ_ART(32, false); else if (W == 16) PQ_ART(16, false); else PQ_ART(8, false); }
#undef PQ_ART
}

static void launch_ew_rot_tok(int mode, uintptr_t x, uintptr_t y, long ys, uintptr_t w, double eps,
                              uintptr_t t, uintptr_t cs, uintptr_t hs, uintptr_t a, uintptr_t as,
                              int M, int N, int krot, uintptr_t stream, int tiled = 0) {
  if (N % PQ_GROUP) throw std::runtime_error("ParoQuant: N must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  const size_t smem = (size_t)N * 2;
  if (smem > 60u * 1024u) throw std::runtime_error("ParoQuant: ew_rot_tok needs N*2 B of LDS; N too large");
  auto st = reinterpret_cast<hipStream_t>(stream);
  const int W = pq_tok_waves(M, N);
#define PQ_EWT(KERN)                                                                             \
  hipLaunchKernelGGL(KERN, dim3(M), dim3(W * 32), smem, st,                                      \
                     reinterpret_cast<const __bf16 *>(x), reinterpret_cast<const __bf16 *>(y),    \
                     ys, reinterpret_cast<const __bf16 *>(w), (float)eps,                         \
                     reinterpret_cast<const unsigned short *>(t),                                \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<__bf16 *>(hs),        \
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as), M, N, krot)
#define PQ_EWT_W(W_, T_)                                                                         \
  if (mode == 0) PQ_EWT((pq_ew_rot_tok<0, W_, T_>));                                             \
  else if (mode == 1) PQ_EWT((pq_ew_rot_tok<1, W_, T_>));                                        \
  else PQ_EWT((pq_ew_rot_tok<2, W_, T_>))
  if (tiled) { if (W == 32) { PQ_EWT_W(32, true); } else if (W == 16) { PQ_EWT_W(16, true); } else { PQ_EWT_W(8, true); } }
  else       { if (W == 32) { PQ_EWT_W(32, false); } else if (W == 16) { PQ_EWT_W(16, false); } else { PQ_EWT_W(8, false); } }
#undef PQ_EWT_W
#undef PQ_EWT
}

// Skinny bf16 GEMM (see pq_skinny_bf16). p = fp32 partial scratch [K/256][M][N], cnt = int[N/16]
// zero-initialised once (the kernel re-zeroes its own counter). N % 16 == 0, K % 256 == 0, M <= 64.
static void launch_skinny_bf16(uintptr_t x, uintptr_t w, uintptr_t c, uintptr_t p, uintptr_t cnt,
                               int M, int N, int K, uintptr_t stream) {
  if (N % 16 || K % PQ_SK_KCH || M < 1 || M > PQ_SK_MAXM) throw std::runtime_error("skinny_bf16: N%16, K%256, 1<=M<=64");
  hipLaunchKernelGGL(pq_skinny_bf16, dim3(N / 16, K / PQ_SK_KCH), dim3(256), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const __bf16 *>(x), reinterpret_cast<const __bf16 *>(w), reinterpret_cast<__bf16 *>(c),
                     reinterpret_cast<float *>(p), reinterpret_cast<int *>(cnt), M, N, K);
}

static void launch_token_quant(uintptr_t xr, uintptr_t asg, uintptr_t a, uintptr_t as,
                               uintptr_t rs, int M, int K, int P, uintptr_t stream,
                               int tiled, int i8 = 0) {
  auto st = reinterpret_cast<hipStream_t>(stream);
  if (tiled) {
    if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: tiled quant needs K % 128 == 0");
    const dim3 grid((M + 15) / 16, P, pq_tq_zsplit(M));
    if (i8) hipLaunchKernelGGL(pq_token_quant_tiled<true>, grid, dim3(PQ_ROT_WAVES * 32), 0, st,
                       reinterpret_cast<const __bf16 *>(xr), reinterpret_cast<const float *>(asg),
                       reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as),
                       reinterpret_cast<float *>(rs), M, K);
    else hipLaunchKernelGGL(pq_token_quant_tiled<false>, grid, dim3(PQ_ROT_WAVES * 32), 0, st,
                       reinterpret_cast<const __bf16 *>(xr), reinterpret_cast<const float *>(asg),
                       reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as),
                       reinterpret_cast<float *>(rs), M, K);
    return;
  }
  const dim3 grid((M + PQ_ROT_WAVES - 1) / PQ_ROT_WAVES, P);
  if (i8) hipLaunchKernelGGL(pq_token_quant<true>, grid, dim3(PQ_ROT_WAVES * 32), 0, st,
                     reinterpret_cast<const __bf16 *>(xr), reinterpret_cast<const float *>(asg),
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as),
                     reinterpret_cast<float *>(rs), M, K);
  else hipLaunchKernelGGL(pq_token_quant<false>, grid, dim3(PQ_ROT_WAVES * 32), 0, st,
                     reinterpret_cast<const __bf16 *>(xr), reinterpret_cast<const float *>(asg),
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(as),
                     reinterpret_cast<float *>(rs), M, K);
}

// Fused residual-add + Gemma RMSNorm (+ rotate + per-group quant when fused=1, the decode band).
// Y/RES [M, K] bf16, W [K] bf16; T/CS the consumer linear's rotation; out HS/RO [M, K] bf16,
// A [P, M, K], ASG/RS [P, M, K/128] (fused only).
static void launch_add_rms_rot(uintptr_t y, uintptr_t res, uintptr_t w, double eps, uintptr_t t,
                               uintptr_t cs, uintptr_t hs, uintptr_t ro, uintptr_t a,
                               uintptr_t asg, uintptr_t rs, int M, int K, int P, int krot,
                               int fused, uintptr_t stream, int i8 = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  auto st = reinterpret_cast<hipStream_t>(stream);
  const int G = K / PQ_GROUP;
  const int gpw = pq_rot_gpw(M);
#define PQ_ARR(KERN, GRID)                                                                       \
  hipLaunchKernelGGL(KERN, GRID, dim3(PQ_ROT_WAVES * 32), 0, st,                                 \
                     reinterpret_cast<const __bf16 *>(y), reinterpret_cast<const __bf16 *>(res),  \
                     reinterpret_cast<const __bf16 *>(w), (float)eps,                             \
                     reinterpret_cast<const unsigned short *>(t),                                \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<__bf16 *>(hs),        \
                     reinterpret_cast<__bf16 *>(ro), reinterpret_cast<unsigned char *>(a),        \
                     reinterpret_cast<float *>(asg), reinterpret_cast<float *>(rs), M, K, krot,   \
                     gpw)
  if (fused && i8) PQ_ARR((pq_add_rms_rot<true, true>), dim3(M, (G + PQ_ROT_WAVES * gpw - 1) / (PQ_ROT_WAVES * gpw), P));
  else if (fused) PQ_ARR(pq_add_rms_rot<true>, dim3(M, (G + PQ_ROT_WAVES * gpw - 1) / (PQ_ROT_WAVES * gpw), P));
  else PQ_ARR(pq_add_rms_rot<false>, dim3(M, 1, 1));
#undef PQ_ARR
}

// Elementwise / per-head producers fused with rotate + quant (decode band, fused=1) or the plain
// producer writing hs only (fused=0). mode 0: silu-mul (x = gate_up [M, 2N]); mode 1: attention
// gate (x [M, N] * sigmoid(y), y row stride ys); mode 2: GDN gated rmsnorm (x [M, N], z = y with
// stride ys, w [128], eps). Single partition. Out: hs [M, N] bf16 always; A [M, N], ASG/RS
// [M, N/128] when fused.
static void launch_ew_rot(int mode, uintptr_t x, uintptr_t y, long ys, uintptr_t w, double eps,
                          uintptr_t t, uintptr_t cs, uintptr_t hs, uintptr_t a, uintptr_t asg,
                          uintptr_t rs, int M, int N, int krot, int fused, uintptr_t stream, int i8 = 0) {
  if (N % PQ_GROUP) throw std::runtime_error("ParoQuant: N must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  auto st = reinterpret_cast<hipStream_t>(stream);
  const int G = N / PQ_GROUP;
  const int gpw = fused ? pq_rot_gpw(M) : 1;
  const dim3 grid(M, (G + PQ_ROT_WAVES * gpw - 1) / (PQ_ROT_WAVES * gpw), 1);
#define PQ_EW(KERN)                                                                              \
  hipLaunchKernelGGL(KERN, grid, dim3(PQ_ROT_WAVES * 32), 0, st,                                 \
                     reinterpret_cast<const __bf16 *>(x), reinterpret_cast<const __bf16 *>(y),    \
                     ys, reinterpret_cast<const __bf16 *>(w), (float)eps,                         \
                     reinterpret_cast<const unsigned short *>(t),                                \
                     reinterpret_cast<const __half *>(cs), reinterpret_cast<__bf16 *>(hs),        \
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(asg),         \
                     reinterpret_cast<float *>(rs), M, N, krot, gpw)
  if (fused && i8) {
    if (mode == 0) PQ_EW((pq_ew_rot<0, true, true>));
    else if (mode == 1) PQ_EW((pq_ew_rot<1, true, true>));
    else PQ_EW((pq_ew_rot<2, true, true>));
  } else if (fused) {
    if (mode == 0) PQ_EW((pq_ew_rot<0, true>));
    else if (mode == 1) PQ_EW((pq_ew_rot<1, true>));
    else PQ_EW((pq_ew_rot<2, true>));
  } else {
    if (mode == 0) PQ_EW((pq_ew_rot<0, false>));
    else if (mode == 1) PQ_EW((pq_ew_rot<1, false>));
    else PQ_EW((pq_ew_rot<2, false>));
  }
#undef PQ_EW
}

// Two-rank one-shot all-reduce fused with add + rmsnorm + rotate + quant (decode band). The
// caller owns the IPC scratch (2 slots of slot_bytes), the flags (>= M*chunks u32, IPC-shared)
// and the device seq counters (same count); grid (M, chunks) with chunks from gpw.
static void launch_ar_add_rms_rot(uintptr_t inp, uintptr_t peer_scratch, uintptr_t my_scratch,
                                  long slot_bytes, uintptr_t peer_flags, uintptr_t my_flags,
                                  uintptr_t seq, int nflags, uintptr_t res, uintptr_t w,
                                  double eps, uintptr_t t, uintptr_t cs, uintptr_t hs,
                                  uintptr_t ro, uintptr_t a, uintptr_t asg, uintptr_t rs,
                                  int M, int K, int P, int krot, int drain, int acq,
                                  uintptr_t stream, int i8 = 0) {
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant: K must be a multiple of 128");
  if (krot < 1 || krot > PQ_KROT_MAX) throw std::runtime_error("ParoQuant: krot out of range");
  if ((long)M * K * 2 > slot_bytes) throw std::runtime_error("ParoQuant AR: message exceeds the slot");
  // One workgroup per row (see the kernel): the grid must stay fully resident on both ranks.
  if (M > nflags || M > 128) throw std::runtime_error("ParoQuant AR: too many rows for the fused path");
  auto st = reinterpret_cast<hipStream_t>(stream);
  auto ar_kern = i8 ? &pq_ar_add_rms_rot<true> : &pq_ar_add_rms_rot<false>;
  hipLaunchKernelGGL(ar_kern, dim3(M), dim3(PQ_AR_WAVES * 32), 0, st,
                     reinterpret_cast<const uint4_t *>(inp), reinterpret_cast<uint4_t *>(peer_scratch),
                     reinterpret_cast<const uint4_t *>(my_scratch), (int)(slot_bytes / 16),
                     reinterpret_cast<unsigned int *>(peer_flags), reinterpret_cast<unsigned int *>(my_flags),
                     reinterpret_cast<unsigned int *>(seq), reinterpret_cast<const __bf16 *>(res),
                     reinterpret_cast<const __bf16 *>(w), (float)eps,
                     reinterpret_cast<const unsigned short *>(t), reinterpret_cast<const __half *>(cs),
                     reinterpret_cast<__bf16 *>(hs), reinterpret_cast<__bf16 *>(ro),
                     reinterpret_cast<unsigned char *>(a), reinterpret_cast<float *>(asg),
                     reinterpret_cast<float *>(rs), M, K, P, krot, drain, acq);
}

// A-tiled per-token prefill GEMM: AT [P, Mt*16, K] fragment-tiled e4m3, AS [P, M], RS [P, M, G]
// plain row-sums. Prefill-class M only -- the decode band reads row-major A.
static void launch_gemm_at(uintptr_t a, uintptr_t w, uintptr_t sz, uintptr_t as, uintptr_t rs,
                           uintptr_t c, int M, int N, int K, int pb1, int pb2,
                           uintptr_t stream, uintptr_t whi = 0, int i8 = 0, int pg = 0) {
  // pg != 0: per-GROUP activation scales (as = ASG [P, M, K/128], rs = rowsum*asg), no epilogue scale.
  // whi != 0: int5 weights (fifth-bit plane WH, fragment order); needs WPERM.
  // i8 != 0: int8 activations on the iu8 WMMA (I8 mode; A/AS/RS from the int8 producers).
  const auto *WH = reinterpret_cast<const unsigned char *>(whi);
  if ((whi || i8) && !wperm()) throw std::runtime_error("ParoQuant int5 / I8 needs RADIANCE_PQ_WPERM=1");
  if (K % PQ_GROUP) throw std::runtime_error("ParoQuant kernel: K must be a multiple of 128");
  const auto *A = reinterpret_cast<const unsigned char *>(a);
  const auto *W = reinterpret_cast<const unsigned int *>(w);
  const auto *SZ = reinterpret_cast<const __half *>(sz);
  const auto *AS = reinterpret_cast<const float *>(as);
  const auto *RS = reinterpret_cast<const float *>(rs);
  auto *C = reinterpret_cast<__bf16 *>(c);
  auto st = reinterpret_cast<hipStream_t>(stream);
  constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
  const dim3 grid((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
#define PQ_AT(WP_, LBK_, H_, B_, I_, P_)                                                         \
  hipLaunchKernelGGL((pq_int4_fp8_gemm_atiled<TN, WP_, LBK_, H_, 0, 0, B_, I_, P_>), grid, block, 0, st, \
                     A, W, SZ, AS, RS, C, M, N, K, pb1, pb2, nullptr, nullptr, WH)
#define PQ_AT_LBK(WP_, B_, I_, P_)                                                               \
  do {                                                                                           \
    if (at_lbk() == 64) { if (at_hoist()) PQ_AT(WP_, 64, true, B_, I_, P_); else PQ_AT(WP_, 64, false, B_, I_, P_); }    \
    else                { if (at_hoist()) PQ_AT(WP_, 128, true, B_, I_, P_); else PQ_AT(WP_, 128, false, B_, I_, P_); }  \
  } while (0)
  if (pg) {
    if (!wperm()) throw std::runtime_error("ParoQuant PG needs RADIANCE_PQ_WPERM=1");
    if (i8) { if (whi) PQ_AT_LBK(true, 5, true, true); else PQ_AT_LBK(true, 4, true, true); }
    else    { if (whi) PQ_AT_LBK(true, 5, false, true); else PQ_AT_LBK(true, 4, false, true); }
  }
  else if (i8) { if (whi) PQ_AT_LBK(true, 5, true, false); else PQ_AT_LBK(true, 4, true, false); }
  else if (whi) PQ_AT_LBK(true, 5, false, false);
  else if (wperm()) PQ_AT_LBK(true, 4, false, false); else PQ_AT_LBK(false, 4, false, false);
#undef PQ_AT_LBK
#undef PQ_AT
}

// A [P, M, K] e4m3   W [N, K/8] u32   SZ [K/128, N, 2] f16 {scale, zscale}
// ASG/RS [P, M, K/128] f32   C [M, N] bf16   pb1/pb2 partition boundary columns (INT_MAX unused)
static void launch_gemm(uintptr_t a, uintptr_t w, uintptr_t sz, uintptr_t asg, uintptr_t rs,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, int ptok,
                        uintptr_t stream, uintptr_t whi = 0, int i8 = 0) {
  // i8 != 0: int8 activations on the iu8 WMMA (I8 mode), fragment-order layout only.
  // whi != 0: int5 weights (fifth-bit plane WH); the 5-bit kernels are instantiated for the
  // fragment-order layout only (WPERM), which the loader asserts for int5.
  const auto *WH = reinterpret_cast<const unsigned char *>(whi);
  if ((whi || i8) && !wperm()) throw std::runtime_error("ParoQuant int5 / I8 needs RADIANCE_PQ_WPERM=1");
  if (K % PQ_GROUP)
    throw std::runtime_error("ParoQuant kernel: K must be a multiple of the 128 group size");

  const auto *A = reinterpret_cast<const unsigned char *>(a);
  const auto *W = reinterpret_cast<const unsigned int *>(w);
  const auto *SZ = reinterpret_cast<const __half *>(sz);
  const auto *ASG = reinterpret_cast<const float *>(asg);
  const auto *RS = reinterpret_cast<const float *>(rs);
  auto *C = reinterpret_cast<__bf16 *>(c);
  auto st = reinterpret_cast<hipStream_t>(stream);

  constexpr int BND = PQ_DEC_DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  const int ks = split_k_for(nblk, M, K);
  const bool have_scratch =
      g_partial && g_cnt && (size_t)ks * M * N * sizeof(float) <= g_partial_bytes;
  // ptok tensors (per-token As, plain row-sums) only fit the prefill kernel's PTOK template;
  // the decode band expects per-group ASG/RS, so a ptok call always takes the prefill path.
  if (!ptok && M > 0 && M <= decode_max_m() && M <= DEC_MTILE * PQ_DEC_MAX_TM &&
      N <= PQ_DEC_MAX_N && (ks == 1 || have_scratch)) {
    const dim3 grid(nblk, 1, ks), block(PQ_DEC_DWN * 32);
    const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;    // smallest tile that covers M
#define PQ_DEC(KS_, TM_, WP_, NT_, B_, I_)                                                      \
    hipLaunchKernelGGL((pq_int4_fp8_gemm_decode<PQ_DEC_DWN, KS_, TM_, true, 0, WP_, NT_, B_, I_>), grid, \
                       block, 0, st, A, W, SZ, ASG, RS, g_partial, g_cnt, C, M, N, K, pb1, pb2, WH)
#define PQ_DEC_TM(KS_, WP_, NT_, B_, I_)                                                        \
    do { if (tm == 1) PQ_DEC(KS_, 1, WP_, NT_, B_, I_); else if (tm == 2) PQ_DEC(KS_, 2, WP_, NT_, B_, I_);      \
         else if (tm == 3) PQ_DEC(KS_, 3, WP_, NT_, B_, I_); else if (tm == 4) PQ_DEC(KS_, 4, WP_, NT_, B_, I_); \
         else if (tm == 5) PQ_DEC(KS_, 5, WP_, NT_, B_, I_); else if (tm == 6) PQ_DEC(KS_, 6, WP_, NT_, B_, I_); \
         else if (tm == 7) PQ_DEC(KS_, 7, WP_, NT_, B_, I_); else PQ_DEC(KS_, 8, WP_, NT_, B_, I_); } while (0)
#define PQ_DEC_KS(WP_, NT_, B_, I_)                                                             \
    do { if (ks == 1) PQ_DEC_TM(1, WP_, NT_, B_, I_); else if (ks == 2) PQ_DEC_TM(2, WP_, NT_, B_, I_);  \
         else PQ_DEC_TM(4, WP_, NT_, B_, I_); } while (0)
    // Streaming loads pay only on the fragment-order layout (see pq_stage_w); the row layout
    // ignores the NT flag. I8 (int8 activations): WPERM only, NT per dec_nt().
    if (i8) { if (whi) { if (dec_nt()) PQ_DEC_KS(true, true, 5, true); else PQ_DEC_KS(true, false, 5, true); }
              else     { if (dec_nt()) PQ_DEC_KS(true, true, 4, true); else PQ_DEC_KS(true, false, 4, true); } }
    else if (whi) { if (dec_nt()) PQ_DEC_KS(true, true, 5, false); else PQ_DEC_KS(true, false, 5, false); }
    else if (wperm()) { if (dec_nt()) PQ_DEC_KS(true, true, 4, false); else PQ_DEC_KS(true, false, 4, false); }
    else PQ_DEC_KS(false, false, 4, false);
#undef PQ_DEC_KS
#undef PQ_DEC_TM
#undef PQ_DEC
    return;
    // If the scratch was never registered the guard above fails and we fall through to prefill,
    // which is correct at any M -- never produce nothing.
  }

  constexpr int TN = 2, BNF_T = AR_WN * TN * 16;
  const dim3 grid((N + BNF_T - 1) / BNF_T, (M + AR_BMF - 1) / AR_BMF), block(AR_NTHREADS);
#define PQ_PRE(PT_, WP_, B_, I_)                                                                \
  hipLaunchKernelGGL((pq_int4_fp8_gemm_prefill<TN, true, 0, PT_, WP_, B_, I_>), grid, block, 0, st, \
                     A, W, SZ, ASG, RS, C, M, N, K, pb1, pb2, WH)
  if (i8)   { if (whi) { if (ptok) PQ_PRE(true, true, 5, true); else PQ_PRE(false, true, 5, true); }
              else     { if (ptok) PQ_PRE(true, true, 4, true); else PQ_PRE(false, true, 4, true); } }
  else if (whi)  { if (ptok) PQ_PRE(true, true, 5, false); else PQ_PRE(false, true, 5, false); }
  else if (ptok) { if (wperm()) PQ_PRE(true, true, 4, false); else PQ_PRE(true, false, 4, false); }
  else      { if (wperm()) PQ_PRE(false, true, 4, false); else PQ_PRE(false, false, 4, false); }
#undef PQ_PRE
}

namespace py = pybind11;
PYBIND11_MODULE(radiance_paroquant_kernel, m) {
  m.def("launch_rotate_tokquant", &launch_rotate_tokquant,
        "x, T, cs, a, as, M, K, P, krot, stream, tiled (tiled=1 writes the fragment-tiled A the A-tiled GEMM reads)", py::arg("x"), py::arg("T"), py::arg("cs"), py::arg("a"), py::arg("as"), py::arg("M"), py::arg("K"), py::arg("P"), py::arg("krot"), py::arg("stream"), py::arg("tiled") = 0, py::arg("rs") = 0, py::arg("i8") = 0);
  m.def("launch_rotate_groupquant", &launch_rotate_groupquant,
        "x, T, cs, a, asg, rs, M, K, P, krot, stream, tiled, i8 (per-group fused producer for the PG A-tiled band)",
        py::arg("x"), py::arg("T"), py::arg("cs"), py::arg("a"), py::arg("asg"), py::arg("rs"), py::arg("M"), py::arg("K"), py::arg("P"), py::arg("krot"), py::arg("stream"), py::arg("tiled") = 0, py::arg("i8") = 0);
  m.def("launch_add_rms_rot_tok", &launch_add_rms_rot_tok,
        "y, res, w, eps, T, cs, hs, ro, a, as, M, K, P, krot, stream, tiled (tiled=1 writes the fragment-tiled A the A-tiled GEMM reads)", py::arg("y"), py::arg("res"), py::arg("w"), py::arg("eps"), py::arg("T"), py::arg("cs"), py::arg("hs"), py::arg("ro"), py::arg("a"), py::arg("as"), py::arg("M"), py::arg("K"), py::arg("P"), py::arg("krot"), py::arg("stream"), py::arg("tiled") = 0);
  m.def("launch_ew_rot_tok", &launch_ew_rot_tok,
        "mode, x, y, ys, w, eps, T, cs, hs, a, as, M, N, krot, stream, tiled (tiled=1 writes the fragment-tiled A the A-tiled GEMM reads)", py::arg("mode"), py::arg("x"), py::arg("y"), py::arg("ys"), py::arg("w"), py::arg("eps"), py::arg("T"), py::arg("cs"), py::arg("hs"), py::arg("a"), py::arg("as"), py::arg("M"), py::arg("N"), py::arg("krot"), py::arg("stream"), py::arg("tiled") = 0);
  m.def("launch_skinny_bf16", &launch_skinny_bf16, "skinny bf16 GEMM C=X.W^T: x, w, c, partials, cnt, M, N, K, stream");
  m.def("launch_token_quant", &launch_token_quant,
        "prefill pass C: per-token scale + e4m3 encode + code row-sums (gfx1201)");
  m.def("launch_rotate_quant", &launch_rotate_quant,
        "fused pairwise-rotation + channel-scale + per-group fp8 quant + row-sums (gfx1201)");
  m.def("launch_gemm", &launch_gemm,
        "ParoQuant int4 g128-asym x fp8 W4A8 GEMM with partition select (gfx1201)");
  m.def("launch_ar_add_rms_rot", &launch_ar_add_rms_rot,
        "two-rank one-shot all-reduce + residual add + rmsnorm + rotate + quant (decode band)");
  m.def("launch_ew_rot", &launch_ew_rot,
        "silu-mul / attention-gate / gdn-gated-norm producer (+ rotate + quant in the decode band)");
  m.def("launch_add_rms_rot", &launch_add_rms_rot,
        "fused residual-add + Gemma rmsnorm (+ rotate + per-group fp8 quant in the decode band)");
  m.def("launch_gemm_at", &launch_gemm_at,
        "ParoQuant per-token GEMM on a fragment-tiled activation (prefill band)");
  m.def("set_decode_scratch", &set_decode_scratch,
        "register the split-K partial buffer (ptr, bytes) and the zeroed block counter");
}
