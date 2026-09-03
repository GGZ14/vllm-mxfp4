#pragma once
// ParoQuant int4 (group-128, ASYMMETRIC) x fp8-e4m3 activation GEMM for gfx1201, plus the fused
// rotate+scale+quantize prologue that feeds it. Forked from ar_kernels.h (AutoRound, symmetric);
// the fork points are marked PARO. Everything not marked is the AutoRound kernel unchanged, and
// the design notes there (clamp-never-predicate, 8-byte staging, hoisted scale, split-K policy)
// all still apply.
//
// FORMAT. ParoQuant exports AWQ buffers per projection: qweight [K, N/8] int32 AWQ-reordered,
// qzeros [K/128, N/8] likewise, scales [K/128, N] fp16 -- plus the rotation: pairs [krot, K]
// int16 (indices local to the 128 group), theta [krot, K/2] fp16, channel_scales [1, K] fp16
// (stored pre-inverted: multiply). The loader repacks to what these kernels read:
//
//   W  [N, K/8]  u32, eight 4-bit codes per word along K, low nibble = lowest k (as AutoRound)
//   SZ [K/128, N, 2] f16 interleaved: sz.x = scale s, sz.y = s * (zp - 8)   ("zscale")
//   T  [P, krot, K/2, 4] u16 rotation records: {i | j<<8, cos(theta) f16, sin(theta) f16, 0}
//   CS [P, K] f16 channel scales
//
// and the prologue produces, per output-partition p of a merged linear:
//
//   A   [P, M, K]      e4m3 codes of the rotated+scaled activations
//   ASG [P, M, K/128]  f32 per-(token, group) activation scale (amax/448 of the rotated group)
//   RS  [P, M, K/128]  f32 per-(token, group) sum of the CODE VALUES (e4m3-decoded)
//
// WHY THE ZERO POINT COSTS ALMOST NOTHING. The AutoRound kernel folds its constant zero of 8
// into the unpack LUT. ParoQuant's zero is per (group, n), so it cannot live in the LUT -- but it
// can stay OUT of the matmul anyway. Keep the (c - 8) LUT and write the true value as
// (c - zp) = (c - 8) - d with d = zp - 8. Then over one group g:
//     sum_k a_k * (c_k - zp) * s  =  s * WMMA(a, c-8)  -  s*d * sum_k a_k
// The row-sum term sum_k a_k is per (token, group) and the prologue computes it for free while it
// quantizes. Both a_k*c_k and d*a_k are EXACT in fp32 (4+4 significand bits), so the rearrangement
// only moves fp32 addition roundoff around. In-kernel cost: one extra f16 (zscale rides in the
// same 4-byte load as the scale), a per-slab LDS stage of asg/rs rows, and ~2 extra VALU per
// accumulator slot per slab -- the decode band has ~31 VALU slots per streamed byte and uses ~5.
//
// WHY PER-GROUP ACTIVATION SCALES. The per-token fp8 scale needs a token-wide amax, which a fused
// rotate+quantize kernel cannot know until every group is done -- that is a global sync. A
// per-group scale keeps the whole prologue one pass with zero cross-workgroup traffic, folds into
// the same per-slab rescale the group scale already pays for, and is strictly finer-grained fp8
// (the paper's kernel is W4A16; per-group A8 is the closest W4A8 gets to it).
//
// WHY PARTITION SELECT. ParoQuant rotations are per PROJECTION, so a merged linear (QKV, gate_up,
// the GDN in_proj merge) needs a differently-rotated A per output range. Splitting the GEMM into
// one launch per projection would triple QKV's launch count on a stack where the launch gap is
// ~20% of decode. Instead the A/ASG/RS tensors carry a leading partition axis and each n-block
// derives its partition from two boundary columns (pb1, pb2; INT_MAX when unused). Partition
// boundaries are multiples of 512 on every shape this model has, so no block straddles one.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

#define PQ_GROUP 128
#define PQ_KROT_MAX 8
#define DEC_MTILE 16
#define DEC_PAD 8

// (c - 8) unpack table, byte-exact from the AutoRound kernel (constants gated by lut.py there,
// re-gated by the 256-value round-trip test in par_harness).
#define AR_NEG_LO 0xCACCCED0u
#define AR_NEG_HI 0xB8C0C4C8u
#define AR_POS_LO 0x44403800u
#define AR_POS_HI 0x4E4C4A48u

__device__ __forceinline__ unsigned int ar_lut4(unsigned int c4) {
  const unsigned int sel = c4 & 0x07070707u;
  const unsigned int mask =
      __builtin_amdgcn_perm(0u, 0x0000FF00u, (c4 & 0x08080808u) >> 3);
  const unsigned int neg = __builtin_amdgcn_perm(AR_NEG_HI, AR_NEG_LO, sel);
  const unsigned int pos = __builtin_amdgcn_perm(AR_POS_HI, AR_POS_LO, sel);
  return (pos & mask) | (neg & ~mask);
}

__device__ __forceinline__ uint2_t ar_unpack8(unsigned int wv) {
  const unsigned int be = ar_lut4(wv & 0x0F0F0F0Fu);          // k = 0,2,4,6
  const unsigned int bo = ar_lut4((wv >> 4) & 0x0F0F0F0Fu);   // k = 1,3,5,7
  return uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                 __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}

// ---------------------------------------------------------------- weight staging (both layouts)
//
// Two weight layouts, ONE sW tile. Row layout (WPERM=false) is the loader's [N, K/8] u32, eight
// codes along K per word. Fragment order (WPERM=true) is the MXFP4 kernel's layout, ported
// verbatim: one 32-lane uint32 slot per (n-tile, k-step); slot lane l of tile (nt, ks) holds row
// nt*16 + (l & 15), k = ks*16 + (l >> 4)*8 .. +7. A wave's read is then 128 contiguous bytes
// instead of sixteen rows K/2 bytes apart, which is what let the MXFP4 decode kernel take
// nontemporal loads (NT: `global_load ... th:TH_LOAD_NT`, the slab is read exactly once per step
// so keeping it out of L2/MALL protects A and the split-K partials). NT on the ROW layout is
// 2-3.6x SLOWER on MXFP4 (a row's slab is half a line; the other half is only wanted by the next
// slab and NT re-fetches it) -- the launcher never asks for it.
//
// Clamp, never predicate: N % 16 == 0 under WPERM (loader-asserted) so clamping to the last
// WSLOTS-aligned row keeps the vector read aligned and every gc + q in bounds.
template <bool NT, typename T>
static __device__ __forceinline__ T pq_ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

template <int ROWS, int LBK, int STR, int NTHREADS, bool WPERM, bool NT, int ABLATE = 0,
          int WSLOT_OVR = 0>
__device__ __forceinline__ void pq_stage_w(unsigned char *__restrict__ sW,
                                           const unsigned int *__restrict__ W, int n0, int N,
                                           int K, int k0, int tid) {
  if constexpr (WPERM) {
    constexpr int KSTEPS_T = LBK / 16, NTILES_T = ROWS / 16;
    constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
    constexpr int WSLOTS = WSLOT_OVR ? WSLOT_OVR : ((TOT_SLOTS >= NTHREADS * 4) ? 4 : 2);
    static_assert(TOT_SLOTS % (NTHREADS * WSLOTS) == 0, "staging must be branch-free");
    const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
    for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
      const int sl = off + tid * WSLOTS;              // first of the group; WSLOTS-aligned
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
      const int gn = n0 + r;
      const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
      const int lanec = (lane_ & 16) | (gc & 15);
      const unsigned int *src = &W[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
      unsigned int wq[WSLOTS];
      if constexpr (WSLOTS == 4) {
        const uint4_t v = pq_ld_w<NT>((const uint4_t *)src);
        wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
      } else {
        const uint2_t v = pq_ld_w<NT>((const uint2_t *)src);
        wq[0] = v[0]; wq[1] = v[1];
      }
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        if constexpr (ABLATE & 1)
          *(uint2_t *)(&sW[(r + q) * STR + kloc]) = uint2_t{wq[q], wq[q]};
        else
          *(uint2_t *)(&sW[(r + q) * STR + kloc]) = ar_unpack8(wq[q]);
      }
    }
  } else {
    constexpr int WPT = LBK / 16;                       // uint2 (16 codes) per row per slab
    static_assert((ROWS * WPT) % NTHREADS == 0, "staging must be branch-free");
    const int kw = K / 8;
#pragma unroll
    for (int off = 0; off < ROWS * WPT; off += NTHREADS) {
      const int idx = off + tid;
      const int r = idx / WPT, c = idx % WPT, gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;
      const uint2_t wv = pq_ld_w<NT>((const uint2_t *)(W + (size_t)gc * kw + k0 / 8 + c * 2));
      if constexpr (ABLATE & 1) {
        *(uint2_t *)(&sW[r * STR + c * 16]) = uint2_t{wv[0], wv[0]};
        *(uint2_t *)(&sW[r * STR + c * 16 + 8]) = uint2_t{wv[1], wv[1]};
      } else {
        *(uint2_t *)(&sW[r * STR + c * 16]) = ar_unpack8(wv[0]);
        *(uint2_t *)(&sW[r * STR + c * 16 + 8]) = ar_unpack8(wv[1]);
      }
    }
  }
}

// ---------------------------------------------------------------- e4m3 encode/decode (OCP)
//
// Software, not the v_cvt_pk_fp8 path, for two reasons: the builtin's availability/semantics on
// gfx1201 under this image's clang is unverified, and the row-sum MUST be the sum of the values
// the codes actually decode to -- deriving it from the same decode function makes that true by
// construction. Both functions are gated exhaustively (256 codes round-trip + RNE boundary cases)
// in par_harness before anything downstream is trusted. The prologue is ~15 VALU per element
// against a GEMM that streams 4.25 bits/weight; this is not the place to spend cleverness.
__host__ __device__ __forceinline__ float pq_e4m3_decode(unsigned char b) {
  const float s = (b >> 7) ? -1.f : 1.f;
  const int E = (b >> 3) & 0xF, m = b & 7;
  // subnormal step 2^-9; normal (1 + m/8) * 2^(E-7)
  return E == 0 ? s * (float)m * 0.001953125f
                : s * (1.f + (float)m * 0.125f) * exp2f((float)(E - 7));
}

// Round-to-nearest-even, saturating to +-448 (no NaN/inf is ever produced; input is finite by
// construction -- amax/448 scaling puts |v| <= 448 up to roundoff).
__host__ __device__ __forceinline__ unsigned char pq_e4m3_encode(float v) {
  const unsigned char sign = v < 0.f ? 0x80 : 0x00;
  float a = fabsf(v);
  if (!(a > 0.f)) return 0;                       // covers +-0 and any stray NaN
  if (a >= 448.f) return sign | 0x7E;
  if (a < 0.015625f) {                            // below min normal 2^-6: subnormal, step 2^-9
    const float q = rintf(a * 512.f);             // RNE in the default rounding mode
    return sign | (unsigned char)q;               // q in 0..8; 8 carries into 0x08 = 2^-6 exactly
  }
  int ex;
  const float mfrac = frexpf(a, &ex);             // a = mfrac * 2^ex, mfrac in [0.5, 1)
  float q = rintf(mfrac * 16.f);                  // (mfrac*2) * 8, mantissa steps of 1/8
  int E = ex - 1 + 7;                             // biased
  if (q >= 16.f) { q = 8.f; ++E; }                // mantissa carry
  if (E > 15 || (E == 15 && q > 14.f)) return sign | 0x7E;   // saturate (0xF6 is 448; 0xF7 NaN)
  return sign | (unsigned char)((E << 3) | ((int)q - 8));
}

// ------------------------------------------------------------------ fused rotate+quant prologue
//
// Grid (K/128, ceil(M / PQ_ROT_TPB), P), block PQ_ROT_WAVES*32. One workgroup owns one
// 128-channel group of one partition: it loads that group's rotation records and channel scales
// into LDS ONCE, then its waves sweep tokens -- one wave per token, four channels per lane, the
// group staged in LDS so the arbitrary pair indices are just ds reads. Rotation layers within a
// wave need only an lgkmcnt wait between them: the 64 pairs of a layer partition the 128 channels
// (dummy pairs are real pairs with theta 0), so no lane's write aliases another's read inside a
// layer, and one wave serves one token so there is no cross-wave hazard at all.
//
// The token sweep, not one-token-per-workgroup, is what keeps the record table off the hot path:
// records are krot*K/2*8 bytes per partition (~0.5 MB on down_proj) and a per-token read of that
// would be 20% of the decode weight stream; per (group-chunk x token-chunk) it is read
// ceil(M/PQ_ROT_TPB) times total, which at decode M is exactly once.
#define PQ_ROT_WAVES 8
#define PQ_ROT_TPB PQ_ROT_WAVES        // tokens per block pass; each wave loops m += PQ_ROT_TPB
#ifndef PQ_ROT_TCHUNK
#define PQ_ROT_TCHUNK 256              // tokens per workgroup before a fresh block re-reads LDS
#endif

// ROTOUT=false: fused rotate+quantize (decode band) -- writes e4m3 codes A, per-group scales
// ASG, and RS = rowsum*asg.
// ROTOUT=true: prefill pass A -- writes the ROTATED bf16 values to XR (aliased through the A
// pointer) and per-group scales to ASG; encode and row-sums move to pq_token_quant, which owns
// the per-TOKEN scale (a token-wide amax cannot exist in this kernel without a global sync).
template <bool ROTOUT = false>
__global__ __launch_bounds__(PQ_ROT_WAVES * 32) void pq_rotate_quant(
    const __bf16 *__restrict__ X,           // [M, K]
    const unsigned short *__restrict__ T,   // [P, krot, K/2, 4]
    const __half *__restrict__ CS,          // [P, K]
    unsigned char *__restrict__ A,          // [P, M, K] codes, or [P, M, K] bf16 XR if ROTOUT
    float *__restrict__ ASG,                // [P, M, K/128]
    float *__restrict__ RS,                 // [P, M, K/128] (unused when ROTOUT)
    int M, int K, int krot) {
  const int g = blockIdx.x, p = blockIdx.z;
  const int G = K / PQ_GROUP;
  const int m_lo = blockIdx.y * PQ_ROT_TCHUNK;
  const int m_hi = min(M, m_lo + PQ_ROT_TCHUNK);

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;

  // LDS: rotation records for this (p, g), channel scales, and one 128-float slab per wave.
  __shared__ unsigned short s_rec[PQ_KROT_MAX * 64 * 4];
  __shared__ float s_cs[PQ_GROUP];
  __shared__ float s_x[PQ_ROT_WAVES][PQ_GROUP];

  // Records: krot*64 entries of 4 u16, loaded as one u64 each by the first krot*64 threads
  // (block has 256; krot <= 8 gives at most 512 entries, so loop twice).
  for (int r = tid; r < krot * 64; r += PQ_ROT_WAVES * 32) {
    const unsigned long long v = *(const unsigned long long *)(
        T + (((size_t)p * krot + r / 64) * (K / 2) + (size_t)g * 64 + (r % 64)) * 4);
    *(unsigned long long *)(s_rec + (size_t)r * 4) = v;
  }
  for (int c = tid; c < PQ_GROUP; c += PQ_ROT_WAVES * 32)
    s_cs[c] = __half2float(CS[(size_t)p * K + (size_t)g * PQ_GROUP + c]);
  __syncthreads();

  const int c0 = lane * 4;                 // this lane's four channels within the group
  for (int m = m_lo + wave; m < m_hi; m += PQ_ROT_TPB) {
    // Load + channel-scale four channels, park them in this wave's LDS slab.
    const __bf16 *xr = X + (size_t)m * K + (size_t)g * PQ_GROUP + c0;
    float v0 = (float)xr[0] * s_cs[c0 + 0], v1 = (float)xr[1] * s_cs[c0 + 1];
    float v2 = (float)xr[2] * s_cs[c0 + 2], v3 = (float)xr[3] * s_cs[c0 + 3];
    s_x[wave][c0 + 0] = v0; s_x[wave][c0 + 1] = v1;
    s_x[wave][c0 + 2] = v2; s_x[wave][c0 + 3] = v3;
    __asm__ volatile("s_waitcnt lgkmcnt(0)");

    // Rotation layers. Two pairs per lane per layer (64 pairs / 32 lanes).
    for (int r = 0; r < krot; ++r) {
#pragma unroll
      for (int t2 = 0; t2 < 2; ++t2) {
        const unsigned short *rec = s_rec + ((size_t)r * 64 + lane + 32 * t2) * 4;
        const unsigned int ij = rec[0];
        const float c = __half2float(*(const __half *)(rec + 1));
        const float s = __half2float(*(const __half *)(rec + 2));
        const int i = ij & 0xFF, j = ij >> 8;
        const float xi = s_x[wave][i], xj = s_x[wave][j];
        s_x[wave][i] = fmaf(c, xi, s * xj);
        s_x[wave][j] = fmaf(c, xj, -s * xi);
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }

    // Per-group amax over the rotated slab (each lane re-reads its four slots -- they were
    // possibly rewritten by other lanes' pairs).
    v0 = s_x[wave][c0 + 0]; v1 = s_x[wave][c0 + 1];
    v2 = s_x[wave][c0 + 2]; v3 = s_x[wave][c0 + 3];
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    const float scale = fmaxf(amax * (1.f / 448.f), 1e-10f);
    const float inv = 1.f / scale;

    if constexpr (ROTOUT) {
      // Prefill pass A: park the rotated values (bf16) and this group's scale; encode happens
      // in pq_token_quant once the token-wide scale is known.
      __bf16 *xr = (__bf16 *)A + ((size_t)p * M + m) * K + (size_t)g * PQ_GROUP + c0;
      xr[0] = (__bf16)v0; xr[1] = (__bf16)v1; xr[2] = (__bf16)v2; xr[3] = (__bf16)v3;
      if (lane == 0) ASG[((size_t)p * M + m) * G + g] = scale;
      continue;
    }

    // Quantize, decode back for the code-domain row-sum, store four codes as one word.
    const unsigned char b0 = pq_e4m3_encode(v0 * inv), b1 = pq_e4m3_encode(v1 * inv);
    const unsigned char b2 = pq_e4m3_encode(v2 * inv), b3 = pq_e4m3_encode(v3 * inv);
    float rs = pq_e4m3_decode(b0) + pq_e4m3_decode(b1) + pq_e4m3_decode(b2) + pq_e4m3_decode(b3);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
      rs += __shfl_xor(rs, off, 32);
    *(unsigned int *)(A + ((size_t)p * M + m) * K + (size_t)g * PQ_GROUP + c0) =
        (unsigned int)b0 | ((unsigned int)b1 << 8) | ((unsigned int)b2 << 16) |
        ((unsigned int)b3 << 24);
    if (lane == 0) {
      ASG[((size_t)p * M + m) * G + g] = scale;
      // RS carries rowsum * asg, so the GEMM's zero-point correction is zsc * RS with no asg
      // factor -- one FMA per element per GROUP instead of a 3-op chain per slab.
      RS[((size_t)p * M + m) * G + g] = rs * scale;
    }
  }
}

// v2 of the prologue, same contract. The records of this lane's two pairs per layer live in
// REGISTERS (16 x u64, loaded straight from global with the token's x and the channel scales in
// the same latency window), so there is no record LDS fill, no __syncthreads, and a rotation
// layer is 4 LDS reads + 4 writes instead of 10 reads + 4 writes. At decode this kernel is pure
// latency chain (40 x P blocks on 64 CUs), so the two removed round trips are the win; at
// prefill the LDS op count is the win. Arithmetic is identical (same fmaf on the same disjoint
// pairs) -> bit-exact against v1, gated in par_harness.
template <bool ROTOUT = false>
__global__ __launch_bounds__(PQ_ROT_WAVES * 32) void pq_rotate_quant2(
    const __bf16 *__restrict__ X, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, unsigned char *__restrict__ A, float *__restrict__ ASG,
    float *__restrict__ RS, int M, int K, int krot) {
  const int g = blockIdx.x, p = blockIdx.z;
  const int G = K / PQ_GROUP;
  const int m_lo = blockIdx.y * PQ_ROT_TCHUNK;
  const int m_hi = min(M, m_lo + PQ_ROT_TCHUNK);
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  __shared__ float s_x[PQ_ROT_WAVES][PQ_GROUP];
  const int c0 = lane * 4;

  // This lane's pairs: t = lane and lane + 32 of every layer. Layers past krot are clamped to
  // the last real one and never applied (uniform branch below) -- clamp, never predicate.
  const unsigned long long *__restrict__ Tb =
      (const unsigned long long *)T + ((size_t)p * krot) * (K / 2) + (size_t)g * 64;
  unsigned long long rec[PQ_KROT_MAX][2];
#pragma unroll
  for (int r = 0; r < PQ_KROT_MAX; ++r) {
    const int rc = r < krot ? r : krot - 1;
    rec[r][0] = Tb[(size_t)rc * (K / 2) + lane];
    rec[r][1] = Tb[(size_t)rc * (K / 2) + lane + 32];
  }
  const uint2_t csv = *(const uint2_t *)(CS + (size_t)p * K + (size_t)g * PQ_GROUP + c0);
  const float cs0 = __half2float(__ushort_as_half((unsigned short)(csv[0] & 0xFFFFu)));
  const float cs1 = __half2float(__ushort_as_half((unsigned short)(csv[0] >> 16)));
  const float cs2 = __half2float(__ushort_as_half((unsigned short)(csv[1] & 0xFFFFu)));
  const float cs3 = __half2float(__ushort_as_half((unsigned short)(csv[1] >> 16)));

  for (int m = m_lo + wave; m < m_hi; m += PQ_ROT_TPB) {
    const uint2_t xv = *(const uint2_t *)(X + (size_t)m * K + (size_t)g * PQ_GROUP + c0);
    float v0 = __uint_as_float(xv[0] << 16) * cs0, v1 = __uint_as_float(xv[0] & 0xFFFF0000u) * cs1;
    float v2 = __uint_as_float(xv[1] << 16) * cs2, v3 = __uint_as_float(xv[1] & 0xFFFF0000u) * cs3;
    s_x[wave][c0 + 0] = v0; s_x[wave][c0 + 1] = v1;
    s_x[wave][c0 + 2] = v2; s_x[wave][c0 + 3] = v3;
    __asm__ volatile("s_waitcnt lgkmcnt(0)");

#pragma unroll
    for (int r = 0; r < PQ_KROT_MAX; ++r) {
      if (r < krot) {
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2) {
          const unsigned long long rv = rec[r][t2];
          const unsigned int ij = (unsigned int)(rv & 0xFFFFu);
          const float c = __half2float(__ushort_as_half((unsigned short)((rv >> 16) & 0xFFFFu)));
          const float sn = __half2float(__ushort_as_half((unsigned short)((rv >> 32) & 0xFFFFu)));
          const int i = ij & 0xFF, j = ij >> 8;
          const float xi = s_x[wave][i], xj = s_x[wave][j];
          s_x[wave][i] = fmaf(c, xi, sn * xj);
          s_x[wave][j] = fmaf(c, xj, -sn * xi);
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)");
      }
    }

    v0 = s_x[wave][c0 + 0]; v1 = s_x[wave][c0 + 1];
    v2 = s_x[wave][c0 + 2]; v3 = s_x[wave][c0 + 3];
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
      amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    const float scale = fmaxf(amax * (1.f / 448.f), 1e-10f);
    const float inv = 1.f / scale;

    if constexpr (ROTOUT) {
      __bf16 *xr = (__bf16 *)A + ((size_t)p * M + m) * K + (size_t)g * PQ_GROUP + c0;
      xr[0] = (__bf16)v0; xr[1] = (__bf16)v1; xr[2] = (__bf16)v2; xr[3] = (__bf16)v3;
      if (lane == 0) ASG[((size_t)p * M + m) * G + g] = scale;
      continue;
    }

    const unsigned char b0 = pq_e4m3_encode(v0 * inv), b1 = pq_e4m3_encode(v1 * inv);
    const unsigned char b2 = pq_e4m3_encode(v2 * inv), b3 = pq_e4m3_encode(v3 * inv);
    float rs = pq_e4m3_decode(b0) + pq_e4m3_decode(b1) + pq_e4m3_decode(b2) + pq_e4m3_decode(b3);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
      rs += __shfl_xor(rs, off, 32);
    *(unsigned int *)(A + ((size_t)p * M + m) * K + (size_t)g * PQ_GROUP + c0) =
        (unsigned int)b0 | ((unsigned int)b1 << 8) | ((unsigned int)b2 << 16) |
        ((unsigned int)b3 << 24);
    if (lane == 0) {
      ASG[((size_t)p * M + m) * G + g] = scale;
      RS[((size_t)p * M + m) * G + g] = rs * scale;
    }
  }
}

// ---------------------------------------- fused residual-add + Gemma RMSNorm + rotate + quant
//
// The decode-band producer for every norm-fed ParoQuant linear (input_layernorm -> qkv /
// in_proj_qkvz, post_attention_layernorm -> gate_up). Mirrors vLLM's ir.ops.fused_add_rms_norm
// exactly as inductor traces it: v = y + res in fp32, residual out = bf16(v), variance = mean(v^2)
// over the UNROUNDED v, hs = bf16((v * rsqrt(var + eps)) * (w + 1)); then the rotate_quant2 body
// on hs. One workgroup per (row, 8-group chunk, partition): every workgroup re-reads the row for
// the variance (20 KB per 10 KB of useful output -- nothing at decode M), only the (chunk 0,
// partition 0) workgroup writes the residual, only partition-0 workgroups write hs. ROT=false is
// the same kernel without the rotation (grid (M, 1, 1)): the prefill-size fallthrough, where the
// linear rotates on its own tiled path from hs.
//
// The block reduction order differs from inductor's, so rsqrt can differ by an ulp and flip a
// bf16 rounding of hs a few times per million elements -- same class as gdn_norm_quant; the
// harness gates fused == (ROT=false kernel + pq_rotate_quant2) bit-exact and both against the
// CPU reference at ppm.
template <bool ROT>
__global__ __launch_bounds__(PQ_ROT_WAVES * 32) void pq_add_rms_rot(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps,
    const unsigned short *__restrict__ T,     // [P, krot, K/2, 4]
    const __half *__restrict__ CS,            // [P, K]
    __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ ASG, float *__restrict__ RS,
    int M, int K, int krot, int gpw) {
  // gpw = groups per wave: 1 at single-stream M (latency-bound, one chain per wave), more at
  // batched M where the per-workgroup row re-read is the cost (M=64 P=3: 960 workgroups x 20 KB).
  const int m = blockIdx.x, chunk = blockIdx.y, p = blockIdx.z;
  const int G = K / PQ_GROUP;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int g0 = (chunk * gpw) * PQ_ROT_WAVES + wave;
  __shared__ float s_red[PQ_ROT_WAVES];
  __shared__ float s_x[PQ_ROT_WAVES][PQ_GROUP];
  const int c0 = lane * 4;

  // First group's records + channel scales, issued before the row pass so they land under it.
  unsigned long long rec[PQ_KROT_MAX][2];
  float cs0 = 1.f, cs1 = 1.f, cs2 = 1.f, cs3 = 1.f;
  auto load_group = [&](int g) {
    const int gc = g < G ? g : G - 1;                 // clamp, never predicate
    const unsigned long long *__restrict__ Tb =
        (const unsigned long long *)T + ((size_t)p * krot) * (K / 2) + (size_t)gc * 64;
#pragma unroll
    for (int r = 0; r < PQ_KROT_MAX; ++r) {
      const int rc = r < krot ? r : krot - 1;
      rec[r][0] = Tb[(size_t)rc * (K / 2) + lane];
      rec[r][1] = Tb[(size_t)rc * (K / 2) + lane + 32];
    }
    const uint2_t csv = *(const uint2_t *)(CS + (size_t)p * K + (size_t)gc * PQ_GROUP + c0);
    cs0 = __half2float(__ushort_as_half((unsigned short)(csv[0] & 0xFFFFu)));
    cs1 = __half2float(__ushort_as_half((unsigned short)(csv[0] >> 16)));
    cs2 = __half2float(__ushort_as_half((unsigned short)(csv[1] & 0xFFFFu)));
    cs3 = __half2float(__ushort_as_half((unsigned short)(csv[1] >> 16)));
  };
  if constexpr (ROT) load_group(g0);

  // Row pass: v = y + res (fp32), sum of squares, residual out.
  const uint4_t *__restrict__ y4 = (const uint4_t *)(Y + (size_t)m * K);
  const uint4_t *__restrict__ r4 = (const uint4_t *)(RES + (size_t)m * K);
  uint4_t *__restrict__ o4 = (uint4_t *)(RO + (size_t)m * K);
  const int KV = K >> 3;
  float ssq = 0.f;
  for (int i = tid; i < KV; i += PQ_ROT_WAVES * 32) {
    const uint4_t vy = y4[i], vr = r4[i];
    uint4_t vo;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float a0 = __uint_as_float(vy[h] << 16) + __uint_as_float(vr[h] << 16);
      const float a1 = __uint_as_float(vy[h] & 0xFFFF0000u) + __uint_as_float(vr[h] & 0xFFFF0000u);
      ssq = fmaf(a0, a0, ssq);
      ssq = fmaf(a1, a1, ssq);
      const __bf16 b0 = (__bf16)a0, b1 = (__bf16)a1;
      vo[h] = (unsigned int)__bfloat16_as_ushort(b0) | ((unsigned int)__bfloat16_as_ushort(b1) << 16);
    }
    if (chunk == 0 && p == 0) o4[i] = vo;
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) ssq += __shfl_xor(ssq, off, 32);
  if (lane == 0) s_red[wave] = ssq;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < PQ_ROT_WAVES; ++w) tot += s_red[w];
  const float inv = rsqrtf(tot / (float)K + eps);

  if constexpr (!ROT) {
    for (int i = tid; i < KV; i += PQ_ROT_WAVES * 32) {
      const uint4_t vy = y4[i], vr = r4[i];
      const uint4_t vw = ((const uint4_t *)Wn)[i];
      uint4_t vo;
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const float a0 = __uint_as_float(vy[h] << 16) + __uint_as_float(vr[h] << 16);
        const float a1 = __uint_as_float(vy[h] & 0xFFFF0000u) + __uint_as_float(vr[h] & 0xFFFF0000u);
        const float w0 = __uint_as_float(vw[h] << 16) + 1.f, w1 = __uint_as_float(vw[h] & 0xFFFF0000u) + 1.f;
        const __bf16 n0 = (__bf16)((a0 * inv) * w0), n1 = (__bf16)((a1 * inv) * w1);
        vo[h] = (unsigned int)__bfloat16_as_ushort(n0) | ((unsigned int)__bfloat16_as_ushort(n1) << 16);
      }
      ((uint4_t *)(HS + (size_t)m * K))[i] = vo;
    }
    return;
  }

  for (int gi = 0; gi < gpw; ++gi) {
  const int g = g0 + gi * PQ_ROT_WAVES;
  if (g >= G) return;                                   // wave-uniform
  if (gi) load_group(g);
  const int kb = g * PQ_GROUP + c0;
  const uint2_t vy = *(const uint2_t *)(Y + (size_t)m * K + kb);
  const uint2_t vr = *(const uint2_t *)(RES + (size_t)m * K + kb);
  const uint2_t vw = *(const uint2_t *)(Wn + kb);
  float nv[4];
  unsigned int hsw[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float a0 = __uint_as_float(vy[h] << 16) + __uint_as_float(vr[h] << 16);
    const float a1 = __uint_as_float(vy[h] & 0xFFFF0000u) + __uint_as_float(vr[h] & 0xFFFF0000u);
    const float w0 = __uint_as_float(vw[h] << 16) + 1.f, w1 = __uint_as_float(vw[h] & 0xFFFF0000u) + 1.f;
    const __bf16 n0 = (__bf16)((a0 * inv) * w0), n1 = (__bf16)((a1 * inv) * w1);
    nv[2 * h] = (float)n0; nv[2 * h + 1] = (float)n1;
    hsw[h] = (unsigned int)__bfloat16_as_ushort(n0) | ((unsigned int)__bfloat16_as_ushort(n1) << 16);
  }
  if (p == 0) *(uint2_t *)(HS + (size_t)m * K + kb) = uint2_t{hsw[0], hsw[1]};

  // rotate_quant2 body on the normalized values
  float v0 = nv[0] * cs0, v1 = nv[1] * cs1, v2 = nv[2] * cs2, v3 = nv[3] * cs3;
  s_x[wave][c0 + 0] = v0; s_x[wave][c0 + 1] = v1;
  s_x[wave][c0 + 2] = v2; s_x[wave][c0 + 3] = v3;
  __asm__ volatile("s_waitcnt lgkmcnt(0)");
#pragma unroll
  for (int r = 0; r < PQ_KROT_MAX; ++r) {
    if (r < krot) {
#pragma unroll
      for (int t2 = 0; t2 < 2; ++t2) {
        const unsigned long long rv = rec[r][t2];
        const unsigned int ij = (unsigned int)(rv & 0xFFFFu);
        const float c = __half2float(__ushort_as_half((unsigned short)((rv >> 16) & 0xFFFFu)));
        const float sn = __half2float(__ushort_as_half((unsigned short)((rv >> 32) & 0xFFFFu)));
        const int i = ij & 0xFF, j = ij >> 8;
        const float xi = s_x[wave][i], xj = s_x[wave][j];
        s_x[wave][i] = fmaf(c, xi, sn * xj);
        s_x[wave][j] = fmaf(c, xj, -sn * xi);
      }
      __asm__ volatile("s_waitcnt lgkmcnt(0)");
    }
  }
  v0 = s_x[wave][c0 + 0]; v1 = s_x[wave][c0 + 1];
  v2 = s_x[wave][c0 + 2]; v3 = s_x[wave][c0 + 3];
  float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  const float scale = fmaxf(amax * (1.f / 448.f), 1e-10f);
  const float qi = 1.f / scale;
  const unsigned char b0 = pq_e4m3_encode(v0 * qi), b1 = pq_e4m3_encode(v1 * qi);
  const unsigned char b2 = pq_e4m3_encode(v2 * qi), b3 = pq_e4m3_encode(v3 * qi);
  float rs = pq_e4m3_decode(b0) + pq_e4m3_decode(b1) + pq_e4m3_decode(b2) + pq_e4m3_decode(b3);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) rs += __shfl_xor(rs, off, 32);
  *(unsigned int *)(A + ((size_t)p * M + m) * K + kb) =
      (unsigned int)b0 | ((unsigned int)b1 << 8) | ((unsigned int)b2 << 16) | ((unsigned int)b3 << 24);
  if (lane == 0) {
    ASG[((size_t)p * M + m) * G + g] = scale;
    RS[((size_t)p * M + m) * G + g] = rs * scale;
  }
  __asm__ volatile("s_waitcnt lgkmcnt(0)");           // s_x reuse across groups
  }
}
static inline int pq_rot_gpw(int M) { return M <= 16 ? 1 : (M <= 32 ? 2 : 4); }

// Prefill pass C: one wave per (partition, token). Reduces the per-group scales to the token
// scale As = max_g ASG (they are amax/448, so their max IS the token amax/448), encodes the
// rotated row against it, and writes plain code-domain row-sums per group (the PTOK GEMM applies
// As once in its epilogue, so RS must NOT carry a scale here).
__global__ __launch_bounds__(PQ_ROT_WAVES * 32) void pq_token_quant(
    const __bf16 *__restrict__ XR,          // [P, M, K] rotated values (pass A)
    const float *__restrict__ ASG,          // [P, M, K/128] per-group scales (pass A)
    unsigned char *__restrict__ A,          // [P, M, K] e4m3 codes out
    float *__restrict__ AS,                 // [P, M] per-token scale out
    float *__restrict__ RS,                 // [P, M, K/128] code row-sums out
    int M, int K) {
  const int G = K / PQ_GROUP;
  const int p = blockIdx.y;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int m = blockIdx.x * PQ_ROT_WAVES + wave;
  if (m >= M) return;

  const float *asg_row = ASG + ((size_t)p * M + m) * G;
  float amx = 0.f;
  for (int g = lane; g < G; g += 32) amx = fmaxf(amx, asg_row[g]);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) amx = fmaxf(amx, __shfl_xor(amx, off, 32));
  const float scale = fmaxf(amx, 1e-10f);
  const float inv = 1.f / scale;
  if (lane == 0) AS[(size_t)p * M + m] = scale;

  const __bf16 *xr = XR + ((size_t)p * M + m) * K;
  unsigned char *ar = A + ((size_t)p * M + m) * K;
  for (int g = 0; g < G; ++g) {
    const int c0 = g * PQ_GROUP + lane * 4;
    const float v0 = (float)xr[c0] * inv, v1 = (float)xr[c0 + 1] * inv;
    const float v2 = (float)xr[c0 + 2] * inv, v3 = (float)xr[c0 + 3] * inv;
    const unsigned char b0 = pq_e4m3_encode(v0), b1 = pq_e4m3_encode(v1);
    const unsigned char b2 = pq_e4m3_encode(v2), b3 = pq_e4m3_encode(v3);
    *(unsigned int *)(ar + c0) = (unsigned int)b0 | ((unsigned int)b1 << 8) |
                                 ((unsigned int)b2 << 16) | ((unsigned int)b3 << 24);
    float rs = pq_e4m3_decode(b0) + pq_e4m3_decode(b1) + pq_e4m3_decode(b2) + pq_e4m3_decode(b3);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) rs += __shfl_xor(rs, off, 32);
    if (lane == 0) RS[((size_t)p * M + m) * G + g] = rs;
  }
}

// ------------------------------------------------------------------ decode path (small M)
//
// Structure is the AutoRound decode kernel. PARO deltas:
//   * SZ replaces S: one 4-byte load per lane per slab yields {scale, zscale}.
//   * s_asg / s_rs: per-slab LDS stage of the activation-side per-group scale and row-sum for the
//     rows in flight (DEC_MTILE*DTM <= 64 floats each), staged with sA under the same barrier.
//   * fold becomes  acc += asg[m] * (sc * t - zsc * rs[m]); the epilogue As multiply is gone
//     (the activation scale is per group now, so it HAS to fold per slab).
//   * A/ASG/RS are indexed through the block's partition (pb1/pb2 boundaries).
template <int DWN, int DKS, int DTM, bool IMAJOR = true, int ABLATE = 0, bool WPERM = false,
          bool NT = false>
__global__ __launch_bounds__(DWN * 32) void pq_int4_fp8_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ SZ, const float *__restrict__ ASG,
    const float *__restrict__ RS, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2) {
  constexpr int DBK = PQ_GROUP;            // one scale group per slab, by construction
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  constexpr int DROWS = DEC_MTILE * DTM;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ float s_asg[DROWS];
  __shared__ float s_rs[DROWS];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;

  // PARO: partition select. All of A/ASG/RS shift by the block's partition.
  const int prt = (n0 >= pb1 ? 1 : 0) + (n0 >= pb2 ? 1 : 0);
  const int G = K / PQ_GROUP;
  A += (size_t)prt * M * K;
  ASG += (size_t)prt * M * G;
  RS += (size_t)prt * M * G;

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  const int n_lane = n0 + wave * 16 + col;
  const int kw = K / 8;

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    // Hoisted exactly as in AutoRound; one 4-byte load now carries scale AND zscale.
    // Clamp, never predicate -- see ar_kernels.h for the six-round-trips story.
    float sc, zsc;
    if constexpr ((ABLATE & 2) != 0) {
      sc = 1.f; zsc = 0.f;
    } else {
      const __half2 szv =
          *(const __half2 *)(SZ + (((size_t)s * N + (n_lane < N ? n_lane : N - 1)) * 2));
      sc = __half2float(szv.x);
      zsc = __half2float(szv.y);
    }
    // PARO: stage this slab's activation group scale and row-sum for every row in flight.
    if (tid < DROWS) {
      const int rc = tid < M ? tid : M - 1;
      s_asg[tid] = ASG[(size_t)rc * G + s];
      s_rs[tid] = RS[(size_t)rc * G + s];
    }
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
    pq_stage_w<BND, DBK, DWSTR, DNTHREADS, WPERM, NT, ABLATE>(sW, W, n0, N, K, k0, tid);
    __syncthreads();

    if constexpr (IMAJOR) {
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        floatx8 t;
#pragma unroll
        for (int e = 0; e < 8; ++e) t[e] = 0.f;
#pragma unroll
        for (int step = 0; step < DBK / 16; ++step) {
          const int kk = step * 16 + kb8;
          int2_t af, wf;
          const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
          af[0] = *(const int *)pa; af[1] = *(const int *)(pa + 4);
          const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
          wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
          t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wf, t);
        }
        // Rows i*16+kb8 .. +7 are contiguous, so asg/rsa come in as two b128 LDS reads per
        // M-fragment instead of one b32 per element. RS carries rowsum*asg (prologue), so the
        // correction is a single FMA with no asg factor.
        const int mlb = i * 16 + kb8;
        const float4 a4l = *(const float4 *)&s_asg[mlb], a4h = *(const float4 *)&s_asg[mlb + 4];
        const float4 r4l = *(const float4 *)&s_rs[mlb], r4h = *(const float4 *)&s_rs[mlb + 4];
        const float av[8] = {a4l.x, a4l.y, a4l.z, a4l.w, a4h.x, a4h.y, a4h.z, a4h.w};
        const float rv[8] = {r4l.x, r4l.y, r4l.z, r4l.w, r4h.x, r4h.y, r4h.z, r4h.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if constexpr (ABLATE & 2) acc[i][e] += t[e];
          else acc[i][e] = fmaf(sc, av[e] * t[e], fmaf(-zsc, rv[e], acc[i][e]));
        }
      }
      __syncthreads();
      continue;
    }

    floatx8 gt[DTM];
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) gt[i][e] = 0.f;

#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa;
        af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw;
      wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        gt[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, gt[i]);
    }

#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int ml = i * 16 + kb8 + e;
        acc[i][e] = fmaf(sc, s_asg[ml] * gt[i][e], fmaf(-zsc, s_rs[ml], acc[i][e]));
      }

    __syncthreads();
  }

  // Epilogue: the activation scale already folded per slab, so C is acc verbatim.
  if constexpr (DKS == 1) {
    if (n_lane < N) {
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n_lane] = (__bf16)acc[i][e];
        }
    }
    return;
  }

  if (n_lane < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n_lane] = acc[i][e];
      }
  }

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)sum;
    }
  }
}

// ------------------------------------------------------------------ prefill path (large M)
//
// Structure is the AutoRound prefill kernel (BMF=256, BK=64, group = two slabs, IMAJOR TN temp
// tiles). PARO deltas mirror the decode kernel's, with one twist from BK=64: the WMMA fold runs
// per slab (asg * sc * t), while the zero-point correction -zsc*asg*rs is applied ONCE per group,
// on the group's second slab -- rs is a per-group quantity and both slabs share sc and asg, so
// folding t per slab and correcting per group is arithmetically the same sum.
#ifndef AR_TM
#define AR_TM 4
#endif
#ifndef AR_WM
#define AR_WM 4
#endif
#ifndef AR_WN
#define AR_WN 2
#endif
#define AR_BK 64
#define AR_PAD 8
#define AR_ASTR (AR_BK + AR_PAD)
#define AR_NWAVE (AR_WM * AR_WN)
#define AR_NTHREADS (AR_NWAVE * 32)
#define AR_BMF (AR_WM * AR_TM * 16)

// PTOK: per-TOKEN activation scale (prefill serving path). ASG is then As [P, M] and the fold
// collapses to AutoRound's single FMA per slab; the zero-point correction stays one FMA per
// element per group against PLAIN code row-sums, and As multiplies once in the epilogue. The
// per-group variant (PTOK=false) remains for the decode-band fallthrough and the harness.
template <int TN, bool IMAJOR, int ABLATE = 0, bool PTOK = false, bool WPERM = false>
__global__ __launch_bounds__(AR_NTHREADS) void pq_int4_fp8_gemm_prefill(
    const unsigned char *__restrict__ A, const unsigned int *__restrict__ W,
    const __half *__restrict__ SZ, const float *__restrict__ ASG,
    const float *__restrict__ RS, __bf16 *__restrict__ C, int M, int N, int K, int pb1,
    int pb2) {
  constexpr int BNF_T = AR_WN * TN * 16;
  __shared__ unsigned char sA[AR_BMF * AR_ASTR];
  __shared__ unsigned char sW[BNF_T * AR_ASTR];
  __shared__ float s_asg[AR_BMF];
  __shared__ float s_rs[AR_BMF];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / AR_WN, wn = wave % AR_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * AR_BMF, n0 = blockIdx.x * BNF_T;
  const int kw = K / 8;

  // PARO: partition select. Under PTOK the ASG argument is As [P, M].
  const int prt = (n0 >= pb1 ? 1 : 0) + (n0 >= pb2 ? 1 : 0);
  const int G = K / PQ_GROUP;
  A += (size_t)prt * M * K;
  if constexpr (PTOK) ASG += (size_t)prt * M;
  else ASG += (size_t)prt * M * G;
  RS += (size_t)prt * M * G;

  floatx8 acc[AR_TM][TN], tmp[IMAJOR ? 1 : AR_TM][TN];
#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
  if constexpr (!IMAJOR)
#pragma unroll
    for (int i = 0; i < AR_TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;

  int ncol[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) ncol[j] = n0 + wn * TN * 16 + j * 16 + col;

  // Both slabs of a group share sc/zsc: load them on the group's FIRST slab and carry them in
  // registers across the second. Half the scale loads the AutoRound kernel pays.
  float sc[TN], zsc[TN];
  for (int k0 = 0; k0 < K; k0 += AR_BK) {
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    const unsigned int *__restrict__ Wb = W + (size_t)n0 * kw + k0 / 8;
    const int g = k0 / PQ_GROUP;
    const bool second = ((k0 / AR_BK) & 1) == 1;   // group's second slab -> apply correction
    if (!second) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        if constexpr ((ABLATE & 2) != 0) {
          sc[j] = 1.f; zsc[j] = 0.f;
        } else {
          // Clamped, not predicated -- see ar_kernels.h.
          const __half2 szv =
              *(const __half2 *)(SZ + (((size_t)g * N + (ncol[j] < N ? ncol[j] : N - 1)) * 2));
          sc[j] = __half2float(szv.x);
          zsc[j] = __half2float(szv.y);
        }
      }
    }
    // PARO: stage asg/rsa rows for this group -- on the group's FIRST slab only (both slabs
    // share the group's values, and the __syncthreads() pair around the WMMA run already orders
    // the reuse). 256 threads cover the 256 rows in one pass.
    if (!second && !(ABLATE & 4)) {
      const int r = tid;
      const int rc = (m0 + r) < M ? (m0 + r) : (M > 0 ? M - 1 : 0);
      if constexpr (!PTOK) s_asg[r] = ASG[(size_t)rc * G + g];
      s_rs[r] = RS[(size_t)rc * G + g];
    }
#pragma unroll
    for (int off = 0; off < AR_BMF * AR_BK; off += AR_NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / AR_BK, c = idx % AR_BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      *(uint4_t *)(&sA[r * AR_ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    pq_stage_w<BNF_T, AR_BK, AR_ASTR, AR_NTHREADS, WPERM, false, ABLATE>(sW, W, n0, N, K, k0,
                                                                          tid);
    __syncthreads();

    if constexpr (IMAJOR) {
      int2_t wfa[AR_BK / 16][TN];
#pragma unroll
      for (int step = 0; step < AR_BK / 16; ++step)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const unsigned char *p =
              &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + step * 16 + kb8];
          wfa[step][j][0] = *(const int *)p; wfa[step][j][1] = *(const int *)(p + 4);
        }
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        floatx8 t[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) t[j][e] = 0.f;
#pragma unroll
        for (int step = 0; step < AR_BK / 16; ++step) {
          int2_t af;
          const unsigned char *pa =
              &sA[(wm * AR_TM * 16 + i * 16 + col) * AR_ASTR + step * 16 + kb8];
          af[0] = *(const int *)pa; af[1] = *(const int *)(pa + 4);
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int j = 0; j < TN; ++j)
            t[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wfa[step][j], t[j]);
        }
        // asg/rsa for this M-fragment's 8 contiguous rows: two b128 LDS reads each, hoisted out
        // of the per-element fold. Fold cost: 2 VALU per element per slab for the scale, plus
        // ONE fma per element per GROUP for the zero-point correction (RS carries rowsum*asg).
        // The old per-element form (b32 read + 3-op chain, per slab) compiled to 209 VGPRs /
        // 7 waves against the AutoRound kernel's 128 / 8-10, and measured 2.7x its time.
        if constexpr ((ABLATE & 4) != 0) {
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] += sc[j] * t[j][e];
          continue;
        }
        const int mlb = wm * AR_TM * 16 + i * 16 + kb8;
        if constexpr (PTOK) {
          // Per-token scale: exactly the AutoRound fold; As lands once in the epilogue.
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = fmaf(sc[j], t[j][e], acc[i][j][e]);
        } else {
          const float4 a4l = *(const float4 *)&s_asg[mlb],
                       a4h = *(const float4 *)&s_asg[mlb + 4];
          const float av[8] = {a4l.x, a4l.y, a4l.z, a4l.w, a4h.x, a4h.y, a4h.z, a4h.w};
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              acc[i][j][e] = fmaf(sc[j], av[e] * t[j][e], acc[i][j][e]);
        }
        if (second) {
          const float4 r4l = *(const float4 *)&s_rs[mlb], r4h = *(const float4 *)&s_rs[mlb + 4];
          const float rv[8] = {r4l.x, r4l.y, r4l.z, r4l.w, r4h.x, r4h.y, r4h.z, r4h.w};
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              acc[i][j][e] = fmaf(-zsc[j], rv[e], acc[i][j][e]);
        }
      }
      __syncthreads();
      continue;
    }

#pragma unroll
    for (int step = 0; step < AR_BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[AR_TM], wf[TN];
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        const unsigned char *p = &sA[(wm * AR_TM * 16 + i * 16 + col) * AR_ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * AR_ASTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < AR_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
    }
    __syncthreads();

    if (second) {
#pragma unroll
      for (int i = 0; i < AR_TM; ++i) {
        const int mlb = wm * AR_TM * 16 + i * 16 + kb8;
        const float4 r4l = *(const float4 *)&s_rs[mlb], r4h = *(const float4 *)&s_rs[mlb + 4];
        const float rv[8] = {r4l.x, r4l.y, r4l.z, r4l.w, r4h.x, r4h.y, r4h.z, r4h.w};
        float av[8];
        if constexpr (PTOK) {
#pragma unroll
          for (int e = 0; e < 8; ++e) av[e] = 1.f;
        } else {
          const float4 a4l = *(const float4 *)&s_asg[mlb],
                       a4h = *(const float4 *)&s_asg[mlb + 4];
          av[0] = a4l.x; av[1] = a4l.y; av[2] = a4l.z; av[3] = a4l.w;
          av[4] = a4h.x; av[5] = a4h.y; av[6] = a4h.z; av[7] = a4h.w;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[i][j][e] = fmaf(sc[j], av[e] * tmp[i][j][e], fmaf(-zsc[j], rv[e], acc[i][j][e]));
            tmp[i][j][e] = 0.f;
          }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * AR_TM * 16 + i * 16 + kb8 + e;
        if (m < M && ncol[j] < N) {
          float o = acc[i][j][e];
          if constexpr (PTOK) o *= ASG[(size_t)m];
          C[(size_t)m * N + ncol[j]] = (__bf16)o;
        }
      }
}

// ------------------------------------------------------------ A-tiled prefill path (per-token)
//
// The activation arrives in WMMA-FRAGMENT-TILED layout, the layout the MXFP4 A-tiled kernel
// reads: each 16m x 16k e4m3 fragment is 256 contiguous bytes in lane order (lane l owns bytes
// 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l/16)*8 .. +7]); fragment (mt, ks) sits at
// (mt*(K/16) + ks)*256 and a partition is Mt*16*K bytes (Mt = ceil(M/16); pad rows hold whatever
// the producer left there -- they only reach accumulators the epilogue drops). One coalesced
// global_load_b64 per fragment per wave lands straight in the af register the WMMA reads, so
// the A tile, its LDS staging and both its LDS round trips are gone; only W goes through LDS
// (8.7 KB at TN=2 / LBK=128), so three blocks share a CU instead of two. On MXFP4 this was
// 12-16% off the folded kernel at M >= 1024, and the A tile is a LARGER share of this kernel
// (the per-slab rescale keeps the tile in LDS for a temp accumulator round anyway).
//
// LBK=128 makes the slab the scale group: sc/zsc are loaded once, the temp accumulator runs
// eight WMMA steps, and the fold (1 FMA/elem for the scale) plus the zero-point correction
// (1 FMA/elem against the plain code row-sums) land once per 128 k -- half the fold VALU the
// BK=64 PTOK kernel pays. The price is af[TM][8] = 64 VGPRs of fragments in flight; LBK=64 keeps
// the old two-slab group structure (correction on the second slab) at 32. Both are built, the
// launcher takes what the harness measured.
//
// WHOIST: read the slab's W fragments from LDS once per wave (wfa[NS][TN], 32 VGPRs at
// LBK=128) instead of once per M-fragment (TM x the ds_reads). Measured either way.
// Prefill pass C, tiled output. The first cut read each lane's 16 B from sixteen different rows
// per k-step and measured 1.3-2x the row kernel (tier: passC bench). This one keeps the row
// kernel's contiguous read (one wave sweeps one 256 B row segment per group), parks the encoded
// 16 x 128 tile in LDS at a 136 B row stride (conflict-free for both the 4 B row writes and the
// 8 B fragment reads), and writes each 256 B fragment with one wave store. Block = one m-tile
// (16 rows) x 8 waves; wave w takes groups g = w, w+8, ... Codes/AS are identical to
// pq_token_quant (same inv, same encode); RS is the same 32-lane tree per (row, group).
#define PQ_TQ_STR 136
// Group split for pq_token_quant_tiled: enough blocks to fill 64 CUs x ~4 blocks, capped at 4.
static inline int pq_tq_zsplit(int M) {
  const int Mt = (M + 15) / 16;
  int z = 512 / (Mt > 0 ? Mt : 1);
  return z < 1 ? 1 : (z > 4 ? 4 : z);
}
__global__ __launch_bounds__(PQ_ROT_WAVES * 32) void pq_token_quant_tiled(
    const __bf16 *__restrict__ XR,          // [P, M, K] rotated values (pass A)
    const float *__restrict__ ASG,          // [P, M, K/128] per-group scales (pass A)
    unsigned char *__restrict__ AT,         // [P, Mt*16, K] e4m3 codes out, fragment-tiled
    float *__restrict__ AS,                 // [P, M] per-token scale out
    float *__restrict__ RS,                 // [P, M, K/128] code row-sums out
    int M, int K) {
  const int G = K / PQ_GROUP, ksteps = K / 16;
  const int Mt = (M + 15) >> 4;
  const int p = blockIdx.y, mt = blockIdx.x;
  // gridDim.z splits the groups across blocks so a small M still fills the GPU (one block per
  // m-tile is 128 blocks at M=2048); split z takes g = z*8 + wave, stepping by 8*gridDim.z.
  const int zs = blockIdx.z, nz = gridDim.z;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  __shared__ __align__(16) unsigned char s_t[PQ_ROT_WAVES][16 * PQ_TQ_STR];

  // Token scales: lane l < 16 owns row mt*16 + l (clamped; pad rows recompute the last row and
  // never write AS/RS). Broadcast per row below with a shuffle.
  const int mrow = mt * 16 + (lane & 15);
  const int mc = mrow < M ? mrow : M - 1;
  float inv_l;
  {
    const float *asg_row = ASG + ((size_t)p * M + mc) * G;
    float amx = 0.f;
    int g4 = 0;
    for (; g4 + 4 <= G; g4 += 4) {
      const float4 v = *(const float4 *)(asg_row + g4);
      amx = fmaxf(amx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    for (; g4 < G; ++g4) amx = fmaxf(amx, asg_row[g4]);
    const float scale = fmaxf(amx, 1e-10f);
    inv_l = 1.f / scale;
    if (zs == 0 && wave == 0 && lane < 16 && mrow < M) AS[(size_t)p * M + mrow] = scale;
  }

  unsigned char *tile = s_t[wave];
  const int c0 = lane * 4;
  unsigned char *at = AT + (size_t)p * Mt * 16 * K + (size_t)mt * ksteps * 256 + lane * 8;
  for (int g = zs * PQ_ROT_WAVES + wave; g < G; g += PQ_ROT_WAVES * nz) {
    // Phase 1: sixteen contiguous row reads, encode, park in LDS, row-sum per row.
#pragma unroll 4
    for (int r = 0; r < 16; ++r) {
      const int m = mt * 16 + r, mr = m < M ? m : M - 1;
      const float inv = __shfl(inv_l, r, 32);
      const uint2_t xv = *(const uint2_t *)(XR + ((size_t)p * M + mr) * K + (size_t)g * PQ_GROUP + c0);
      const float v0 = __uint_as_float(xv[0] << 16) * inv, v1 = __uint_as_float(xv[0] & 0xFFFF0000u) * inv;
      const float v2 = __uint_as_float(xv[1] << 16) * inv, v3 = __uint_as_float(xv[1] & 0xFFFF0000u) * inv;
      const unsigned char b0 = pq_e4m3_encode(v0), b1 = pq_e4m3_encode(v1);
      const unsigned char b2 = pq_e4m3_encode(v2), b3 = pq_e4m3_encode(v3);
      *(unsigned int *)(tile + r * PQ_TQ_STR + c0) =
          (unsigned int)b0 | ((unsigned int)b1 << 8) | ((unsigned int)b2 << 16) | ((unsigned int)b3 << 24);
      float rs = pq_e4m3_decode(b0) + pq_e4m3_decode(b1) + pq_e4m3_decode(b2) + pq_e4m3_decode(b3);
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) rs += __shfl_xor(rs, off, 32);
      if (lane == 0 && m < M) RS[((size_t)p * M + m) * G + g] = rs;
    }
    __asm__ volatile("s_waitcnt lgkmcnt(0)");
    // Phase 2: eight fragment stores, 256 B contiguous each. Lane l reads row l&15, k-half l>>4.
    const unsigned char *src = tile + (lane & 15) * PQ_TQ_STR + (lane >> 4) * 8;
#pragma unroll
    for (int st = 0; st < PQ_GROUP / 16; ++st) {
      const uint2_t v = *(const uint2_t *)(src + st * 16);
      *(uint2_t *)(at + (size_t)(g * (PQ_GROUP / 16) + st) * 256) = v;
    }
    __asm__ volatile("s_waitcnt lgkmcnt(0)");   // reads done before the next group's writes
  }
}

template <int TN, bool WPERM, int LBK, bool WHOIST = true, int WSLOT_OVR = 0>
__global__ __launch_bounds__(AR_NTHREADS) void pq_int4_fp8_gemm_atiled(
    const unsigned char *__restrict__ AT,   // [P, Mt*16, K] fragment-tiled e4m3
    const unsigned int *__restrict__ W, const __half *__restrict__ SZ,
    const float *__restrict__ AS,           // [P, M] per-token scale
    const float *__restrict__ RS,           // [P, M, K/128] plain code row-sums
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2) {
  static_assert(LBK == 64 || LBK == 128, "slab is one group or half a group");
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + AR_PAD;
  constexpr int BNF_T = AR_WN * TN * 16;
  __shared__ unsigned char sW[BNF_T * LWSTR];
  __shared__ float s_rs[AR_BMF];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / AR_WN, wn = wave % AR_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * AR_BMF, n0 = blockIdx.x * BNF_T;
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  // PARO: partition select.
  const int prt = (n0 >= pb1 ? 1 : 0) + (n0 >= pb2 ? 1 : 0);
  const int G = K / PQ_GROUP;
  AT += (size_t)prt * Mt * 16 * K;
  AS += (size_t)prt * M;
  RS += (size_t)prt * M * G;

  // Wave-uniform tile bases (SGPR) + one per-lane offset; tile-granular clamp, never predicate.
  const unsigned char *abase[AR_TM];
#pragma unroll
  for (int i = 0; i < AR_TM; ++i) {
    int mt = (m0 >> 4) + wm * AR_TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)mt * ksteps_g * 256;
  }
  const int aoff = lane * 8;

  floatx8 acc[AR_TM][TN];
#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  int ncol[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) ncol[j] = n0 + wn * TN * 16 + j * 16 + col;

  float sc[TN], zsc[TN];
  for (int k0 = 0; k0 < K; k0 += LBK) {
    const int ks0 = k0 / 16, g = k0 / PQ_GROUP;
    // LBK=128: every slab is a whole group. LBK=64: the group's first slab loads the scales and
    // stages the row-sums, the second applies the correction (both share sc/zsc/rs).
    const bool first = (LBK == PQ_GROUP) || (((k0 / AR_BK) & 1) == 0);
    const bool second = (LBK == PQ_GROUP) || !first;
    // A fragments straight from global, issued before the W staging so they land under it.
    int2_t af[AR_TM][NS];
#pragma unroll
    for (int i = 0; i < AR_TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + ((ks0 + st) * 256 + aoff));
    if (first) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const __half2 szv =
            *(const __half2 *)(SZ + (((size_t)g * N + (ncol[j] < N ? ncol[j] : N - 1)) * 2));
        sc[j] = __half2float(szv.x);
        zsc[j] = __half2float(szv.y);
      }
      const int r = tid;
      const int rc = (m0 + r) < M ? (m0 + r) : (M > 0 ? M - 1 : 0);
      s_rs[r] = RS[(size_t)rc * G + g];
    }
    pq_stage_w<BNF_T, LBK, LWSTR, AR_NTHREADS, WPERM, false, 0, WSLOT_OVR>(sW, W, n0, N, K, k0,
                                                                          tid);
    __syncthreads();

    int2_t wfa[WHOIST ? NS : 1][TN];
    if constexpr (WHOIST) {
#pragma unroll
      for (int st = 0; st < NS; ++st)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const unsigned char *pw = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + st * 16 + kb8];
          wfa[st][j][0] = *(const int *)pw; wfa[st][j][1] = *(const int *)(pw + 4);
        }
    }
#pragma unroll
    for (int i = 0; i < AR_TM; ++i) {
      floatx8 t[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) t[j][e] = 0.f;
#pragma unroll
      for (int st = 0; st < NS; ++st) {
        if constexpr (WHOIST) {
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int j = 0; j < TN; ++j)
            t[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][st], wfa[st][j], t[j]);
        } else {
          int2_t wf[TN];
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const unsigned char *pw = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + st * 16 + kb8];
            wf[j][0] = *(const int *)pw; wf[j][1] = *(const int *)(pw + 4);
          }
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int j = 0; j < TN; ++j)
            t[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][st], wf[j], t[j]);
        }
      }
      const int mlb = wm * AR_TM * 16 + i * 16 + kb8;
      if (second) {
        const float4 r4l = *(const float4 *)&s_rs[mlb], r4h = *(const float4 *)&s_rs[mlb + 4];
        const float rv[8] = {r4l.x, r4l.y, r4l.z, r4l.w, r4h.x, r4h.y, r4h.z, r4h.w};
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e)
            acc[i][j][e] = fmaf(sc[j], t[j][e], fmaf(-zsc[j], rv[e], acc[i][j][e]));
      } else {
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] = fmaf(sc[j], t[j][e], acc[i][j][e]);
      }
    }
    __syncthreads();
  }

  // Epilogue: As once per row. Full-tile fast path (no per-element guards) when the tile is
  // interior, else the guarded form.
  {
    const bool full = (m0 + wm * AR_TM * 16 + (AR_TM - 1) * 16 + kb8 + 7 < M) &&
                      (ncol[TN - 1] < N);
    if (full) {
      __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * AR_TM * 16) * N;
      const float *__restrict__ Asb = AS + m0 + wm * AR_TM * 16;
#pragma unroll
      for (int i = 0; i < AR_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            Cb[r * N + ncol[j]] = (__bf16)(acc[i][j][e] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < AR_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * AR_TM * 16 + i * 16 + kb8 + e;
        if (m < M && ncol[j] < N) C[(size_t)m * N + ncol[j]] = (__bf16)(acc[i][j][e] * AS[m]);
      }
}
