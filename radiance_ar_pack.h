#pragma once
// radiance_ar_pack.h: rotated 6-bit payload for the one-shot P2P all-reduce on dual R9700
// (gfx1201/RDNA4, TP=2, PCIe). Companion to radiance_ar_ext.hip: reuses the IPC scratch,
// per-block flags and device-resident seq counters that module allocates. Dispatched for
// large messages when RADIANCE_AR_QUANT=1.
//
// Design:
//  * Each group of GROUP=64 is rotated by an orthonormal Walsh-Hadamard, scaled by its own
//    amax and stored as BITS=6 uniform bits plus a bf16 scale -> 6.25 bits/element. The
//    rotation removes the outlier channel, which is what makes 6 uniform bits enough.
//  * PUSH rotates, quantizes and packs straight into the peer's scratch, keeping a copy of
//    the same words locally. REDUCE unpacks both halves, adds in fp32, inverse-rotates (the
//    transform is its own inverse) and stores bf16/fp16.
//  * The local half is kept packed rather than recomputed: re-deriving it would repeat the
//    rotation and the amax reduction.
//  * Both ranks fold the identical pair and fp32 add commutes, so the ranks stay
//    bit-identical. MUST be built with -ffp-contract=off: the two ranks see the two products
//    in opposite order, and contracting either into an FMA makes them disagree by ~1 ULP.
//  * cudagraph-safe and double-buffered as in radiance_ar_ext.hip: per-block device-resident
//    seq counter, slot = seq&1.
//
// Layout, per slot, inside the shared 2*max_bytes scratch:
//   [ packed symbols, chunk-interleaved ] ... [ scale_off_bytes ] ... [ bf16 scales: n_groups ]
// A CHUNK is 32 groups = 2048 elements. One wave owns a chunk and every lane emits exactly 64
// symbols = 384 bits = three uint4, so flush points are lane-uniform and lane L's u-th uint4
// lands at c*U4PC + u*WAVE + L: one coalesced 512B store per wave.
//
// n_groups need not tile a chunk. Groups past the end quantize zeros and are skipped on the way
// out. That is only safe because a group spans LPG lane-contiguous lanes and every cross-lane
// exchange in wht() is intra-group (m < LPG), so a group is wholly active or wholly inactive and
// no shuffle is ever split across the guard.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define RADIANCE_SPIN_MAX 4000000000ULL
#define RADIANCE_MAX_BLOCKS 512
#define GROUP 64                  // elements per rotation + scaling group
#define BITS 6                    // bits per symbol
#define EPL 8                     // elements per lane per group
#define LPG (GROUP / EPL)         // 8 lanes span one group
#define WAVE 32
#define GPW (WAVE * EPL / GROUP)  // 4 groups per wave per iteration
#define CHUNK_GROUPS 32           // groups per wave-chunk (makes 64 symbols/lane = 3 uint4)
#define CHUNK_ELEMS (CHUNK_GROUPS * GROUP)
#define ITERS (CHUNK_ELEMS / (WAVE * EPL))          // 8
#define U4PC (CHUNK_ELEMS * BITS / 8 / 16)          // 96 uint4 per chunk
#define NWORD (ITERS * EPL * BITS / 32)             // 12 words held per lane
#define NU4 (NWORD / 4)                             // 3 uint4 per lane

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void do_drain(int drain) {
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
}

__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// RTNE to bf16 and back: the scale ships as its top 16 bits, so the sender's local copy has to be
// exactly what the receiver reconstructs.
__device__ __forceinline__ float bf16_round(float v) {
  unsigned u = __float_as_uint(v);
  const unsigned lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return __uint_as_float(u & 0xffff0000u);
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this routine also undoes it.
// A group is LPG lanes x EPL elements: the in-register butterflies are free ALU, only the
// cross-lane stages cost a shuffle.
__device__ __forceinline__ void wht(float* v, int lig) {
#pragma unroll
  for (int h = 1; h < EPL; h <<= 1)
#pragma unroll
    for (int i = 0; i < EPL; ++i)
      if ((i & h) == 0) { const float x = v[i], y = v[i ^ h]; v[i] = x + y; v[i ^ h] = x - y; }
#pragma unroll
  for (int m = 1; m < LPG; m <<= 1) {
    const bool lo = (lig & m) == 0;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const float o = __shfl_xor(v[i], m);
      v[i] = lo ? (v[i] + o) : (o - v[i]);
    }
  }
  const float nrm = rsqrtf((float)GROUP);
#pragma unroll
  for (int i = 0; i < EPL; ++i) v[i] *= nrm;
}

template <typename T>
__global__ void radiance_ar_mb_pack(
    const T* __restrict__ in, char* peer_scratch, const char* my_scratch, char* loc_pack,
    T* __restrict__ out, unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n_groups, int n_chunks, long slot_stride_bytes, long scale_off_bytes, int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid & (WAVE - 1), wid = tid / WAVE, nwarp = nt / WAVE;
  const int lig = lane & (LPG - 1), gw = lane / LPG;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;   // per-block, replay-safe (cudagraph)
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  char* pbase = peer_scratch + (size_t)slot * slot_stride_bytes;
  const char* mbase = my_scratch + (size_t)slot * slot_stride_bytes;
  uint4* p_pay = (uint4*)pbase;
  unsigned short* p_sc = (unsigned short*)(pbase + scale_off_bytes);
  const uint4* m_pay = (const uint4*)mbase;
  const unsigned short* m_sc = (const unsigned short*)(mbase + scale_off_bytes);
  uint4* l_pay = (uint4*)loc_pack;                          // this rank's half, local only
  float* l_sc = (float*)(loc_pack + scale_off_bytes);

  const int cpb = (n_chunks + nb - 1) / nb;                 // block b owns chunks [c0,c1)
  const int c0 = b * cpb; int c1 = c0 + cpb; if (c1 > n_chunks) c1 = n_chunks;

  const int qmin = -(1 << (BITS - 1)), qmax = (1 << (BITS - 1)) - 1;
  const unsigned int MASK = (1u << BITS) - 1u;

  // PUSH: rotate, amax, quantize, pack -> peer scratch, plus a local copy for the reduce.
  for (int c = c0 + wid; c < c1; c += nwarp) {
    unsigned int acc[NWORD];
#pragma unroll
    for (int k = 0; k < NWORD; ++k) acc[k] = 0u;
#pragma unroll
    for (int t = 0; t < ITERS; ++t) {
      const int g = c * CHUNK_GROUPS + t * GPW + gw;
      const bool live = (g < n_groups);                     // uniform across a group's lanes
      const long off = (long)g * GROUP + (long)lig * EPL;
      float v[EPL];
#pragma unroll
      for (int i = 0; i < EPL; ++i) v[i] = live ? rd_to_f(in[off + i]) : 0.f;
      wht(v, lig);
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < EPL; ++i) a = fmaxf(a, fabsf(v[i]));
#pragma unroll
      for (int m = 1; m < LPG; m <<= 1) a = fmaxf(a, __shfl_xor(a, m));
      float sc = (a > 0.f) ? bf16_round(a * (1.0f / (float)qmax)) : 1.0f;
      if (!(sc > 0.f)) sc = 1.0f;
      if (lig == 0 && live) {
        p_sc[g] = (unsigned short)(__float_as_uint(sc) >> 16);
        l_sc[g] = sc;
      }
      const float inv = 1.0f / sc;
#pragma unroll
      for (int i = 0; i < EPL; ++i) {
        const int q = min(max((int)rintf(v[i] * inv), qmin), qmax) - qmin;
        const unsigned int sym = (unsigned int)q;
        const int bit = (t * EPL + i) * BITS, wi = bit >> 5, bo = bit & 31;
        acc[wi] |= sym << bo;
        if (bo + BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
      }
    }
#pragma unroll
    for (int u = 0; u < NU4; ++u) {
      const uint4 q = make_uint4(acc[4 * u], acc[4 * u + 1], acc[4 * u + 2], acc[4 * u + 3]);
      const long o = (long)c * U4PC + (long)u * WAVE + lane;
      p_pay[o] = q;
      l_pay[o] = q;
    }
  }

  // handshake (drain the pushes into the fabric, then a per-block release flag + spin)
  do_drain(drain);
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0; while (load_sys_acq(&my_flags[b]) < s) { if (++z > RADIANCE_SPIN_MAX) break; } }
  __syncthreads();
  if (acq) __threadfence_system();

  // REDUCE: unpack both halves, add in fp32, inverse-rotate, store.
  for (int c = c0 + wid; c < c1; c += nwarp) {
    unsigned int pac[NWORD], lac[NWORD];
#pragma unroll
    for (int u = 0; u < NU4; ++u) {
      const long o = (long)c * U4PC + (long)u * WAVE + lane;
      const uint4 pq = m_pay[o], lq = l_pay[o];
      pac[4 * u] = pq.x; pac[4 * u + 1] = pq.y; pac[4 * u + 2] = pq.z; pac[4 * u + 3] = pq.w;
      lac[4 * u] = lq.x; lac[4 * u + 1] = lq.y; lac[4 * u + 2] = lq.z; lac[4 * u + 3] = lq.w;
    }
#pragma unroll
    for (int t = 0; t < ITERS; ++t) {
      const int g = c * CHUNK_GROUPS + t * GPW + gw;
      if (g >= n_groups) continue;                          // whole group: no shuffle is split
      const long off = (long)g * GROUP + (long)lig * EPL;
      const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
      const float lsc = l_sc[g];
      float v[EPL];
#pragma unroll
      for (int i = 0; i < EPL; ++i) {
        const int bit = (t * EPL + i) * BITS, wi = bit >> 5, bo = bit & 31;
        unsigned int ps = pac[wi] >> bo, ls = lac[wi] >> bo;
        if (bo + BITS > 32) { ps |= pac[wi + 1] << (32 - bo); ls |= lac[wi + 1] << (32 - bo); }
        v[i] = (float)((int)(ps & MASK) + qmin) * psc + (float)((int)(ls & MASK) + qmin) * lsc;
      }
      wht(v, lig);
#pragma unroll
      for (int i = 0; i < EPL; ++i) rd_from_f(v[i], out[off + i]);
    }
  }
}

static int clamp_blocks(int nb, int upper) {
  if (nb < 1) nb = 1; if (nb > upper) nb = upper;
  if (nb > RADIANCE_MAX_BLOCKS) nb = RADIANCE_MAX_BLOCKS; return nb;
}
static int clamp_threads(int nt) { if (nt < 64) nt = 64; if (nt > 1024) nt = 1024; return (nt / 64) * 64; }

// Operates on scratch/flags/seq allocated by radiance_ar_ext; slot_stride_bytes = max_bytes,
// scale_off_bytes = max_bytes/2. loc_pack is a plain local buffer (not IPC) holding this rank's
// packed half and its fp32 scales, split at the same scale_off_bytes.
// bf16/fp16 payloads only; n_elem must be a multiple of GROUP.
static void all_reduce_pack(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                            int64_t my_flags, int64_t seq_ctrs, int64_t loc_pack,
                            int64_t slot_stride_bytes, int64_t scale_off_bytes, int64_t inp,
                            int64_t out, int64_t n_elem, int64_t dtype, int64_t stream_i,
                            int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq) {
  hipStream_t st = (hipStream_t)stream_i;
  if (dtype != 0 && dtype != 1) throw std::runtime_error("pack ar: only bf16/fp16 payloads");
  if (n_elem % GROUP != 0) throw std::runtime_error("pack ar: n_elem not a multiple of GROUP");
  const int n_groups = (int)(n_elem / GROUP);
  const int n_chunks = (n_groups + CHUNK_GROUPS - 1) / CHUNK_GROUPS;
  const int nb = clamp_blocks((int)nblocks, n_chunks);
  const int nt = clamp_threads((int)nthreads);
#define LF(T) radiance_ar_mb_pack<T><<<nb, nt, 0, st>>>( \
    (const T*)inp, (char*)peer_scratch, (const char*)my_scratch, (char*)loc_pack, (T*)out, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n_groups, n_chunks, \
    (long)slot_stride_bytes, (long)scale_off_bytes, (int)drain, (int)acq)
  if (dtype == 0) LF(__hip_bfloat16);
  else LF(__half);
#undef LF
}
