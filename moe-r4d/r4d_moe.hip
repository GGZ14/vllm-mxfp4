// r4d_moe: libr4d v0.5.0's paged prefill attention, instantiated for GQA 8 (Qwen3.5-35B-A3B: 16 q /
// 2 kv heads, head 256), as a pybind11 module whose entry points take exactly the positional
// arguments of r4d.attn_prefill_h256_gqa6_{fp8,bf16}kv. radiance_r4d_attn.py can therefore call
// either one from the same call site.
//
// r4d_attn_prefill_kernel is a template over <NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT>; the image's
// r4d.so only instantiates GQA 6. Nothing in the kernel depends on GQA beyond ROWS % GQA == 0
// (384 rows = 48 query tokens x 8 heads per workgroup), so this is the same kernel, launch law
// (24 warps, 48-key tile, opt 3955211) and 16-token paged block, one template argument apart.
// r4d.h / r4d_common.h / r4d_dt16.h / r4d_attn_prefill_h256_gqa6.hip come from a libr4d v0.5.0
// checkout (R4D_SRC, see build.sh); nothing from libr4d is copied into this repo.
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"

namespace py = pybind11;

#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211
#define M_GQA    8
#define M_BS     16

template <int KVP>
static int prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != 256 || a->block_size != M_BS) return -1;
    if (a->kv_heads <= 0 || a->q_heads != a->kv_heads * M_GQA) return -2;
    if (a->q_len <= 0 || a->num_seqs <= 0) return -3;
    constexpr int BQ = (P_NWARPS * 16) / M_GQA;
    static_assert((P_NWARPS * 16) % M_GQA == 0, "rows must hold whole query tokens");
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, 256, M_GQA, M_BS, KVP, P_OPT>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}

template <int KVP>
static void attn_prefill(int64_t q, int64_t kv, int64_t block_table, int64_t seqused_k, int64_t out,
                         int64_t k_descale, int64_t v_descale, int64_t scratch, int num_seqs,
                         int q_len, int q_heads, int kv_heads, int head_dim, int block_size,
                         int max_blocks, int64_t kv_block_stride, int64_t kv_head_stride,
                         double scale, int splits, int max_ctx, int64_t stream) {
    R4DArgs a{};
    a.q = (const void*)q;
    a.kv = (const void*)kv;
    a.block_table = (const int*)block_table;
    a.seqused_k = (const int*)seqused_k;
    a.out = (void*)out;
    a.k_descale = (const float*)k_descale;
    a.v_descale = (const float*)v_descale;
    a.q_descale = nullptr;
    a.scratch = (void*)scratch;
    a.num_seqs = num_seqs;
    a.q_len = q_len;
    a.q_heads = q_heads;
    a.kv_heads = kv_heads;
    a.head_dim = head_dim;
    a.block_size = block_size;
    a.max_blocks = max_blocks;
    a.kv_block_stride = (long)kv_block_stride;
    a.kv_head_stride = (long)kv_head_stride;
    a.scale = (float)scale;
    a.splits = splits;
    a.max_ctx = max_ctx;
    const int rc = prefill_launch<KVP>(&a, (hipStream_t)stream);
    if (rc != 0)
        throw std::runtime_error(std::string("r4d_moe attn_prefill_h256_gqa8_") +
                                 (KVP ? "bf16kv" : "fp8kv") + " refused (code " +
                                 std::to_string(rc) + ")");
}

PYBIND11_MODULE(r4d_moe, m) {
    m.doc() = "libr4d v0.5.0 paged prefill attention instantiated at GQA 8 (head 256, block 16)";
    m.attr("ATTN_GQA") = M_GQA;
    m.attr("ATTN_HEAD_DIM") = 256;
    m.attr("ATTN_BLOCK_SIZE") = M_BS;
    m.attr("R4D_BASE") = "v0.5.0";
    m.def("attn_prefill_h256_gqa8_fp8kv", &attn_prefill<0>);
    m.def("attn_prefill_h256_gqa8_bf16kv", &attn_prefill<1>);
}
