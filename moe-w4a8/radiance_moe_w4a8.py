"""radiance_moe_w4a8: W4A8 (MXFP4 weights x fp8 e4m3 activations) expert GEMMs for the gfx1201 MoE lane, on the fp8
WMMA (radiance-moe-w4a8-2/). Drop-in for the two moe_gemm_a16w4 calls of vLLM's aiter_triton_kernel_w4a16_moe_forward:
same routing (AITER block_pid_map / gather / scatter), same weight tensors (the a16w4 lane's StridedLayout views; their
underlying [E, N, K/2] / [E, N, K/32] memory is read as is), same sorted-order outputs, same reduce_grouped.

    x [T, H] bf16 --quant_rows--> fp8 + per-token scale
      --gemm(w13, gather, SiLU*up)--> bf16 [T*topk, I] (sorted)  --quant_rows--> fp8 + per-row scale
      --gemm(w2, gammas)--> bf16 [T*topk, H] (sorted)  --reduce_grouped--> [T, H]

The only new persistent memory is Wref (per-row max e8m0 exponent, [E, N] uint8 per weight; 0.75 MiB per layer), built
on the first call for a weight and cached by its data pointer.
Knobs: RADIANCE_MOE_W4A8_MIN_TOKENS (calls with fewer tokens stay on a16w4), RADIANCE_MOE_W4A8_CFG13_BM<bm> /
RADIANCE_MOE_W4A8_CFG2_BM<bm> (tile config per routing block_m, see the DISPATCH table in radiance_moe_w4a8.hip).
"""
import os

import torch

import radiance_moe_w4a8_hip as _hip

# Below ~1,025 tokens AITER's routing picks block_m <= 32 and the a16w4 lane wins (the expert GEMMs are weight-stream
# bound there, and at decode / MTP-verify sizes they run at the weight-bandwidth floor); from block_m 64 up W4A8 wins
# (tests/test_w4a8.py: 1.38x at 1,056 tokens, 2.06x at 2,224, 1.85x at 3,991, per layer incl. quant + reduce).
MIN_TOKENS = int(os.environ.get("RADIANCE_MOE_W4A8_MIN_TOKENS", "1025"))
# tile config per routing block_m, separately for w13 (K=2048) and w2 (K=512); ids index DISPATCH in the .hip
_CFG13 = {128: 7, 64: 6}
_CFG2 = {128: 5, 64: 4}
CFG13 = {bm: int(os.environ.get(f"RADIANCE_MOE_W4A8_CFG13_BM{bm}", c)) for bm, c in _CFG13.items()}
CFG2 = {bm: int(os.environ.get(f"RADIANCE_MOE_W4A8_CFG2_BM{bm}", c)) for bm, c in _CFG2.items()}
CFG = CFG13
# TEST-ONLY numerics arm: RADIANCE_MOE_W4A8_FORCE_ALL=1 also routes block_m 16 / 32 calls (short prefills) to W4A8, so
# short-prompt benchmarks (gsm8k) exercise its numerics. Slower than a16w4 there; never for production.
FORCE_ALL = os.environ.get("RADIANCE_MOE_W4A8_FORCE_ALL", "0") == "1"
if FORCE_ALL:
    CFG13.update({32: 0, 16: 1})
    CFG2.update({32: 1, 16: 1})
_REF = {}


def nk_view(t):
    """The a16w4 lane's [E, K/x, N] view with stride(-2) == 1 -> its contiguous [E, N, K/x] memory."""
    u = t.transpose(-2, -1)
    if not u.is_contiguous():
        raise RuntimeError(f"radiance_moe_w4a8: weight view {tuple(t.shape)} stride {t.stride()} is not the "
                           "StridedLayout transpose of a contiguous [E, N, K] tensor")
    return u


def wref(ws_nk):
    key = (ws_nk.data_ptr(), tuple(ws_nk.shape))
    r = _REF.get(key)
    if r is None:
        r = ws_nk.amax(-1).contiguous()
        _REF[key] = r
    return r


def _stream():
    return torch.cuda.current_stream().cuda_stream


def quant_rows(x):
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
        raise RuntimeError(f"radiance_moe_w4a8.quant_rows: need row-contiguous bf16, got {x.dtype} {x.stride()}")
    rows, K = x.shape
    q = torch.empty((rows, K), dtype=torch.uint8, device=x.device)
    s = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _hip.quant_rows(x.data_ptr(), x.stride(0), q.data_ptr(), s.data_ptr(), rows, K, _stream())
    return q, s


def gemm(aq, a_s, w_nk, ws_nk, w_ref, rd, gather=None, gammas=None, swiglu=False, cfg=None):
    E, N, Kh = w_nk.shape
    K = 2 * Kh
    if aq.shape[1] != K:
        raise RuntimeError(f"radiance_moe_w4a8.gemm: activation K {aq.shape[1]} != weight K {K}")
    Mtot = gather.shape[0] if gather is not None else aq.shape[0]
    if gather is not None and gather.dtype != torch.int32:
        gather = gather.to(torch.int32)   # AITER routing hands out uint16 indices while T * topk < 65536
    bm = rd.block_m
    ed = rd.expt_data
    for name, t in (("hist", ed.hist), ("token_offs_raw", ed.token_offs_raw), ("block_pid_map", ed.block_pid_map)) + (
            (("gather", gather),) if gather is not None else ()):
        if t.dtype != torch.int32 or not t.is_contiguous():
            raise RuntimeError(f"radiance_moe_w4a8.gemm: {name} must be contiguous int32, got {t.dtype}")
    if gammas is not None and (gammas.dtype != torch.float32 or not gammas.is_contiguous()):
        gammas = gammas.float().contiguous()   # routing's gate_scal is in the logits dtype (bf16)
    grid_m = rd.n_blocks(Mtot, bm)
    y = torch.empty((Mtot, N // 2 if swiglu else N), dtype=torch.bfloat16, device=aq.device)
    _hip.gemm(aq.data_ptr(), a_s.data_ptr(), w_nk.data_ptr(), ws_nk.data_ptr(), w_ref.data_ptr(),
              gather.data_ptr() if gather is not None else 0, ed.hist.data_ptr(), ed.token_offs_raw.data_ptr(),
              ed.block_pid_map.data_ptr(), grid_m, gammas.data_ptr() if gammas is not None else 0, y.data_ptr(),
              N, K, rd.n_expts_act, bm, swiglu, (CFG13 if swiglu else CFG2)[bm] if cfg is None else cfg, _stream())
    return y


def supported(hidden_states, routing_data, w1_bias, w2_bias, swiglu_limit, apply_router_weight_on_input):
    # never inside a CUDA-graph capture: captured (decode / MTP-verify) sizes stay on a16w4
    return (hidden_states.dim() == 2 and hidden_states.shape[0] >= MIN_TOKENS and routing_data is not None
            and routing_data.block_m in CFG13 and not torch.cuda.is_current_stream_capturing()
            and w1_bias is None and w2_bias is None and swiglu_limit is None and not apply_router_weight_on_input
            and hidden_states.dtype == torch.bfloat16)


def moe_forward(hidden_states, w1, w2, w1_scale, w2_scale, routing_data, gather_idx, scatter_idx, gammas,
                cfg13=None, cfg2=None):
    """w1/w2/w*_scale: the a16w4 lane's raw torch views ([E, K/2, N] / [E, K/32, N], stride(-2) == 1)."""
    from aiter.ops.triton.moe.reduce import reduce_grouped
    w1n, s1n, w2n, s2n = nk_view(w1), nk_view(w1_scale), nk_view(w2), nk_view(w2_scale)
    xq, xs = quant_rows(hidden_states)
    inter = gemm(xq, xs, w1n, s1n, wref(s1n), routing_data, gather=gather_idx, swiglu=True, cfg=cfg13)
    iq, isc = quant_rows(inter)
    y = gemm(iq, isc, w2n, s2n, wref(s2n), routing_data, gammas=gammas, cfg=cfg2)
    topk = routing_data.n_expts_act
    out = torch.empty((hidden_states.shape[0], w2n.shape[1]), dtype=torch.bfloat16, device=hidden_states.device)
    return reduce_grouped(y[None], scatter_idx.view(-1, topk), out)
