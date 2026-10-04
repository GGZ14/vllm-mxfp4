"""Draft-only compressed lm_head for the MoE lane's MTP drafter (moe-drafthead/, MOE-GFX1201.md).

RADIANCE_MOE_DRAFT_HEAD = off (default) | fp8 | int4 | int2

On Qwen3.5-35B-A3B (hidden 2048, vocab 248,320) the untied bf16 lm_head is 1.017 GB, and an MTP-4 decode step
reads it once for the verify pass plus once per draft pass (the image's dynamic-draft controller averages ~3.3
passes). At 634 GB/s each read is 1.6 ms: a third of decode kernel time (MOE-GFX1201.md, decode attribution). The verify
pass needs the exact head; the drafter only needs a good argmax (draft_sample_method is "greedy", drafts are
one-hot in rejection sampling) plus a top-1 softmax confidence for radiance_draft's depth gate. So the drafter's
LogitsProcessor gets its own compressed copy of the head and the target's LogitsProcessor (a separate instance)
keeps the bf16 one: accepted tokens are exactly what the target would emit; a worse draft only costs acceptance.

  fp8   e4m3 weights with one fp32 scale per vocabulary row (max -> 448), bf16 activations (no activation
        quantization). Triton weight-only GEMV: the e4m3 byte is placed into the fp16 exponent/mantissa
        ((b & 0x80) << 8 | (b & 0x7F) << 7 == value * 2^-8, normals and subnormals alike), so the decode is two
        integer ops and exact; x is bf16 -> fp16; the 2^8 is folded into the row scale. 0.508 GB.
        ~835-880 us at M = 1..16 (vs 1.60-1.68 ms bf16), 600 GB/s.
  int4  symmetric uint4b8 per (row, group of 128), bf16 group scales, vLLM's RDNA W4A16 layout (ExLlama-shuffled
        int32 viewed as int8 [N, K/2]). M <= 5: vLLM's HIP skinny kernel wvSplitK_int4_g (413 us at M=1, 527 at
        M=5); M > 5: vLLM's triton_w4a16_skinny_fmt_gemm (~1.05 ms). 0.270 GB.
  int2  the dense lane's radiance_drafthead (as shipped in the image): int2 g128 asymmetric coarse pass + exact
        bf16 rerank of the top-32 block maxima against the shared head (~475 us flat to M=16). 0.143 GB.
        Coarse values everywhere but the reranked candidates, so the depth gate's softmax sees coarse logits.

Armed from Qwen3_5MTP.load_weights (patch_moe_drafthead.py), i.e. while the drafter's own copy of lm_head (loaded
from the checkpoint's lm_head.weight, before vLLM swaps in the target's identical tensor) is populated and BEFORE
vLLM sizes the KV cache, so the compressed copy's memory comes out of the KV pool visibly ("Available KV cache
memory"). No reference to the bf16 tensor is kept, so the drafter's duplicate still frees when vLLM shares the
target head. The first real call checks a 512-row sample of the compressed copy against the lm_head it is handed
(the shared target head) and raises if they disagree -- the silent failure mode is a garbage draft head, which
leaves output correct and only collapses acceptance to ~1.0.
"""
import os
import sys
import types

import torch

MODE = os.environ.get("RADIANCE_MOE_DRAFT_HEAD", "off").strip().lower()
GROUP4 = 128
_TOL = {"fp8": 0.08, "int4": 0.30, "int2": 0.80}   # max rel. Frobenius error of the 512-row sample vs bf16
_SAMPLE = 512


def _log(msg):
    sys.stderr.write(f"[radiance.drafthead] {msg}\n")
    sys.stderr.flush()


try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _fp8w_head(X, W, S, Y, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
        """y[M, N] = (x[M, K] @ W8[N, K]^T) * S[N]; W8 raw e4m3 bytes, S per row (incl. the 2^8 decode factor)."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_m[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0).to(tl.float16)
            b = tl.load(W + offs_n[None, :] * K + offs_k[:, None], mask=mask_n[None, :], other=0).to(tl.uint16)
            w = (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True)
            acc += tl.dot(x, w)
        s = tl.load(S + offs_n, mask=mask_n, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], (acc * s[None, :]).to(tl.bfloat16),
                 mask=mask_m[:, None] & mask_n[None, :])


# measured optima on gfx1201 for [248320, 2048] (microbenchmark of this head, MOE-GFX1201.md): M=1 -> 839 us, M=16 -> 884
_FP8_CFG_SMALL = dict(BLOCK_N=32, BLOCK_K=512, num_warps=2, num_stages=1)   # M <= 8
_FP8_CFG_16 = dict(BLOCK_N=64, BLOCK_K=512, num_warps=4, num_stages=1)      # 8 < M <= 16


def _fp8_matmul(x, w8, s, y):
    m, k = x.shape
    n = w8.shape[0]
    cfg = _FP8_CFG_SMALL if m <= 8 else _FP8_CFG_16
    _fp8w_head[(triton.cdiv(n, cfg["BLOCK_N"]),)](x, w8, s, y, m, n, k, BLOCK_M=16, **cfg)


def _finish(self, y, hidden_states, embedding_bias):
    if embedding_bias is not None:
        y = y + embedding_bias
    if self.head_dtype is not None and self.head_dtype != y.dtype:
        y = y.to(self.head_dtype)
    return y.reshape(*hidden_states.shape[:-1], -1)


def _apply_head_fp8(self, lm_head, hidden_states, embedding_bias):
    if not self._rmdh_checked:
        _first_call_check(self, lm_head)
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    m = x.shape[0]
    y = torch.empty(m, self._rmdh_n, dtype=torch.bfloat16, device=x.device)
    for i in range(0, m, 16):                       # drafter M = batch <= max-num-seqs (16 here): one launch
        j = min(m, i + 16)
        _fp8_matmul(x[i:j], self._rmdh_w, self._rmdh_s, y[i:j])
    return _finish(self, y, hidden_states, embedding_bias)


def _apply_head_int4(self, lm_head, hidden_states, embedding_bias):
    if not self._rmdh_checked:
        _first_call_check(self, lm_head)
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    m = x.shape[0]
    if m <= 5:
        y = self._rmdh_ops.wvSplitK_int4_g(self._rmdh_w, x, self._rmdh_s, self._rmdh_cu, GROUP4, None, None)
    else:
        y = self._rmdh_tri(x, self._rmdh_w.view(torch.int32), self._rmdh_s, GROUP4)
    return _finish(self, y, hidden_states, embedding_bias)


def _apply_head_int2(self, lm_head, hidden_states, embedding_bias):
    if not self._rmdh_checked:
        _first_call_check(self, lm_head)
    return self._rmdh_int2(lm_head, hidden_states, embedding_bias)


# ------------------------------------------------------------------------------------------------ quantizers
def _quant_fp8(w):
    n, k = w.shape
    q = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    s = torch.empty(n, dtype=torch.float32, device=w.device)
    for i in range(0, n, 8192):
        wf = w[i:i + 8192].float()
        sc = (wf.abs().amax(dim=1) / 448.0).clamp(min=1e-30)
        q[i:i + 8192] = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        s[i:i + 8192] = sc * 256.0                   # fold the fp16-decode 2^-8 back in
        del wf, sc
    return q, s


def _dequant_fp8(q, s, rows):
    return q[rows].view(torch.float8_e4m3fn).float() * (s[rows] / 256.0)[:, None]


def _quant_int4(w):
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import pack_int4_exllama_shuffle
    n, k = w.shape
    packed = torch.empty(n, k // 8, dtype=torch.int32, device=w.device)
    scales = torch.empty(n, k // GROUP4, dtype=torch.bfloat16, device=w.device)
    for i in range(0, n, 8192):
        wf = w[i:i + 8192].float().reshape(-1, k // GROUP4, GROUP4)
        sc = (wf.abs().amax(dim=2) / 7.0).clamp(min=1e-30)
        sc = sc.to(torch.bfloat16).float()           # quantize against the scale the kernel will use
        q = (torch.round(wf / sc[:, :, None]) + 8).clamp(0, 15).to(torch.int32).reshape(-1, k)
        packed[i:i + 8192] = pack_int4_exllama_shuffle(q)
        scales[i:i + 8192] = sc.to(torch.bfloat16)
        del wf, sc, q
    return packed.view(torch.int8), scales


def _dequant_int4(q8, s, rows):
    p = q8.view(torch.int32)[rows]                   # [R, K/8], interleave [0,2,4,6,1,3,5,7]
    r, k8 = p.shape
    order = [0, 2, 4, 6, 1, 3, 5, 7]                 # nibble j holds K-offset order[j]
    out = torch.empty(r, k8, 8, dtype=torch.float32, device=p.device)
    for j, kk in enumerate(order):
        out[:, :, kk] = ((p >> (4 * j)) & 15).float() - 8.0
    out = out.reshape(r, k8 * 8)
    return (out.reshape(r, -1, GROUP4) * s[rows].float()[:, :, None]).reshape(r, -1)


def _dequant_int2(lp, rows):
    """Reconstruct rows of radiance_drafthead's int2 packing: byte j carries k = j, K/4+j, K/2+j, 3K/4+j at
    shifts 0,2,4,6; the kernel computes (1 + v/4) * S - ZS per (row, group)."""
    b = lp._radiance_wq[rows].to(torch.int32)          # [R, K/4]
    r, q4 = b.shape
    k = q4 * 4
    g = 128
    v = torch.cat([((b >> (2 * q)) & 3).float() for q in range(4)], dim=1)   # [R, K], plane q = K-quarter q
    S = lp._radiance_scale[rows].float()            # [R, K/g] (quarter-major group index gi = q*NG + g)
    Z = lp._radiance_zs[rows].float()
    return ((1.0 + v / 4.0).reshape(r, k // g, g) * S[:, :, None] - Z[:, :, None]).reshape(r, k)


def _first_call_check(lp, lm_head):
    """Compare a fixed 512-row sample of the compressed copy against the head this call is handed (the target's
    shared bf16 lm_head). Raise on mismatch: never serve a draft head built from the wrong tensor."""
    lp._rmdh_checked = True
    w = getattr(lm_head, "weight", None)
    if w is None or tuple(w.shape) != (lp._rmdh_n, lp._rmdh_k) or w.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"[radiance.drafthead] {lp._rmdh_mode}: lm_head at first use is "
                           f"{None if w is None else (tuple(w.shape), w.dtype)}, not the bf16 "
                           f"({lp._rmdh_n}, {lp._rmdh_k}) head the draft copy was built from")
    rows = lp._rmdh_rows
    ref = w[rows].float()
    deq = lp._rmdh_deq(rows)
    rel = float((deq - ref).norm() / ref.norm().clamp(min=1e-30))
    tol = _TOL[lp._rmdh_mode]
    if not rel < tol:
        raise RuntimeError(f"[radiance.drafthead] {lp._rmdh_mode} draft head does not match the lm_head it is "
                           f"used with: sample rel. error {rel:.4f} >= {tol} (built-time {lp._rmdh_rel0:.4f})")
    _log(f"{lp._rmdh_mode} draft head verified against the shared lm_head at first use: sample rel. error "
         f"{rel:.4f} (tol {tol})")


# ------------------------------------------------------------------------------------------------ arming
def arm(mtp, lp_attr="logits_processor"):
    """Build the compressed draft head from the drafter's populated lm_head and bind it to the drafter's own
    LogitsProcessor. Called at the end of Qwen3_5MTP.load_weights. Fails loudly (raises) on anything off."""
    mode = MODE
    if mode in ("", "off", "0"):
        return
    if mode not in _TOL:
        raise RuntimeError(f"[radiance.drafthead] RADIANCE_MOE_DRAFT_HEAD={mode!r}: expected off, fp8, int4 or int2")
    lm_head = getattr(mtp, "lm_head", None)
    lp = getattr(mtp, lp_attr, None)
    w = getattr(lm_head, "weight", None)
    if lp is None or w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"[radiance.drafthead] {mode}: drafter lm_head unusable "
                           f"({None if w is None else (tuple(w.shape), w.dtype)})")
    if getattr(lm_head, "tp_size", 1) != 1:
        raise RuntimeError("[radiance.drafthead] only TP=1 is supported (the head is not vocab-sharded here)")
    wd = w.data
    if float(wd[:: max(1, wd.shape[0] // 4096)].abs().amax()) == 0.0:
        raise RuntimeError("[radiance.drafthead] drafter lm_head is all-zero at load_weights: nothing to compress")
    n, k = wd.shape
    if k % GROUP4:
        raise RuntimeError(f"[radiance.drafthead] hidden size {k} not a multiple of {GROUP4}")
    gen = torch.Generator(device="cpu").manual_seed(1234)
    rows = torch.randperm(n, generator=gen)[:_SAMPLE].sort().values.to(wd.device)
    ref = wd[rows].float()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    if mode == "fp8":
        if triton is None:
            raise RuntimeError("[radiance.drafthead] fp8 needs triton")
        q, s = _quant_fp8(wd)
        lp._rmdh_w, lp._rmdh_s = q, s
        lp._rmdh_deq = lambda r, _q=q, _s=s: _dequant_fp8(_q, _s, r)
        nbytes = q.numel() + s.numel() * 4
        fn = _apply_head_fp8
        desc = "fp8 e4m3 per-row scale, bf16 activations, Triton weight-only GEMV"
    elif mode == "int4":
        import vllm._custom_ops as ops
        from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import \
            triton_w4a16_skinny_fmt_gemm
        from vllm.utils.platform_utils import num_compute_units
        if not hasattr(torch.ops, "_rocm_C") or not hasattr(torch.ops._rocm_C, "wvSplitK_int4_g"):
            raise RuntimeError("[radiance.drafthead] int4 needs vLLM's _rocm_C.wvSplitK_int4_g")
        q, s = _quant_int4(wd)
        lp._rmdh_w, lp._rmdh_s = q, s
        lp._rmdh_ops, lp._rmdh_tri, lp._rmdh_cu = ops, triton_w4a16_skinny_fmt_gemm, num_compute_units()
        lp._rmdh_deq = lambda r, _q=q, _s=s: _dequant_int4(_q, _s, r)
        nbytes = q.numel() + s.numel() * 2
        fn = _apply_head_int4
        desc = f"int4 sym g{GROUP4} (uint4b8), wvSplitK_int4_g M<=5 / triton_w4a16_skinny M>5"
    else:  # int2: the dense lane's module, as shipped in the image
        import radiance_drafthead as rd
        status = rd._quantize_head_now(lp, types.SimpleNamespace(weight=wd))
        int2_apply = lp._apply_head                  # bound _apply_head_int2 (coarse + exact rerank)
        lp._rmdh_int2 = int2_apply
        lp._rmdh_deq = lambda r, _lp=lp: _dequant_int2(_lp, r)
        nbytes = lp._radiance_wq.numel() + (lp._radiance_scale.numel() + lp._radiance_zs.numel()) * 2
        fn = _apply_head_int2
        desc = f"int2 g128 asym + exact top-{rd.RERANK} rerank (radiance_drafthead: {status})"
    deq = lp._rmdh_deq(rows)
    rel0 = float((deq - ref).norm() / ref.norm())
    del deq, ref
    if not rel0 < _TOL[mode]:
        raise RuntimeError(f"[radiance.drafthead] {mode}: build-time sample rel. error {rel0:.4f} >= {_TOL[mode]}")
    lp._rmdh_mode, lp._rmdh_n, lp._rmdh_k, lp._rmdh_rows, lp._rmdh_rel0 = mode, n, k, rows, rel0
    lp._rmdh_checked = False
    lp._apply_head = types.MethodType(fn, lp)
    # warm-up: compile the Triton variants now, not on the first request
    with torch.no_grad():
        for m in ((1, 9) if mode == "fp8" else (1, 6, 9, 16) if mode == "int4" else (1, 16)):
            xw = torch.zeros(m, k, dtype=wd.dtype, device=wd.device)
            if mode == "int2":
                lp._rmdh_int2(types.SimpleNamespace(weight=wd), xw, None)
            else:
                saved = lp._rmdh_checked
                lp._rmdh_checked = True
                lp._apply_head(lm_head, xw, None)
                lp._rmdh_checked = saved
    del wd, w
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    used = free0 - torch.cuda.mem_get_info()[0]
    _log(f"MoE draft head {mode} ON: lm_head ({n}, {k}) {lm_head.weight.dtype} -> {desc}; "
         f"{nbytes / 2**30:.3f} GiB extra (device free memory -{used / 2**30:.3f} GiB); "
         f"build-time sample rel. error {rel0:.4f}; verify pass keeps the bf16 head")
