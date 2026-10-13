#!/usr/bin/env python3
"""Correctness + speed of unified_attention's 2D vs 3D (split-KV) path for multi-token queries.

Run inside the Radiance image AFTER patch_attn_3d_multiq.py. RADIANCE_ATTN_3D_MAX_Q is read per
call, so the same process runs stock (=1: 2D for any q_len > 1) and patched (=16: 3D) back to back
on identical inputs. Shapes mirror Qwen3.5/3.6-35B-A3B under TRITON_ATTN on one R9700: 16 q heads,
2 kv heads, head_dim 256, fp8 per-tensor KV, attention block 2224 tokens, 16 KV segments.
Reference: fp32 causal attention over the dequantized cache.
"""
import json
import math
import os
import sys

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

DEV = "cuda"
HQ, HKV, D = 16, 2, 256
BS = 2224
SEGS, ROWS, THRESH = 16, 128, 64
FP8 = current_platform.fp8_dtype()
KVMODE = getattr(KVQuantMode, "FP8_PER_TENSOR", None)
SCALE = 1.0 / math.sqrt(D)


def boundary_seq_len(q_len, lo=6000, hi=7000):
    """A seq_len whose last 3D segment starts inside the query window (fully masked rows)."""
    for s in range(lo, hi):
        seg = math.ceil(s / (SEGS * 32)) * 32
        if 0 < s % seg < q_len:
            return s
    return None


def make_case(seq_lens, q_len, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = len(seq_lens)
    nb = [math.ceil(s / BS) for s in seq_lens]
    total = sum(nb) + 2
    k = (torch.randn(total, BS, HKV, D, device=DEV, generator=g) * 0.7).to(FP8)
    v = (torch.randn(total, BS, HKV, D, device=DEV, generator=g) * 0.7).to(FP8)
    perm = torch.randperm(total - 1, device=DEV, generator=g) + 1
    bt = torch.zeros(B, max(nb), dtype=torch.int32, device=DEV)
    i = 0
    for b in range(B):
        bt[b, : nb[b]] = perm[i : i + nb[b]].to(torch.int32)
        i += nb[b]
    q = torch.randn(B * q_len, HQ, D, device=DEV, generator=g).to(torch.bfloat16)
    cu = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=DEV)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return q, k, v, bt, cu, sl


def reference(q, k, v, bt, seq_lens, q_len):
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    grp = HQ // HKV
    for b, s in enumerate(seq_lens):
        blocks = bt[b, : math.ceil(s / BS)].long()
        kk = k[blocks].float().reshape(-1, HKV, D)[:s]  # [s, HKV, D]
        vv = v[blocks].float().reshape(-1, HKV, D)[:s]
        qq = q[b * q_len : (b + 1) * q_len].float()  # [q_len, HQ, D]
        kh = kk.repeat_interleave(grp, dim=1)  # [s, HQ, D]
        vh = vv.repeat_interleave(grp, dim=1)
        sc = torch.einsum("qhd,khd->hqk", qq, kh) * SCALE
        pos = torch.arange(s - q_len, s, device=DEV)[:, None]
        mask = torch.arange(s, device=DEV)[None, :] > pos  # [q_len, s]
        sc = sc.masked_fill(mask[None], float("-inf"))
        out[b * q_len : (b + 1) * q_len] = torch.einsum(
            "hqk,khd->qhd", torch.softmax(sc, dim=-1), vh
        )
    return out


def run(q, k, v, bt, cu, sl, seq_lens, q_len, bufs, max_q):
    os.environ["RADIANCE_ATTN_3D_MAX_Q"] = str(max_q)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=DEV)
    B = len(seq_lens)
    ones = torch.ones(1, dtype=torch.float32, device=DEV)
    kw = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=sl,
        max_seqlen_k=max(seq_lens), softmax_scale=SCALE, causal=True, window_size=(-1, -1),
        block_table=bt, softcap=0, q_descale=None,
        k_descale=ones.expand(B, HKV), v_descale=ones.expand(B, HKV),
        seq_threshold_3D=THRESH, num_par_softmax_segments=SEGS,
        softmax_segm_output=bufs[0], softmax_segm_max=bufs[1], softmax_segm_expsum=bufs[2],
    )
    if KVMODE is not None:
        kw["kv_quant_mode"] = KVMODE
    unified_attention(**kw)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        unified_attention(**kw)
    st.record()
    iters = 30
    for _ in range(iters):
        unified_attention(**kw)
    en.record()
    torch.cuda.synchronize()
    return out.float(), st.elapsed_time(en) * 1000 / iters


def main():
    print("KVQuantMode used:", KVMODE, "| fp8 dtype:", FP8, flush=True)
    bufs = (
        torch.empty(ROWS, HQ, SEGS, D, dtype=torch.float32, device=DEV),
        torch.empty(ROWS, HQ, SEGS, dtype=torch.float32, device=DEV),
        torch.empty(ROWS, HQ, SEGS, dtype=torch.float32, device=DEV),
    )
    cases = []
    for s in (830, 6144, 18293):
        for ql in (1, 2, 5, 9, 16):
            for B in (1, 4, 8):
                cases.append(([s] * B, ql))
    for ql in (5, 9, 16):
        bs = boundary_seq_len(ql)
        if bs:
            cases.append(([bs], ql))
            cases.append(([bs, 830, 18293, bs + 1], ql))
    cases.append(([18293] * 8, 17))  # above the 3D q_len cap: both paths must be 2D
    rows, worst = [], 0.0
    for n, (seq_lens, ql) in enumerate(cases):
        q, k, v, bt, cu, sl = make_case(seq_lens, ql, seed=n)
        ref = reference(q, k, v, bt, seq_lens, ql)
        o2, t2 = run(q, k, v, bt, cu, sl, seq_lens, ql, bufs, max_q=1)
        o3, t3 = run(q, k, v, bt, cu, sl, seq_lens, ql, bufs, max_q=16)
        e2 = (o2 - ref).abs().max().item()
        e3 = (o3 - ref).abs().max().item()
        nan = bool(torch.isnan(o3).any() or torch.isnan(o2).any())
        worst = max(worst, e3)
        r = dict(seq_lens=seq_lens if len(set(seq_lens)) > 1 else f"{seq_lens[0]}x{len(seq_lens)}",
                 q_len=ql, err_stock=round(e2, 5), err_patched=round(e3, 5), nan=nan,
                 us_stock=round(t2, 1), us_patched=round(t3, 1), speedup=round(t2 / t3, 2))
        rows.append(r)
        print(json.dumps(r), flush=True)
    ok = all(not r["nan"] for r in rows) and worst < 0.05
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "worst_err_patched": worst}), flush=True)
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/out/attn3d.json", "w"), indent=1)


if __name__ == "__main__":
    main()
