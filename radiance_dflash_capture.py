"""Capture the DFlash drafter's training inputs during real serving.

With RADIANCE_DFLASH_CAPTURE_DIR set, wraps DFlashProposer.propose and records, per request, the
target's concatenated aux hidden states (what the drafter's `fc` reads, [tokens, 5 x hidden]) and the
token at each position -- exactly the tensors the proposer receives, so a drafter trained on them
sees the served target's distribution (this checkpoint, this kernel path). Hidden states are stored
as e4m3 with one scale per token (the FP8 drafter quantizes its activations per token anyway).
Chunks are buffered per request and flushed as one file when the request leaves the batch.
Rank 0 only. Serve with prefix caching OFF, or cached prefixes never produce hidden states.
"""
import atexit
import os
import sys

import torch

DIR = os.environ.get("RADIANCE_DFLASH_CAPTURE_DIR", "")
MAX_TOKENS = int(os.environ.get("RADIANCE_DFLASH_CAPTURE_MAX_TOKENS", "4096"))
_buf: dict = {}
_stats = {"reqs": 0, "tokens": 0, "err": 0}


def _flush(req_id):
    parts = _buf.pop(req_id, None)
    if not parts:
        return
    pos = torch.cat([p[0] for p in parts]); tok = torch.cat([p[1] for p in parts])
    hid = torch.cat([p[2] for p in parts]); sc = torch.cat([p[3] for p in parts])
    order = torch.argsort(pos)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(req_id))[:80]
    torch.save({"positions": pos[order], "tokens": tok[order], "hid_e4m3": hid[order],
                "scale": sc[order]}, os.path.join(DIR, f"{safe}.pt"))
    _stats["reqs"] += 1; _stats["tokens"] += int(tok.numel())
    if _stats["reqs"] in (1, 10, 100, 500, 1000, 2000, 4000):
        sys.stderr.write(f"[radiance.dflash_capture] {_stats['reqs']} requests, {_stats['tokens']} tokens written to {DIR}\n")


def _flush_all():
    for r in list(_buf):
        _flush(r)


def _capture(self, input_batch, aux_hidden_states, num_rejected):
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() != 0 or not aux_hidden_states:
        return
    req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
    qsl = input_batch.query_start_loc_np
    n = int(input_batch.num_tokens)
    h = torch.cat([a[:n] for a in aux_hidden_states], dim=-1).float()          # [n, 5 x hidden], what fc reads
    scale = h.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    q = (h / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).cpu()
    sc = scale.to(torch.float16).cpu()
    tok = self.target_input_buffers.input_ids[:n].to(torch.int32).cpu()
    pos = self.target_input_buffers.positions[:n]
    pos = (pos[0] if pos.dim() == 2 else pos).to(torch.int32).cpu()
    rej = num_rejected[: len(req_ids)].to(torch.int32).cpu().tolist()          # rejected draft slots trail each segment
    live = set()
    for i, rid in enumerate(req_ids):
        a, b = int(qsl[i]), int(qsl[i + 1]) - int(rej[i])
        live.add(rid)
        if b > a:
            cur = _buf.setdefault(rid, [])
            have = sum(p[1].numel() for p in cur)
            if have < MAX_TOKENS:
                cur.append((pos[a:b], tok[a:b], q[a:b], sc[a:b]))
    for rid in [r for r in _buf if r not in live]:
        _flush(rid)


def install():
    if not DIR:
        return
    os.makedirs(DIR, exist_ok=True)
    # V2 model runner: DFlash2Speculator inherits propose() from DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    orig = DFlashSpeculator.propose

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, *a, **kw):
        if not kw.get("dummy_run", False) and not kw.get("is_profile", False):
            try:
                _capture(self, input_batch, aux_hidden_states, num_rejected)
            except Exception as e:  # noqa: BLE001
                _stats["err"] += 1
                if _stats["err"] <= 3:
                    sys.stderr.write(f"[radiance.dflash_capture] capture failed: {e!r}\n")
        return orig(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                    num_sampled, num_rejected, *a, **kw)

    DFlashSpeculator.propose = propose
    atexit.register(_flush_all)
    sys.stderr.write(f"[radiance.dflash_capture] ON -> {DIR} (V2 DFlash speculator; e4m3 + per-token scale, max {MAX_TOKENS} tokens/request)\n")


install()
