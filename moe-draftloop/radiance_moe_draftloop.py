"""radiance-gaps: less host time in the MTP drafter's serial loop (MoE lane, vLLM 0.27.1). Two independent knobs.

RADIANCE_MOE_DRAFT_GRAPH=1
  vLLM 0.27.1 runs the drafter PIECEWISE only: per loop pass two graph pieces plus an eager attention segment (KV write,
  q fp8 quant, AITER unified attention, reduce_segments), each re-entered from Python. On one stream the GPU idles for
  that host time (~0.25 ms per pass on the R9700, radiance-gaps Phase 1), because the dynamic-draft gate drains the GPU
  at the end of every pass. This captures each loop pass's whole drafter forward (the compiled pieces run in NONE mode
  inside the capture, plus the eager attention) as ONE graph per (real batch, padded batch) and replays it.
  Capture safety: loop passes are uniform (1 token per request), so every host-side decision in the forward (attention
  plan, AITER 3D config with max_seqlen_k = max_model_len, MoE routing shapes) depends only on the key. The graph reads
  query_start_loc (proposer arange), block_table (runner tensor), slot_mapping (proposer buffer), positions /
  input_ids / hidden_states (proposer buffers) at fixed addresses; seq_lens is a per-step tensor, so it is copied
  into a persistent staging buffer before every replay and the captured metadata points at that buffer. Every replay
  first compares the live metadata's tensor addresses and scalars with the captured ones and falls back to the normal
  piecewise path on any difference. A key is replayed only from its third use on: use 1 runs the normal path, use 2
  runs the same forward in NONE mode (so every kernel is compiled outside the capture), use 3 captures and replays.
  A capture error disables the knob for the process and that pass runs the normal path. Draft quality cannot affect
  outputs (the target verifies every token); a stale input would show up as lower acceptance.

RADIANCE_MOE_DRAFT_OVERLAP=1
  The dynamic-draft gate (radiance_draft.greedy_sample) blocks on an 8-byte D2H right after each pass's draft head
  and only then lets the loop prepare the next pass (positions / slot mapping, attention metadata, input copies).
  With this knob the D2H goes to pinned memory without blocking, and the decision is resolved after that preparation,
  right before the next forward (or in the draft postprocess). Which passes run is unchanged: the stop check still
  precedes every forward; a pass the gate stops only wastes its preparation.
"""
import contextlib
import os
import sys

import torch

GRAPH = os.environ.get("RADIANCE_MOE_DRAFT_GRAPH", "0") == "1"
OVERLAP = os.environ.get("RADIANCE_MOE_DRAFT_OVERLAP", "0") == "1"
_STATS_EVERY = int(os.environ.get("RADIANCE_MOE_DRAFT_GRAPH_STATS", "0") or 0)

_graphs = {}          # (B, Bp) -> state dict
_disabled = [False]
_pool = [None]
_stage = [None]       # persistent int32 seq_lens staging buffer
_stats = {"replay": 0, "fallback": 0, "capture": 0, "normal": 0}
_said = set()


def _log(msg):
    sys.stderr.write(f"[radiance.draftloop] {msg}\n")
    sys.stderr.flush()


def _say_once(key, msg):
    if key not in _said:
        _said.add(key)
        _log(msg)


def describe():
    return f"draft graph {'ON' if GRAPH else 'off'}, gate overlap {'ON' if OVERLAP else 'off'}"


# ---------------------------------------------------------------- overlap -----------------------------------------
def before_forward(prop):
    """Loop hook right before each loop pass's forward: resolve the previous slot's deferred gate decision.
    Returns True when the controller says stop (the caller breaks before the forward)."""
    r = getattr(prop, "_radiance_resolve", None)
    if r is not None:
        r()
    return bool(getattr(prop, "_radiance_stop", False))


def resolve_pending(prop):
    r = getattr(prop, "_radiance_resolve", None)
    if r is not None:
        r()


def defer_d2h(prop, packed, decide):
    """greedy_sample hook: start the packed [2, B] D2H into pinned memory and defer decide(arr) until resolved."""
    resolve_pending(prop)                       # keep decisions in slot order whatever the caller does
    B = packed.shape[1]
    pin = getattr(prop, "_radiance_pin", None)
    if pin is None or pin.numel() < 2 * B:
        pin = prop._radiance_pin = torch.empty(2 * max(B, 16), dtype=torch.float32, pin_memory=True)
        prop._radiance_pin_ev = torch.cuda.Event()
    dst = pin[:2 * B].view(2, B)                # contiguous, so the copy stays asynchronous
    dst.copy_(packed, non_blocking=True)
    ev = prop._radiance_pin_ev
    ev.record()

    def _resolve(dst=dst, ev=ev, packed=packed):
        prop._radiance_resolve = None
        ev.synchronize()
        decide(dst.numpy().copy())

    prop._radiance_resolve = _resolve


# ---------------------------------------------------------------- graph -------------------------------------------
_SCALARS = ("num_actual_tokens", "max_query_len", "causal", "use_cascade", "common_prefix_len")


def _tensor_sig(plam, skip_ptr):
    """(layer, field) -> (data_ptr, shape, stride) for every tensor field of the per-layer metadata except the staged
    seq_lens, plus the scalars the attention launch depends on."""
    sig = {}
    for lname, md in plam.items():
        for k, v in vars(md).items():
            if isinstance(v, torch.Tensor):
                if k == "seq_lens" or v.data_ptr() == skip_ptr:
                    continue
                sig[(lname, k)] = (v.data_ptr(), tuple(v.shape), tuple(v.stride()))
            elif k in _SCALARS:
                sig[(lname, k)] = v
    return sig


def _seq_tensor(plam):
    for md in plam.values():
        s = getattr(md, "seq_lens", None)
        if isinstance(s, torch.Tensor):
            return s
    return None


def _maybe_stats():
    if _STATS_EVERY and (_stats["replay"] + _stats["fallback"] + _stats["normal"]) % _STATS_EVERY == 0:
        _log(f"stats {_stats} graphs {len(_graphs)}")


@contextlib.contextmanager
def forward_ctx(prop, plam, cam, B, Bp, dp, mode):
    """Context for one drafter loop pass. Yields fn(model_kwargs) -> model output."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    def normal(rt_mode):
        return set_forward_context(plam, prop.vllm_config, num_tokens=Bp, num_tokens_across_dp=dp,
                                   cudagraph_runtime_mode=rt_mode, slot_mapping=prop._get_slot_mapping(Bp))

    usable = (GRAPH and not _disabled[0] and mode == CUDAGraphMode.PIECEWISE and dp is None
              and not torch.cuda.is_current_stream_capturing() and cam.max_query_len == 1
              and cam.num_actual_tokens == B)
    st = None
    if usable:
        st = _graphs.setdefault((B, Bp), {"uses": 0})
        st["uses"] += 1
    if st is None or st["uses"] == 1 or (st["uses"] > 3 and "graph" not in st):
        _stats["normal"] += 1
        with normal(mode):
            yield lambda kw: prop.model(**kw)
        return
    if st["uses"] == 2:                          # second use: same forward in NONE mode, outside any capture
        _stats["normal"] += 1
        with normal(CUDAGraphMode.NONE):
            yield lambda kw: prop.model(**kw)
        return
    if "graph" in st:                            # replay
        live = _seq_tensor(plam)
        if live is None or _tensor_sig(plam, -1) != st["sig"]:
            _stats["fallback"] += 1
            _say_once(("fb", B, Bp), f"B={B}/{Bp}: live metadata differs from the captured one -> piecewise path "
                                     f"(logged once per key)")
            with normal(mode):
                yield lambda kw: prop.model(**kw)
            return
        stage = _stage[0]

        def replay(kw):
            stage[:B].copy_(live[:B])
            st["graph"].replay()
            _stats["replay"] += 1
            _maybe_stats()
            return st["out"]
        yield replay
        return

    # third use: capture this pass's forward, then replay it for this pass's result
    def run_normal(kw):
        with normal(mode):
            return prop.model(**kw)

    def cap_and_replay(kw):
        try:
            if _stage[0] is None:
                n = max(64, int(prop.vllm_config.scheduler_config.max_num_seqs) + 1)
                _stage[0] = torch.zeros(n, dtype=torch.int32, device=cam.seq_lens.device)
            if _pool[0] is None:
                _pool[0] = torch.cuda.graph_pool_handle()
            stage = _stage[0]
            live = cam.seq_lens
            stage[:B].copy_(live[:B])
            saved = cam.seq_lens
            cam.seq_lens = stage[:B]
            try:
                _, plam_c = prop.build_per_group_and_layer_attn_metadata(cam)
            finally:
                cam.seq_lens = saved
            sq = _seq_tensor(plam_c)
            if sq is None or sq.data_ptr() != stage.data_ptr():
                raise RuntimeError("captured metadata does not read the staged seq_lens")
            sig = _tensor_sig(plam_c, stage.data_ptr())
            if _tensor_sig(plam, -1) != sig:
                raise RuntimeError("rebuilt metadata differs from the live one beyond seq_lens")
            g = torch.cuda.CUDAGraph()
            with set_forward_context(plam_c, prop.vllm_config, num_tokens=Bp, num_tokens_across_dp=dp,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                     slot_mapping=prop._get_slot_mapping(Bp)):
                with torch.cuda.graph(g, pool=_pool[0], capture_error_mode="thread_local"):
                    out = prop.model(**kw)
        except Exception as e:                   # never take the serve down: disable, run this pass normally
            _disabled[0] = True
            _log(f"capture failed for B={B}/{Bp} ({type(e).__name__}: {e}); draft graph disabled for this process")
            torch.cuda.synchronize()
            return run_normal(kw)
        st["graph"], st["out"], st["sig"] = g, out, sig
        _stats["capture"] += 1
        _log(f"captured drafter loop pass B={B} (padded {Bp}): {len(_graphs)} keys; stats {_stats}")
        stage[:B].copy_(live[:B])
        g.replay()
        return out
    yield cap_and_replay
