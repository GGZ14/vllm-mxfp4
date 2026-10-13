"""Padded CUDA-graph rows reuse a real row's experts in the MoE router (radiance-decode3/).

RADIANCE_MOE_PAD_ROUTE=1 (default off; read once at import)

vLLM pads every CUDA-graph replay to a captured size; with MTP-4 the sizes are multiples of 1 + 4 = 5. A single
stream therefore verifies 2..5 real rows as 5, and every MTP draft pass after the first (1 real row) also runs as 5.
The padded rows hold stale hidden states, and the MoE router gives each of them its own top-8 experts, so the expert
GEMMs read the weights of up to 8 extra experts per padded row and layer. On the MoE lane that costs ~0.66 ms per
decode step in the target's a16w4 experts and ~0.83 ms in the drafter's bf16 experts (radiance-decode3 Phase 1:
graphs-on vs no-graph trace at the same real row count), ~7% of the 21 ms step.

The fix: right after the router GEMV inside the MoE runner (the opaque moe_forward custom op, so it is captured into
the graphs), one tiny kernel copies row 0's router logits into every padded row. Padded rows then pick row 0's experts,
which are read anyway, so no extra weight bytes move. The router logits of real rows are never written (by
construction; self-checked at load), so real rows keep their experts; padded rows' outputs are discarded by vLLM as
before. End to end, greedy outputs still flip at near-ties (top-2 margin <= 0.25) vs off -- the same class as
production's own restart-to-restart flips -- since the expert kernels now see different per-expert row counts.

The real-row count must reach the captured kernel as a DEVICE value: a Python int would be baked into the graph at
capture. set_real(n_real, n_padded) writes it into a persistent int32 buffer on the model stream, from eager code,
right before every real forward that can be padded (patched in at the three call sites: the target forward in
gpu_model_runner.execute_model, the drafter's first pass and each loop pass in llm_base_proposer.propose). The buffer
holds a large value when the forward is not padded, so the kernel is a no-op. In eager forwards that are not padded
(prefill chunks) the kernel is not launched at all; during graph capture it is always launched.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_MOE_PAD_ROUTE", "0").strip() == "1"
_BIG = 1 << 30

_buf = None            # int32 [1] on the model device: real rows of the current forward (or _BIG)
_last = None           # last value written (stream-ordered, so an equal value needs no new write)
_padded = False        # the current eager forward is padded
_checked = False
_seen = {"target": 0, "drafter": 0}


def _log(msg):
    sys.stderr.write(f"[radiance.padroute] {msg}\n")
    sys.stderr.flush()


try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    # N is not specialized: one compiled variant (built by the eager self-check) serves every captured graph size, so
    # nothing compiles or loads a module while a stream is capturing.
    @triton.jit(do_not_specialize=["N"])
    def _padfix_kernel(L, NREAL, N, E, stride, BLOCK_E: tl.constexpr):
        """L[r, :] = L[0, :] for n_real <= r < N (one program; row 0 is never written)."""
        n = tl.load(NREAL)
        offs = tl.arange(0, BLOCK_E)
        m = offs < E
        v = tl.load(L + offs, mask=m)
        for r in range(n, N):
            tl.store(L + r * stride + offs, v, mask=m)


def _launch(logits, nreal_buf):
    n, e = logits.shape
    _padfix_kernel[(1,)](logits, nreal_buf, n, e, logits.stride(0), BLOCK_E=triton.next_power_of_2(e), num_warps=1)


def _self_check(device, dtype):
    global _checked, _buf, _last
    _checked = True
    if _buf is None:
        _buf = torch.full((1,), _BIG, dtype=torch.int32, device=device)
        _last = _BIG
    torch.manual_seed(0)
    x = torch.randn(5, 256, device=device, dtype=dtype)
    ref = x.clone()
    nb = torch.full((1,), 2, dtype=torch.int32, device=device)
    _launch(x, nb)
    ok_real = torch.equal(x[:2], ref[:2])
    ok_pad = all(torch.equal(x[r], ref[0]) for r in range(2, 5))
    nb.fill_(_BIG)
    y = ref.clone()
    _launch(y, nb)
    ok_noop = torch.equal(y, ref)
    torch.cuda.synchronize()
    if not (ok_real and ok_pad and ok_noop):
        raise RuntimeError(f"[radiance.padroute] self-check failed: real rows untouched {ok_real}, padded rows = row 0 "
                           f"{ok_pad}, no-op when unpadded {ok_noop}")
    _log("MoE pad-route ON: padded CUDA-graph rows take row 0's router logits (one tiny kernel after each router "
         "GEMV; real rows untouched); self-check OK")


def set_real(n_real, n_padded, who="target"):
    """Eager code, right before a forward: n_real real rows, padded to n_padded."""
    global _buf, _last, _padded
    n_real, n_padded = int(n_real), int(n_padded)
    _padded = 0 < n_real < n_padded
    v = n_real if _padded else _BIG
    if _buf is None:
        _buf = torch.full((1,), _BIG, dtype=torch.int32, device=torch.device("cuda", torch.cuda.current_device()))
        _last = _BIG
    if v != _last:
        _buf.fill_(v)
        _last = v
    if _padded and _seen[who] < 1:
        _seen[who] += 1
        _log(f"first padded {who} forward: {n_real} real rows padded to {n_padded}")


def fix(logits):
    """Called with the router logits [T, E] inside the MoE runner; returns them (modified in place when padded)."""
    global _buf
    if triton is None or logits.dim() != 2 or logits.shape[0] < 2 or logits.stride(-1) != 1:
        return logits
    capturing = torch.cuda.is_current_stream_capturing()
    if not _checked:
        # The first MoE forward is vLLM's eager profile run, before any graph capture: compile the kernel, check it,
        # and allocate the count buffer there, so captured graphs read a buffer that set_real() updates in place.
        if capturing:
            raise RuntimeError("[radiance.padroute] first use during graph capture: self-check did not run")
        _self_check(logits.device, logits.dtype)
    if not capturing and not _padded:
        return logits
    _launch(logits, _buf)
    return logits
