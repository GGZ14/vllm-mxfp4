"""Adaptive prefill chunk budget for the MoE lane (radiance-adaptive-chunk/).

RADIANCE_ADAPTIVE_CHUNK=<big budget> | 0 (default 0 = off; read once at import)
RADIANCE_ADAPTIVE_CHUNK_CAP=<cap>          (default 4096)
RADIANCE_ADAPTIVE_CHUNK_SYNC=1 | 0         (default 0; see "Async scheduling" below)
RADIANCE_ADAPTIVE_CHUNK_STATS_S=<seconds>  (default 60; periodic stats line while busy, 0 = only on switches)

Why: in align-mode prefix caching the attention block is 2,160 tokens and vLLM floors every non-final prefill chunk to
whole blocks (Scheduler._mamba_block_aligned_split), so --max-num-batched-tokens 4096 prefills a long prompt in
2,160-token steps. Each step streams nearly all expert weights, so 4,320-token steps (2 blocks) are ~6% faster on an idle
server (radiance-prefill2). But a plain 4,320 budget lets the long prefill take the whole budget, and a short request
that arrives meanwhile waits for the rest of the prefill (2.6 s instead of 0.5 s for a 32k prompt, chunk-ab).

Policy, decided at the top of every Scheduler.schedule() from scheduler state only: the step's token budget is the full
max_num_scheduled_tokens (== RADIANCE_ADAPTIVE_CHUNK, checked) only when exactly one request is in the scheduler
(running + waiting + skipped_waiting, no paused streaming session) and that request still has more than CAP tokens to
compute. A lone decoding request (1 + K tokens left with MTP-K) or a lone short prompt does not qualify (the budget would
not matter). In every other case the budget is min(full, CAP), which is exactly today's 4096 behaviour: the long prefill's
chunk floors to one 2,160 block and the rest of the budget goes to the other requests. vLLM still applies its own rules
on top (block-aligned split with the full max_num_scheduled_tokens as the alignment limit, max_model_len, the async
max_tokens guard, MTP spec tokens), so chunk boundaries stay on block multiples and prefix-cache states land where they
do today; only the number of blocks per solo step changes.

Async scheduling (RADIANCE_MOE_ASYNC=1): vLLM runs a 2-deep batch queue, so schedule() for step N+1 runs while step N
executes and sees N's tokens as already computed (request.num_computed_tokens advances at schedule time); this module
uses the same counters, so "tokens left" is what is still unscheduled. New requests enter the scheduler only between
engine steps, so an arrival is first seen by the schedule() call after the executing step finishes, and is first run in
the step after the already-queued one: it waits rem(N) + T(N+1) + its own step. With this policy alone both N and N+1
can be big solo steps. RADIANCE_ADAPTIVE_CHUNK_SYNC=1 adds an engine hook (EngineCore.step_with_batch_queue): while a
step is in flight and the next step would be a big solo step, it does not schedule ahead but waits for the in-flight
step first, so big solo steps run 1-deep and an arrival waits rem(N) + its own (capped) step. Cost: the host prep of
each big step is no longer hidden behind the previous one.

Instrumentation (stderr, prefix [radiance.adaptive_chunk]): one ON line at the first schedule(); a line on every
switch between big and capped budgets with the trigger; a stats line every RADIANCE_ADAPTIVE_CHUNK_STATS_S seconds when
the counts changed. Counts per schedule() call that had requests: big (full budget, solo long prefill), cap_long (capped
while some request had more than CAP tokens left: the policy held a long prefill back), cap (capped, nothing long:
budget irrelevant), hold (SYNC: schedule-ahead skipped). The chunk the long request actually got is measured as the
delta of its num_computed_tokens at the next call and kept as a histogram per class.
"""
import os
import sys
import time


def _int_env(name, default):
    v = os.environ.get(name, str(default)).strip()
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"[radiance.adaptive_chunk] {name} must be an integer, got {v!r}") from None


BIG = _int_env("RADIANCE_ADAPTIVE_CHUNK", 0)
CAP = _int_env("RADIANCE_ADAPTIVE_CHUNK_CAP", 4096)
SYNC = os.environ.get("RADIANCE_ADAPTIVE_CHUNK_SYNC", "0").strip() == "1"
STATS_S = float(os.environ.get("RADIANCE_ADAPTIVE_CHUNK_STATS_S", "60") or 0)
ENABLED = BIG > 0

_checked = False
_count = {"big": 0, "cap_long": 0, "cap": 0, "hold": 0}
_chunks = {"big": {}, "cap_long": {}}   # chunk sizes the long request got, per decision class
_track = None                           # (class, request, num_computed_tokens at the decision)
_mode = None                            # last class (for switch lines)
_streak = 0                             # calls in the current mode
_last_stats = 0.0
_last_logged = None


def _log(msg):
    sys.stderr.write(f"[radiance.adaptive_chunk] {msg}\n")
    sys.stderr.flush()


def _check(sched):
    global _checked
    _checked = True
    full = sched.max_num_scheduled_tokens
    if full != BIG:
        raise RuntimeError(f"[radiance.adaptive_chunk] RADIANCE_ADAPTIVE_CHUNK={BIG} but the scheduler's "
                           f"max_num_scheduled_tokens is {full}: start vLLM with --max-num-batched-tokens {BIG}")
    if not 0 < CAP < BIG:
        raise RuntimeError(f"[radiance.adaptive_chunk] need 0 < RADIANCE_ADAPTIVE_CHUNK_CAP ({CAP}) < "
                           f"RADIANCE_ADAPTIVE_CHUNK ({BIG})")
    blk = getattr(sched.cache_config, "block_size", None)
    _log(f"ON: step budget {BIG} when one request is alone with > {CAP} tokens left to compute, else {CAP} "
         f"(scheduler {type(sched).__name__}, block {blk}, align split {sched.need_mamba_block_aligned_split}, "
         f"sync={int(SYNC)})")


def _left(r):
    # tokens this request still has to schedule (the scheduler's own num_new_tokens for a running request)
    return r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens


def _solo_long(sched):
    """The single request in the scheduler if it has more than CAP tokens left, else None."""
    nr, nw, ns = len(sched.running), len(sched.waiting), len(sched.skipped_waiting)
    if nr + nw + ns != 1 or getattr(sched, "num_waiting_for_streaming_input", 0):
        return None
    r = sched.running[0] if nr else (sched.waiting.peek_request() if nw else sched.skipped_waiting.peek_request())
    return r if _left(r) > CAP else None


def _first_long(sched):
    for r in sched.running:
        if _left(r) > CAP:
            return r
    for q in (sched.waiting, sched.skipped_waiting):
        for r in q:
            if _left(r) > CAP:
                return r
    return None


def _fmt(h):
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(h.items(), key=lambda kv: -kv[1])) + "}"


def _stats_line():
    return (f"big={_count['big']} cap_long={_count['cap_long']} cap={_count['cap']} hold={_count['hold']} "
            f"chunks big {_fmt(_chunks['big'])} cap_long {_fmt(_chunks['cap_long'])}")


def budget(sched, full):
    """Token budget for this schedule() call (called with the scheduler's max_num_scheduled_tokens)."""
    global _track, _mode, _streak, _last_stats, _last_logged
    if not _checked:
        _check(sched)
    if _track is not None:  # the chunk the tracked long request got in the previous step
        cls, r, c0 = _track
        d = r.num_computed_tokens - c0
        if d > 0:
            h = _chunks[cls]
            h[d] = h.get(d, 0) + 1
        _track = None
    if not (sched.running or sched.waiting or sched.skipped_waiting):
        return min(full, CAP)
    solo = _solo_long(sched)
    if solo is not None:
        cls, b, long_req = "big", full, solo
    else:
        long_req = _first_long(sched)
        cls, b = ("cap_long" if long_req is not None else "cap"), min(full, CAP)
    _count[cls] += 1
    if long_req is not None:
        _track = (cls, long_req, long_req.num_computed_tokens)
    if cls != _mode and (cls == "big" or _mode == "big"):
        why = (f"1 request alone, {_left(solo)} tokens left" if solo is not None else
               f"{len(sched.running)} running, {len(sched.waiting) + len(sched.skipped_waiting)} waiting"
               + (f", long request {_left(long_req)} tokens left" if long_req is not None else ""))
        _log(f"-> {cls} budget {b} ({why}) after {_streak} {_mode or '-'} calls; {_stats_line()}")
        _streak = 0
    elif cls == "cap" and _mode == "cap_long":  # a held-back long prefill is fully scheduled: snapshot
        _log(f"long prefill done after {_streak} cap_long calls; {_stats_line()}")
        _streak = 0
    if cls != _mode:
        _streak = 0
    _mode = cls
    _streak += 1
    now = time.monotonic()
    if STATS_S > 0 and now - _last_stats >= STATS_S:
        _last_stats = now
        line = _stats_line()
        if line != _last_logged:
            _last_logged = line
            _log("stats " + line)
    return b


def hold(sched):
    """SYNC: True when a step is in flight (caller checks) and the next step would be a big solo step."""
    if _solo_long(sched) is None:
        return False
    _count["hold"] += 1
    return True
