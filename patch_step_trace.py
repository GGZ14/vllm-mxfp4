#!/usr/bin/env python3
"""Per-step worker timing trace, env-gated: RADIANCE_STEP_TRACE=N logs a summary every N steps on
rank 0 (0 = off, nothing is wrapped). Four numbers locate a scheduling bubble without a profiler:
  exec_model   CPU time inside Worker.execute_model (input prep + enqueue; GPU is async)
  sample_tok   CPU time inside Worker.sample_tokens (sampler + drafter enqueue + AsyncOutput)
  rpc_wait     idle gap from the end of sample_tokens(N) to the start of execute_model(N+1):
               the engine's schedule + IPC latency as seen by the worker
  out_wait     time the async output thread blocks in AsyncOutput.get_output (GPU step tail)
  out_iv       interval between consecutive outputs = the effective step time
"""
import sys
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
W = SP / "vllm/v1/worker/gpu_worker.py"
A = SP / "vllm/v1/worker/gpu/async_utils.py"
SENT = "RADIANCE step trace"

TAIL_A = '''

# ---- RADIANCE step trace (patch_step_trace.py) ----------------------------------------------
import os as _rt_os, time as _rt_time
_RT_OUT = {"wait": [], "iv": [], "last": None}
if int(_rt_os.environ.get("RADIANCE_STEP_TRACE", "0") or 0) > 0:
    _rt_orig_get_output = AsyncOutput.get_output
    _rt_orig_ao_init = AsyncOutput.__init__

    def _rt_ao_init(self, *a, **k):
        self._rt_ev_mid = torch.cuda.Event(enable_timing=True)
        self._rt_ev_mid.record()  # main stream: sampler enqueued, drafter not yet
        _rt_orig_ao_init(self, *a, **k)

    AsyncOutput.__init__ = _rt_ao_init

    def _rt_get_output(self):
        t0 = _rt_time.perf_counter()
        self.copy_event.synchronize()
        t1 = _rt_time.perf_counter()
        _RT_OUT["wait"].append(t1 - t0)
        if _RT_OUT["last"] is not None:
            _RT_OUT["iv"].append(t1 - _RT_OUT["last"])
        _RT_OUT["last"] = t1
        return _rt_orig_get_output(self)

    AsyncOutput.get_output = _rt_get_output
'''

TAIL_W = '''

# ---- RADIANCE step trace (patch_step_trace.py) ----------------------------------------------
import os as _rt_os, time as _rt_time
_RT_N = int(_rt_os.environ.get("RADIANCE_STEP_TRACE", "0") or 0)
if _RT_N > 0:
    from vllm.logger import init_logger as _rt_init_logger
    _rt_log = _rt_init_logger("vllm.radiance.steptrace")
    import statistics as _rt_stats, torch as _rt_torch
    _RT = {"em": [], "st": [], "rpc": [], "gpu": [], "tail": [], "toks": [], "evs": [], "n": 0, "last_st_end": None}
    _rt_orig_em = Worker.execute_model
    _rt_orig_st = Worker.sample_tokens

    def _rt_fmt(v):
        return f"{1000 * _rt_stats.median(v):6.2f}/{1000 * max(v):7.2f}" if v else "   -  /   -   "

    def _rt_execute_model(self, *a, **k):
        t0 = _rt_time.perf_counter()
        if _RT["last_st_end"] is not None:
            _RT["rpc"].append(t0 - _RT["last_st_end"])
            _RT["last_st_end"] = None
        so = a[0] if a else k.get("scheduler_output")
        _RT["toks"].append(float(getattr(so, "total_num_scheduled_tokens", 0) or 0) / 1000.0)
        ev_s = _rt_torch.cuda.Event(enable_timing=True); ev_s.record()
        _RT["cur_ev"] = ev_s
        r = _rt_orig_em(self, *a, **k)
        _RT["em"].append(_rt_time.perf_counter() - t0)
        return r

    def _rt_sample_tokens(self, *a, **k):
        t0 = _rt_time.perf_counter()
        r = _rt_orig_st(self, *a, **k)
        t1 = _rt_time.perf_counter()
        _RT["st"].append(t1 - t0)
        ev_e = _rt_torch.cuda.Event(enable_timing=True); ev_e.record()
        ev_mid = getattr(r, "_rt_ev_mid", None)
        if _RT.get("cur_ev") is not None:
            _RT["evs"].append((_RT.pop("cur_ev"), ev_mid, ev_e))
        done = [p for p in _RT["evs"] if p[2].query()]
        for p in done:
            _RT["gpu"].append(p[0].elapsed_time(p[2]) / 1000.0)
            if p[1] is not None:
                _RT["tail"].append(p[1].elapsed_time(p[2]) / 1000.0)
            _RT["evs"].remove(p)
        _RT["last_st_end"] = t1
        _RT["n"] += 1
        if _RT["n"] % _RT_N == 0:
            from vllm.v1.worker.gpu import async_utils as _au
            o = getattr(_au, "_RT_OUT", {"wait": [], "iv": []})
            _rt_log.info(
                "rank%d steps=%d  median/max ms  exec_model %s | sample_tok %s | rpc_wait %s | gpu_span %s | drafter_tail %s | out_wait %s | out_iv %s | toks/step %s",
                getattr(self, "rank", -1), _RT["n"], _rt_fmt(_RT["em"]), _rt_fmt(_RT["st"]), _rt_fmt(_RT["rpc"]), _rt_fmt(_RT["gpu"]), _rt_fmt(_RT["tail"]),
                _rt_fmt(o["wait"]), _rt_fmt(o["iv"]), _rt_fmt(_RT["toks"]))
            _RT["em"].clear(); _RT["st"].clear(); _RT["rpc"].clear(); _RT["gpu"].clear(); _RT["tail"].clear(); _RT["toks"].clear()
            o["wait"].clear(); o["iv"].clear()
        return r

    Worker.execute_model = _rt_execute_model
    Worker.sample_tokens = _rt_sample_tokens
    _rt_log.info("RADIANCE_STEP_TRACE=%d armed (Worker.execute_model / sample_tokens / AsyncOutput.get_output wrapped)", _RT_N)
'''

TAIL_S = '''

# ---- RADIANCE step trace (patch_step_trace.py) ----------------------------------------------
import os as _rt_os, time as _rt_time
_RT_N = int(_rt_os.environ.get("RADIANCE_STEP_TRACE", "0") or 0)
if _RT_N > 0:
    from vllm.logger import init_logger as _rt_init_logger
    _rt_log = _rt_init_logger("vllm.radiance.steptrace")
    _RTS = {"sch": [], "upd": [], "period": [], "gap": [], "n": 0, "last_sch": None, "last_upd_end": None}
    _rt_orig_schedule = Scheduler.schedule
    _rt_orig_update = Scheduler.update_from_output

    import statistics as _rt_stats

    def _rt_fmt(v):
        return f"{1000 * _rt_stats.median(v):6.2f}/{1000 * max(v):7.2f}" if v else "   -  /   -   "

    def _rt_schedule(self, *a, **k):
        t0 = _rt_time.perf_counter()
        if _RTS["last_sch"] is not None:
            _RTS["period"].append(t0 - _RTS["last_sch"])
        if _RTS["last_upd_end"] is not None:
            _RTS["gap"].append(t0 - _RTS["last_upd_end"])
            _RTS["last_upd_end"] = None
        _RTS["last_sch"] = t0
        r = _rt_orig_schedule(self, *a, **k)
        _RTS["sch"].append(_rt_time.perf_counter() - t0)
        return r

    def _rt_update(self, *a, **k):
        t0 = _rt_time.perf_counter()
        r = _rt_orig_update(self, *a, **k)
        t1 = _rt_time.perf_counter()
        _RTS["upd"].append(t1 - t0)
        _RTS["last_upd_end"] = t1
        _RTS["n"] += 1
        if _RTS["n"] % _RT_N == 0:
            _rt_log.info(
                "engine steps=%d  median/max ms  schedule %s | update_from_output %s | sched_period %s | upd_end->next_sched %s | running=%d",
                _RTS["n"], _rt_fmt(_RTS["sch"]), _rt_fmt(_RTS["upd"]), _rt_fmt(_RTS["period"]), _rt_fmt(_RTS["gap"]), len(self.running))
            for key in ("sch", "upd", "period", "gap"):
                _RTS[key].clear()
        return r

    Scheduler.schedule = _rt_schedule
    Scheduler.update_from_output = _rt_update
'''

TAIL_C = '''

# ---- RADIANCE step trace (patch_step_trace.py) ----------------------------------------------
import os as _rt_os, time as _rt_time
_RT_N = int(_rt_os.environ.get("RADIANCE_STEP_TRACE", "0") or 0)
if _RT_N > 0:
    from vllm.logger import init_logger as _rt_init_logger
    _rt_log = _rt_init_logger("vllm.radiance.steptrace")
    _rt_orig_init = EngineCore.__init__

    def _rt_init(self, vllm_config, *a, **k):
        _rt_orig_init(self, vllm_config, *a, **k)
        try:
            _rt_log.info(
                "engine: batch_queue_size=%s step_fn=%s scheduler=%s async_scheduling=%s max_concurrent_batches=%s v2_runner=%s pp=%s",
                self.batch_queue_size, getattr(self.step_fn, "__name__", "?"), type(self.scheduler).__name__,
                vllm_config.scheduler_config.async_scheduling, vllm_config.max_concurrent_batches,
                vllm_config.use_v2_model_runner, vllm_config.parallel_config.pipeline_parallel_size)
        except Exception as e:  # never let tracing kill the engine
            _rt_log.info("engine trace init failed: %r", e)
        if self.batch_queue is not None:
            _orig_swbq = self.step_with_batch_queue
            _RTQ = {"entry": [], "dur": [], "early": 0, "n": 0}

            def _swbq():
                n0 = len(self.batch_queue); t0 = _rt_time.perf_counter()
                r = _orig_swbq()
                dt = _rt_time.perf_counter() - t0
                _RTQ["entry"].append(n0); _RTQ["dur"].append(dt); _RTQ["n"] += 1
                if dt < 0.002: _RTQ["early"] += 1
                if _RTQ["n"] % (2 * _RT_N) == 0:
                    e = _RTQ["entry"]
                    _rt_log.info("engine step_with_batch_queue: calls=%d entry_len0=%d entry_len1=%d entry_len2=%d fast(<2ms)=%d median_dur=%.2fms",
                                 len(e), e.count(0), e.count(1), e.count(2), _RTQ["early"], 1000 * sorted(_RTQ["dur"])[len(e) // 2])
                    _RTQ["entry"].clear(); _RTQ["dur"].clear(); _RTQ["early"] = 0
                return r
            self.step_fn = _swbq

    EngineCore.__init__ = _rt_init
'''

S = SP / "vllm/v1/core/sched/scheduler.py"
C = SP / "vllm/v1/engine/core.py"
for path, tail, need in ((A, TAIL_A, "class AsyncOutput("), (W, TAIL_W, "class Worker("), (S, TAIL_S, "class Scheduler("), (C, TAIL_C, "class EngineCore")):
    src = path.read_text()
    if SENT in src:
        print(f"[patch_step_trace] already applied: {path}"); continue
    if need not in src:
        print(f"[patch_step_trace] FATAL: {need!r} not in {path}", file=sys.stderr); sys.exit(1)
    path.write_text(src + tail)
    print(f"[patch_step_trace] applied: {path}")
