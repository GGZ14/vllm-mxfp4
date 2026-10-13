#!/usr/bin/env python3
"""Adaptive prefill chunk budget (radiance-adaptive-chunk/, RADIANCE_ADAPTIVE_CHUNK=<big budget>).

Copies moe-adaptivechunk/radiance_adaptive_chunk.py into site-packages and patches two vLLM files:
  1. v1/core/sched/scheduler.py  Scheduler.schedule: right after `token_budget = self.max_num_scheduled_tokens`, the
     step's budget becomes radiance_adaptive_chunk.budget(self, token_budget): the full budget (the server must run with
     --max-num-batched-tokens <big budget>, checked at the first call) only when one request is alone in the scheduler
     with more than RADIANCE_ADAPTIVE_CHUNK_CAP (4096) tokens left; otherwise the cap (today's 4096 behaviour).
  2. v1/engine/core.py  EngineCore.step_with_batch_queue (async scheduling's 2-deep batch queue), only with
     RADIANCE_ADAPTIVE_CHUNK_SYNC=1: while a step is in flight and the next step would be a big solo step, do not
     schedule ahead; wait for the in-flight step first (big solo steps run 1-deep, so an arrival waits for at most one
     big step plus its own).
Inert unless RADIANCE_ADAPTIVE_CHUNK is a positive integer at runtime (module flags read at import; nothing imported
otherwise). Usage: python3 patch_adaptive_chunk.py [SITE_PACKAGES]. Idempotent (sentinels); fails before writing on
drift (_patchlib.apply: unique anchor, ast-checked).
"""
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])

src = HERE / "moe-adaptivechunk" / "radiance_adaptive_chunk.py"
if not src.exists():
    raise SystemExit(f"  FAIL  adaptive chunk: {src} missing")
shutil.copyfile(src, SP / "radiance_adaptive_chunk.py")
print(f"  OK    adaptive chunk: installed {SP / 'radiance_adaptive_chunk.py'}")

_ON = "__import__(\"os\").environ.get(\"RADIANCE_ADAPTIVE_CHUNK\", \"0\").strip() not in (\"\", \"0\")"

# 1) the per-step budget in the scheduler
S = SP / "vllm/v1/core/sched/scheduler.py"
apply(S, "\nlogger = init_logger(__name__)\n",
      "\nlogger = init_logger(__name__)\n"
      "\n# radiance-adaptive-chunk: RADIANCE_ADAPTIVE_CHUNK module flag (patch_adaptive_chunk.py)\n"
      f"_RADIANCE_ACHUNK = {_ON}\n"
      "if _RADIANCE_ACHUNK:\n"
      "    import radiance_adaptive_chunk as _radiance_achunk  # fail loudly: asked for, must be present\n",
      "radiance-adaptive-chunk: RADIANCE_ADAPTIVE_CHUNK module flag",
      "adaptive chunk: module flag in scheduler.py")
apply(
    S,
    "        token_budget = self.max_num_scheduled_tokens\n"
    "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    "            # Do not schedule any requests when paused.\n"
    "            token_budget = 0\n",
    "        token_budget = self.max_num_scheduled_tokens\n"
    "        # radiance-adaptive-chunk: full budget only for a long prefill alone in the scheduler, else the cap\n"
    "        if _RADIANCE_ACHUNK:\n"
    "            token_budget = _radiance_achunk.budget(self, token_budget)\n"
    "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    "            # Do not schedule any requests when paused.\n"
    "            token_budget = 0\n",
    "radiance-adaptive-chunk: full budget only for a long prefill alone",
    "adaptive chunk: per-step budget in Scheduler.schedule",
)

# 2) SYNC: no schedule-ahead behind a big solo step (async scheduling's batch queue)
E = SP / "vllm/v1/engine/core.py"
apply(E, "\nlogger = init_logger(__name__)\n",
      "\nlogger = init_logger(__name__)\n"
      "\n# radiance-adaptive-chunk: RADIANCE_ADAPTIVE_CHUNK_SYNC module flag (patch_adaptive_chunk.py)\n"
      f"_RADIANCE_ACHUNK_SYNC = {_ON} and __import__(\"os\").environ.get(\n"
      "    \"RADIANCE_ADAPTIVE_CHUNK_SYNC\", \"0\").strip() == \"1\"\n"
      "if _RADIANCE_ACHUNK_SYNC:\n"
      "    import radiance_adaptive_chunk as _radiance_achunk  # fail loudly: asked for, must be present\n",
      "radiance-adaptive-chunk: RADIANCE_ADAPTIVE_CHUNK_SYNC module flag",
      "adaptive chunk: sync flag in engine/core.py")
apply(
    E,
    "        model_executed = False\n"
    "        deferred_scheduler_output = None\n"
    "        if self.scheduler.has_requests():\n",
    "        model_executed = False\n"
    "        deferred_scheduler_output = None\n"
    "        # radiance-adaptive-chunk SYNC: a big solo prefill step is not scheduled ahead of the in-flight step;\n"
    "        # fall through to the blocking pop below (the queue is non-empty), then schedule with fresh arrivals.\n"
    "        _rad_hold = (_RADIANCE_ACHUNK_SYNC and bool(batch_queue)\n"
    "                     and _radiance_achunk.hold(self.scheduler))\n"
    "        if _rad_hold:\n"
    "            model_executed = True  # a step is in flight: no idle sleep in _process_engine_step\n"
    "        if self.scheduler.has_requests() and not _rad_hold:\n",
    "radiance-adaptive-chunk SYNC: a big solo prefill step",
    "adaptive chunk: sync hold in EngineCore.step_with_batch_queue",
)
