#!/usr/bin/env python3
"""MoE lane under vLLM async scheduling (radiance-glue/, RADIANCE_MOE_ASYNC=1).

The launcher's knob turns on --async-scheduling and turns off disable_padded_drafter_batch (vLLM refuses the
combination). Async scheduling then prepares step N+1 (scheduler, _update_states, _prepare_inputs, attention metadata)
and launches it while step N's sampler and MTP drafter run on the GPU. That hides the ~2.2 ms per decode step of
GPU-idle host glue measured in radiance-glue Phase 1 (target input prep ~1.4 ms, drafter input prep ~0.4 ms, engine
~0.16 ms, bookkeeping ~0.1 ms), and every verify batch is uniform (1 + K rows per request), so it always replays the
FULL CUDA graph.

What does not fit: the image's dynamic draft controller (radiance_draft.py, RADIANCE_DYNAMIC_DRAFT). It returns a
ragged list[list[int]] of per-request drafts; vLLM's async path keeps drafts on the GPU as one [B, K] tensor and
asserts on anything else (gpu_model_runner._prepare_input_ids). It also gates every drafter pass on a host D2H, and
builds its n-gram context from token_ids_cpu, which under async scheduling holds placeholders for the newest tokens.
This patch makes radiance_draft.install() a no-op under RADIANCE_MOE_ASYNC=1, so MTP drafts the stock fixed K and the
target verifies all K (the drafter-loop graph knob, RADIANCE_MOE_DRAFT_GRAPH, keeps working: the loop passes stay
uniform; RADIANCE_MOE_DRAFT_OVERLAP has nothing left to overlap).

Inert unless RADIANCE_MOE_ASYNC=1 at runtime (the inserted check reads the env). Usage:
python3 patch_moe_async.py [SITE_PACKAGES]. Idempotent; fails before writing on drift (_patchlib.apply).
"""
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])

apply(
    SP / "radiance_draft.py",
    "def install():\n"
    "    \"\"\"Entry from radiance_kernels.install_all(). Env-gated on RADIANCE_DYNAMIC_DRAFT.\"\"\"\n",
    "def install():\n"
    "    \"\"\"Entry from radiance_kernels.install_all(). Env-gated on RADIANCE_DYNAMIC_DRAFT.\"\"\"\n"
    "    # radiance-glue: RADIANCE_MOE_ASYNC=1 (patch_moe_async.py) -- vLLM async scheduling keeps drafts on the GPU as\n"
    "    # one [B, K] tensor; this controller's ragged drafts and per-pass host gate do not fit, so stock MTP drafts.\n"
    "    if os.environ.get(\"RADIANCE_MOE_ASYNC\", \"0\").strip() == \"1\":\n"
    "        _log(\"RADIANCE_DYNAMIC_DRAFT=OFF under RADIANCE_MOE_ASYNC=1 (async scheduling: stock MTP, fixed width)\")\n"
    "        return\n",
    "radiance-glue: RADIANCE_MOE_ASYNC=1 (patch_moe_async.py)",
    "moe async: dynamic draft off under async scheduling (radiance_draft.py)",
)
