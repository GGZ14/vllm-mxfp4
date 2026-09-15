#!/usr/bin/env python3
"""REJECTED 2026-09-14: folding the drafter query rows is 4-34% SLOWER on gfx1201. Not wired in.

Kept as the reproducer and the record, because the idea is a natural one to re-derive from
vLLM #45743 (MiniMax M3 on MI355X) -- "launched one workgroup per request and processed all
draft positions together, reusing key loads", worth up to 48.9% on their indexer kernel.

The analogous defect is real here. `kernel_unified_attention_*` partitions work as
`BLOCK_Q = BLOCK_M // num_queries_per_kv` query POSITIONS per program, and upstream picks
`BLOCK_M = 16` for any GQA ratio <= 16. The DFlash2 drafter is GQA 4 and presents
`q_len = 1 + num_speculative_tokens = 8` rows per request, so BLOCK_Q = 4 splits every request
across two programs that each re-stream the same sliding-window-2048 K/V. Setting BLOCK_M = 32
gives BLOCK_Q = 8 and one program per (request, kv head), and the output is **bit-identical**
(0.00e+00 max difference at every shape below) -- the reasoning about correctness was right.

It is the performance reasoning that does not survive contact. Measured in-image on one R9700
(gfx1201, 64 CU), drafter shape 16 q heads / 4 kv heads per rank at TP=2, head 128, window 2048,
median of 5 x 60 calls, us/call:

    causal nseq  ctx     BLOCK_M=16   BLOCK_M=32    delta
      1     1    2048        63.7         79.0     +24.1%
      1     1    32768       58.1         72.1     +24.1%
      1     8    2048        71.8         96.5     +34.3%
      1     8    32768       69.6         89.7     +28.8%
      0     1    8192        67.4         70.1      +4.0%
      0     8    8192        73.7         90.7     +23.0%

Why: this kernel is not K/V-bandwidth-bound at our scale, it is parallelism-starved, and the
fold spends the scarce resource. Captured launch geometry at ctx 8192:

    nseq=1  BLOCK_M=16 -> grid (3, 4) =  12 programs   |  BLOCK_M=32 -> grid (2, 4) =  8
    nseq=8  BLOCK_M=16 -> grid (24, 4) = 96 programs   |  BLOCK_M=32 -> grid (16, 4) = 64

12 programs on 64 CUs is 19% occupancy before the fold and 12.5% after; even at our maximum
`--max-num-seqs 8` the fold lands at exactly 64 programs, one per CU with no latency hiding.
The blog's win came at concurrency 128+ where the grid is saturated and re-reading keys is the
actual cost. Same kernel, same change, opposite regime -- which is the whole lesson: fill the
GPU before judging a tiling.

Re-test only if `--max-num-seqs` rises by an order of magnitude, or on a part with many more
CUs. To re-run the measurement, apply this patch to a scratch copy of the module and flip the
module-global `_RADIANCE_FOLD_Q` between "" and "auto" between timed runs (no restart needed).

Knobs, if ever wired in: RADIANCE_DRAFT_ATTN_BLOCK_Q (unset = upstream; `auto` = cover the
batch q_len; an integer = force a BLOCK_Q floor), RADIANCE_DRAFT_ATTN_MAX_Q (default 16, keeps
real prefill on its own tiling), RADIANCE_DRAFT_ATTN_BM_MAX (default 32, which deliberately
excludes the R4D fallback shape -- GQA 6 at head_size 256 -- since that would need BLOCK_M=64,
the tile most likely to spill).

Upstream ships the same manoeuvre as `tuned_large_head` (BLOCK_M = 32) gated to
`current_platform.is_device_capability_family(100)`, i.e. NVIDIA B200 only. This patch leaves
that path untouched and only acts when it did not fire.

Idempotent. NOT registered in run_paroquant.sh -- an unused patch is an upgrade liability
(anchors drift and have to be re-derived on every vLLM bump) with no win to pay for it.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
# One sentinel per edit: apply() short-circuits on the whole file, so a shared sentinel would
# let the second edit mask the third and silently ship the knobs without the fold.
SENT_IMP = "import os\nfrom typing import Any"
SENT_CONST = "_RADIANCE_FOLD_BM_MAX ="
SENT_BM = "one program per request, not per q-block"

# ---- 1. os import + the knobs, read once at module import ------------------------------------
IMP_ANCHOR = """from typing import Any

import torch
"""
IMP_NEW = """import os
from typing import Any

import torch
"""

CONST_ANCHOR = """logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())
"""
CONST_NEW = '''logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())

# --- radiance (patch_draft_attn_blockm.py) -----------------------------------------------------
# Query-row folding for speculative batches. Empty/unset preserves upstream tiling exactly.
_RADIANCE_FOLD_Q = os.environ.get("RADIANCE_DRAFT_ATTN_BLOCK_Q", "").strip()
_RADIANCE_FOLD_Q_MAX = int(os.environ.get("RADIANCE_DRAFT_ATTN_MAX_Q") or 16)
_RADIANCE_FOLD_BM_MAX = int(os.environ.get("RADIANCE_DRAFT_ATTN_BM_MAX") or 32)
'''

# ---- 2. the BLOCK_M selection itself ---------------------------------------------------------
BM_ANCHOR = """    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2
"""
BM_NEW = '''    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # --- radiance (patch_draft_attn_blockm.py): one program per request, not per q-block ------
    # BLOCK_Q is how many query POSITIONS a program covers. At the DFlash2 drafter's GQA 4 the
    # default BLOCK_M=16 gives BLOCK_Q=4, so its q_len=8 request is split across two programs
    # that each re-stream the same sliding-window K/V. Widening BLOCK_M until BLOCK_Q covers
    # q_len folds them into one and the keys are loaded once (cf. vLLM #45743). Per-row math is
    # unchanged -- rows keep their own causal limits and accumulators.
    if (
        not tuned_large_head
        and _RADIANCE_FOLD_Q
        and 1 < max_seqlen_q <= _RADIANCE_FOLD_Q_MAX
    ):
        _want_q = (
            triton.next_power_of_2(max_seqlen_q)
            if _RADIANCE_FOLD_Q == "auto"
            else int(_RADIANCE_FOLD_Q)
        )
        _bm = triton.next_power_of_2(max(16, num_queries_per_kv * _want_q))
        # Only ever widen, and never past the spill guard: a tile that costs occupancy trades
        # the K/V saving straight back.
        if BLOCK_M < _bm <= _RADIANCE_FOLD_BM_MAX:
            BLOCK_M = _bm
            BLOCK_Q = BLOCK_M // num_queries_per_kv
'''

apply(TARGET, IMP_ANCHOR, IMP_NEW, SENT_IMP, "triton_unified_attention.py (import os)")
apply(TARGET, CONST_ANCHOR, CONST_NEW, SENT_CONST, "triton_unified_attention.py (knobs)")
apply(TARGET, BM_ANCHOR, BM_NEW, SENT_BM, "triton_unified_attention.py (BLOCK_M fold)")
