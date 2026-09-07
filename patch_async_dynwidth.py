#!/usr/bin/env python3
"""Apply the RADIANCE dynamic verify-width cap under ASYNC scheduling.

patch_dynwidth.py caps request.spec_token_ids in Scheduler.update_draft_token_ids, but the async
engine loop never calls that (drafts stay on the GPU); AsyncScheduler._update_after_schedule instead
points every decode request at ONE shared placeholder list of the full num_spec_tokens width, so
under --async-scheduling every request verified the full width (GPU active 25.5 -> 29.0 ms at
conc-8, 2026-08-31). Give each request its own copy of the placeholder list and run the same cap on
it; schedule() then turns the shorter list into fewer verify rows and the worker consumes only
that many GPU drafts (prepare_inputs sizes draft tokens from scheduled_spec_decode_tokens).
No-op when RADIANCE_DYNAMIC_WIDTH=0 (the cap returns early) or when patch_dynwidth.py is absent.
"""
import sys
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/v1/core/sched/async_scheduler.py"
SENTINEL = "_radiance_cap_spec_width"
ANCHOR = "            request.spec_token_ids = self._spec_token_placeholders\n"
ADD = ANCHOR + """            # RADIANCE (patch_async_dynwidth.py): dynamic verify-width cap under async scheduling.
            # The placeholder list is shared and read-only -- copy before the cap trims it.
            _rad_cap = getattr(self, "_radiance_cap_spec_width", None)
            if _rad_cap is not None and request.spec_token_ids:
                request.spec_token_ids = list(self._spec_token_placeholders)
                _rad_cap(request)
"""
src = TARGET.read_text()
if SENTINEL in src:
    print(f"[patch_async_dynwidth] already applied: {TARGET}")
    sys.exit(0)
if src.count(ANCHOR) != 1:
    print(f"[patch_async_dynwidth] FATAL: anchor found {src.count(ANCHOR)}x in {TARGET}", file=sys.stderr)
    sys.exit(1)
TARGET.write_text(src.replace(ANCHOR, ADD))
print(f"[patch_async_dynwidth] applied: {TARGET}")
