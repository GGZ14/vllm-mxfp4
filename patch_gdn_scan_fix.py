#!/usr/bin/env python3
"""libr4d #4: optional exact GDN chunk scan (radiance_gdn2) in place of libr4d v0.5.0's.

libr4d v0.5.0's r4d_gdn_chunk_scan_k128_v128_c64_bf16 splits each in-chunk decay product at the chunk
midpoint and clamps each half at e^80, so a (sequence, head) whose chunk span G[first] - G[last] exceeds
160 gets finite but wrong outputs and carried state. radiance_gdn2 is the same kernel without the split
(every decay factor it forms is <= 1), with the same 18-argument ABI. Details: MOE-GFX1201.md.

With RADIANCE_GDN_SCAN_FIX=1, radiance_gdn binds _CHUNK_SCAN to radiance_gdn2's scan instead of r4d's.
Default off: with the knob unset nothing is imported and radiance_gdn runs exactly as before.

The module must be built into site-packages before the server starts (moe-gdn2/build.sh, R4D_SRC = a
libr4d v0.5.0 checkout). Asked for and missing is fatal (import error at startup), never a silent
fallback to the inexact scan.

Usage: python3 patch_gdn_scan_fix.py [PATH_TO_radiance_gdn.py]
(default: radiance_gdn.py in the interpreter's site-packages, the copy vLLM imports; not a radiance_gdn.py
that happens to sit next to this script). Idempotent; fails before writing on source drift.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _patchlib import apply  # noqa: E402

F = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"]) / "radiance_gdn.py"
if not F.is_file():
    raise SystemExit(f"  FAIL  gdn scan fix: {F} not found")

SENT = "RADIANCE_GDN_SCAN_FIX"

BIND_OLD = '_CHUNK_SCAN = _bind("gdn_chunk_scan", head_k=HEAD_K, head_v=HEAD_V, chunk=CHUNK)\n'
BIND_NEW = (
    BIND_OLD +
    "# RADIANCE_GDN_SCAN_FIX=1 (libr4d #4, patch_gdn_scan_fix.py): the exact chunk scan from radiance_gdn2\n"
    "# (libr4d v0.5.0's kernel without the midpoint decay split; same ABI) replaces r4d's. Default off.\n"
    "_SCAN_FIX = None\n"
    "if os.environ.get(\"RADIANCE_GDN_SCAN_FIX\", \"0\") == \"1\" and _CHUNK_SCAN is not None:\n"
    "    import radiance_gdn2 as _SCAN_FIX   # fail loudly: asked for, must be present\n"
    "    if (int(_SCAN_FIX.GDN_HEAD_K), int(_SCAN_FIX.GDN_HEAD_V), int(_SCAN_FIX.GDN_CHUNK)) != \\\n"
    "            (HEAD_K, HEAD_V, CHUNK):\n"
    "        raise RuntimeError(\"radiance_gdn2 geometry does not match r4d's gdn_chunk_scan\")\n"
    "    _CHUNK_SCAN = _SCAN_FIX.gdn_chunk_scan_k128_v128_c64_bf16\n"
    "    sys.stderr.write(\"[radiance.gdn] exact chunk scan ON (radiance_gdn2, libr4d #4 fix)\\n\")\n"
)

s = F.read_text()
if SENT in s:
    print("  NOOP  gdn scan fix already applied")
    sys.exit(0)
if s.count("def fused_prefill(") != 1 or s.count("    _CHUNK_SCAN(\n") != 1:
    raise SystemExit("  FAIL  gdn scan fix: fused_prefill / its _CHUNK_SCAN call not unique")
apply(F, BIND_OLD, BIND_NEW, "_SCAN_FIX = None", "gdn scan fix bind")
