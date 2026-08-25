#!/usr/bin/env python3
"""Point vLLM's rms_norm + per-token-fp8-quant fusion at a radiance op instead of aiter's.

vLLM's AiterRMSNormDynamicQuantPattern (and its fused-add sibling) carry the right PATTERN -- the
matcher key is exactly what scaled_fp8_quant(use_per_token_if_dynamic=True) emits -- but their
REPLACEMENT is `rocm_aiter_ops.get_rmsnorm_fused_*_dynamic_quant_op()`. aiter's RMSNorm does not work
on RDNA4 (hence VLLM_ROCM_USE_AITER_RMSNORM=0 in this build), so enabling
`pass_config.fuse_norm_quant` dies inside aiter's JIT:

    _rocm_aiter_rmsnorm_fused_dynamic_quant_impl -> aiter/ops/rmsnorm.py -> aiter/jit/core.py
    RuntimeError: HIP runtime library (libamdhip64.so) not found ... TileLang's ROCm backend

This swaps only the two FUSED_OP class attributes. The patterns, the matcher keys and the pass are
untouched. Gated by RADIANCE_RMS_QUANT_FUSION (default off). Idempotent.

NOTE the replacement can only ever FIRE if the quant is visible in the traced graph -- with the quant
inside radiance_mxfp4's opaque custom op the pattern matches zero times, measured. That is what
RADIANCE_MXFP4_HOIST_QUANT is for; the two go together.
"""
import sysconfig
from pathlib import Path

F = (Path(sysconfig.get_paths()["purelib"])
     / "vllm/compilation/passes/fusion/rocm_aiter_fusion.py")

MARK = "# --- RADIANCE: native rms+quant replacement op ---"
SHIM = f'''{MARK}
import os as _rq_os
if _rq_os.environ.get("RADIANCE_RMS_QUANT_FUSION", "0") == "1":
    import radiance_rmsquant as _rq
    AiterRMSNormDynamicQuantPattern.FUSED_OP = staticmethod(_rq.fused_rmsnorm_quant)
    AiterFusedAddRMSNormDynamicQuantPattern.FUSED_OP = staticmethod(_rq.fused_add_rmsnorm_quant)
    import sys as _rq_sys
    _rq_sys.stderr.write("[radiance.rmsquant] rms+quant fusion -> radiance op "
                         "(aiter replacement bypassed)\\n")
'''

src = F.read_text()
if MARK in src:
    print("[patch_rmsquant_fusion] already applied")
    raise SystemExit(0)

# Append after both classes exist. Anchor on the last of the two we override.
anchor = "class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):"
if anchor not in src:
    raise SystemExit("[patch_rmsquant_fusion] anchor not found -- vLLM layout changed, NOT applied")
src = src.replace(anchor, SHIM + "\n\n" + anchor, 1)
F.write_text(src)
print(f"[patch_rmsquant_fusion] applied to {F}")
