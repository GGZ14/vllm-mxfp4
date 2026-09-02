#!/usr/bin/env python3
"""Skip the two `.contiguous()` copies on the GDN gate tensors when the R4D path serves the step.

In QwenGDN's forward_cuda (Qwen3.5 layout) `b, a = self.split_ba(ba)` are column slices of the
in_proj_ba output, and vLLM copies both to contiguous before the core op. The R4D kernels take a
row stride (radiance_gdn passes `a.stride(0)`, and _plan only requires the head axis to be unit
stride), so on that path the copies are dead work: one launch + one ~3 us gap per linear-attention
layer, 48 layers per decode step (census 2026-09-02). The Triton fallback body still gets
contiguous tensors: the copies move to just after the radiance hook's return, where only a step
the R4D path declined reaches them. Gated by RADIANCE_GDN_STRIDED_GATES at runtime (default 0) so
the same build A/Bs; it needs patch_r4d.py to have run first (the hook text is the anchor).

VERDICT 2026-09-02: neutral at serve level (22.34-22.41 vs 22.31-22.33 ms/step on top of the fused
GDN norm, output byte-identical, GSM8K 97.80%). Kept dark at 0; the copies are evidently not on
the critical path, or inductor re-packs the custom-op inputs regardless.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

PURELIB = Path(sysconfig.get_paths()["purelib"])
L = PURELIB / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"

SPLIT_OLD = (
    "            b, a = self.split_ba(ba)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()\n"
)
SPLIT_NEW = (
    "            b, a = self.split_ba(ba)\n"
    "            # --- RADIANCE (patch_gdn_glue.py): the R4D core reads strided gates ---\n"
    "            if not (_radiance_gdn is not None and _radiance_gdn.STRIDED_GATES):\n"
    "                b = b.contiguous()\n"
    "                a = a.contiguous()\n"
)
HOOK_OLD = (
    "            if _radiance_gdn.forward_core_fused(self, mixed_qkv, b, a, core_attn_out):\n"
    "                return\n"
)
HOOK_NEW = (
    HOOK_OLD
    + "            b = b.contiguous()   # patch_gdn_glue.py: the Triton body below wants them packed\n"
    + "            a = a.contiguous()\n"
)

apply(L, SPLIT_OLD, SPLIT_NEW, "patch_gdn_glue.py): the R4D core reads strided gates", "gdn strided gates")
apply(L, HOOK_OLD, HOOK_NEW, "patch_gdn_glue.py: the Triton body below", "gdn fallback re-pack")
