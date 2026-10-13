#!/usr/bin/env python3
"""fp8 per-channel dense layers + fused shared-expert gate for the MoE lane (moe-densefp8/, MOE-GFX1201.md).

RADIANCE_MOE_DENSE_FP8=1: at the end of vLLM's model_loader process_weights_after_loading, radiance_moe_densefp8.convert
  (model) quantizes the target's bf16 dense linears (in_proj_qkvz, out_proj, qkv_proj, o_proj, shared expert
  gate_up/down) to e4m3 with per-output-channel scales, frees the bf16 tensors before KV sizing, and routes them to the
  measured fastest kernel per batch size. Kept bf16: in_proj_ba, the router, shared_expert_gate, lm_head, the drafter.
RADIANCE_MOE_GATE_FIX=1: Qwen2MoeMLP.forward computes sigmoid(expert_gate(x)) * out in one fused kernel instead of a
  hipBLASLt GEMV + sigmoid + mul (target and drafter; bit-identical on real weights).
Both unset / 0: nothing is imported, vLLM runs as before. Asked for and missing is fatal at load.

This script copies moe-densefp8/radiance_moe_densefp8.py into site-packages and patches two vLLM files. The HIP kernel
radiance_fp8w is built separately (moe-densefp8/build.sh; only RADIANCE_MOE_DENSE_FP8=1 needs it).
Usage: python3 patch_moe_densefp8.py [SITE_PACKAGES].
Idempotent (sentinels); fails before writing on drift (_patchlib.apply: unique anchor, ast-checked).
"""
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])

src = HERE / "moe-densefp8" / "radiance_moe_densefp8.py"
if not src.exists():
    raise SystemExit(f"  FAIL  moe dense fp8: {src} missing")
shutil.copyfile(src, SP / "radiance_moe_densefp8.py")
print(f"  OK    moe dense fp8: installed {SP / 'radiance_moe_densefp8.py'}")

# 1) the conversion hook: after every per-layer / model-level post-load step, before the torchao reload bookkeeping
apply(
    SP / "vllm/model_executor/model_loader/utils.py",
    "    # Needed for torchao model reloading via model.reload_weights\n"
    "    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`\n"
    "    if model_config.quantization == \"torchao\":\n",
    "    # radiance: RADIANCE_MOE_DENSE_FP8 / RADIANCE_MOE_GATE_FIX (patch_moe_densefp8.py, moe-densefp8/): fp8\n"
    "    # per-channel copies of the target's bf16 dense layers (bf16 freed before KV sizing) + the fused expert gate.\n"
    "    if (__import__(\"os\").environ.get(\"RADIANCE_MOE_DENSE_FP8\", \"0\").strip() == \"1\"\n"
    "            or __import__(\"os\").environ.get(\"RADIANCE_MOE_GATE_FIX\", \"0\").strip() == \"1\"):\n"
    "        import radiance_moe_densefp8  # fail loudly: asked for, must be present\n"
    "\n"
    "        with torch.no_grad():\n"
    "            radiance_moe_densefp8.convert(model)\n"
    "\n"
    "    # Needed for torchao model reloading via model.reload_weights\n"
    "    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`\n"
    "    if model_config.quantization == \"torchao\":\n",
    "radiance: RADIANCE_MOE_DENSE_FP8 / RADIANCE_MOE_GATE_FIX (patch_moe_densefp8.py",
    "moe dense fp8: convert hook in process_weights_after_loading",
)

# 2) the fused expert gate in Qwen2MoeMLP (shared by Qwen3-Next / Qwen3.5 MoE and their MTP drafters)
Q = SP / "vllm/model_executor/models/qwen2_moe.py"
apply(
    Q,
    "import torch.nn.functional as F\nfrom torch import nn\n",
    "import torch.nn.functional as F\nfrom torch import nn\n"
    "\n"
    "# radiance: RADIANCE_MOE_GATE_FIX module flag (patch_moe_densefp8.py, moe-densefp8/)\n"
    "_RADIANCE_GATE_FIX = __import__(\"os\").environ.get(\"RADIANCE_MOE_GATE_FIX\", \"0\").strip() == \"1\"\n"
    "if _RADIANCE_GATE_FIX:\n"
    "    import radiance_moe_densefp8 as _radiance_densefp8  # fail loudly: asked for, must be present\n",
    "radiance: RADIANCE_MOE_GATE_FIX module flag",
    "moe gate fix: module flag in qwen2_moe.py",
)
apply(
    Q,
    "        if self.expert_gate is not None:\n"
    "            out = F.sigmoid(self.expert_gate(x)[0]) * out\n",
    "        if self.expert_gate is not None:\n"
    "            # radiance: RADIANCE_MOE_GATE_FIX - one fused kernel instead of hipBLASLt GEMV + sigmoid + mul\n"
    "            if _RADIANCE_GATE_FIX and _radiance_densefp8.gate_ok(x, self.expert_gate.weight, out):\n"
    "                out = torch.ops.vllm.radiance_gate_mul(x, self.expert_gate.weight, out)\n"
    "            else:\n"
    "                out = F.sigmoid(self.expert_gate(x)[0]) * out\n",
    "radiance: RADIANCE_MOE_GATE_FIX - one fused kernel",
    "moe gate fix: fused expert gate in Qwen2MoeMLP.forward",
)
