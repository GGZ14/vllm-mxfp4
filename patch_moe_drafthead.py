#!/usr/bin/env python3
"""Draft-only compressed lm_head for the MoE lane's MTP drafter (moe-drafthead/, MOE-GFX1201.md).

With RADIANCE_MOE_DRAFT_HEAD=fp8|int4|int2, Qwen3_5MTP.load_weights (the MTP drafter of Qwen3.5 / 3.6 dense and
MoE checkpoints; Qwen3_5MoeMTP inherits it) ends by calling radiance_moe_drafthead.arm(self): the drafter's own
LogitsProcessor gets a compressed copy of lm_head, while the target model's LogitsProcessor -- the verify pass --
keeps the bf16 head, so accepted tokens stay exact. Unset / off: nothing is imported, the drafter runs as before.
Asked for and missing (radiance_moe_drafthead not installed) is fatal at load, never a silent fallback.

This script copies moe-drafthead/radiance_moe_drafthead.py into site-packages and patches qwen3_5_mtp.py.
Usage: python3 patch_moe_drafthead.py [SITE_PACKAGES]. Idempotent (sentinel); fails before writing on drift.
"""
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])
F = SP / "vllm/model_executor/models/qwen3_5_mtp.py"

src = HERE / "moe-drafthead" / "radiance_moe_drafthead.py"
if not src.exists():
    raise SystemExit(f"  FAIL  moe draft head: {src} missing")
shutil.copyfile(src, SP / "radiance_moe_drafthead.py")
print(f"  OK    moe draft head: installed {SP / 'radiance_moe_drafthead.py'}")

apply(
    F,
    "        loader = AutoWeightsLoader(self)\n"
    "        return loader.load_weights(remap_weight_names(weights))\n"
    "\n"
    "\n"
    "class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):\n",
    "        loader = AutoWeightsLoader(self)\n"
    "        loaded = loader.load_weights(remap_weight_names(weights))\n"
    "        # radiance: RADIANCE_MOE_DRAFT_HEAD (patch_moe_drafthead.py, moe-drafthead/) - the drafter's own\n"
    "        # LogitsProcessor gets a compressed copy of lm_head; the target's (verify) keeps the bf16 head.\n"
    "        if __import__(\"os\").environ.get(\"RADIANCE_MOE_DRAFT_HEAD\", \"off\").strip().lower() not in (\"\", \"off\", \"0\"):\n"
    "            import radiance_moe_drafthead  # fail loudly: asked for, must be present\n"
    "\n"
    "            radiance_moe_drafthead.arm(self)\n"
    "        return loaded\n"
    "\n"
    "\n"
    "class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):\n",
    "radiance: RADIANCE_MOE_DRAFT_HEAD (patch_moe_drafthead.py",
    "moe draft head: arm in Qwen3_5MTP.load_weights",
)
