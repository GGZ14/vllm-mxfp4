#!/usr/bin/env python3
"""Padded CUDA-graph rows reuse row 0's experts in the MoE router (radiance-decode3/, RADIANCE_MOE_PAD_ROUTE=1).

Copies moe-padroute/radiance_moe_padroute.py into site-packages and patches three vLLM files:
  1. fused_moe/runner/moe_runner.py  MoERunner._forward_impl: after the router GEMV, radiance_moe_padroute.fix(logits)
     copies row 0's logits into the padded rows (one tiny kernel, captured into the graphs; real rows untouched).
  2. v1/worker/gpu_model_runner.py   execute_model: set_real(num_tokens_unpadded, num_tokens_padded) right before the
     target forward.
  3. v1/spec_decode/llm_base_proposer.py  propose: set_real(...) right before the drafter's first pass and before each
     loop pass.
Inert unless RADIANCE_MOE_PAD_ROUTE=1 (module flags read at import; nothing is imported otherwise). Asked for and
missing is fatal. Usage: python3 patch_moe_padroute.py [SITE_PACKAGES]. Idempotent (sentinels); fails before writing
on drift (_patchlib.apply: unique anchor, ast-checked).
"""
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])

src = HERE / "moe-padroute" / "radiance_moe_padroute.py"
if not src.exists():
    raise SystemExit(f"  FAIL  moe pad-route: {src} missing")
shutil.copyfile(src, SP / "radiance_moe_padroute.py")
print(f"  OK    moe pad-route: installed {SP / 'radiance_moe_padroute.py'}")

FLAG = ("\n# radiance: RADIANCE_MOE_PAD_ROUTE module flag (patch_moe_padroute.py, radiance-decode3/)\n"
        "_RADIANCE_PAD_ROUTE = __import__(\"os\").environ.get(\"RADIANCE_MOE_PAD_ROUTE\", \"0\").strip() == \"1\"\n"
        "if _RADIANCE_PAD_ROUTE:\n"
        "    import radiance_moe_padroute as _radiance_padroute  # fail loudly: asked for, must be present\n")

# 1) the fix in the MoE runner (target a16w4 monolithic lane and drafter bf16 modular lane both pass through here)
R = SP / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
apply(R, "import torch.nn.functional as F\n", "import torch.nn.functional as F\n" + FLAG,
      "radiance: RADIANCE_MOE_PAD_ROUTE module flag", "moe pad-route: module flag in moe_runner.py")
apply(
    R,
    "            else:\n"
    "                router_logits, _ = self.gate(hidden_states)\n"
    "\n"
    "        with self._sequence_parallel_context():\n",
    "            else:\n"
    "                router_logits, _ = self.gate(hidden_states)\n"
    "            # radiance: RADIANCE_MOE_PAD_ROUTE - padded graph rows take row 0's logits (same experts, no extra bytes)\n"
    "            if _RADIANCE_PAD_ROUTE:\n"
    "                router_logits = _radiance_padroute.fix(router_logits)\n"
    "\n"
    "        with self._sequence_parallel_context():\n",
    "radiance: RADIANCE_MOE_PAD_ROUTE - padded graph rows",
    "moe pad-route: fix after the router GEMV in MoERunner._forward_impl",
)

# 2) the target forward's real-row count
G = SP / "vllm/v1/worker/gpu_model_runner.py"
apply(G, "\nimport torch\n", "\nimport torch\n" + FLAG,
      "radiance: RADIANCE_MOE_PAD_ROUTE module flag", "moe pad-route: module flag in gpu_model_runner.py")
apply(
    G,
    "        if self.eplb_state is not None:\n"
    "            self.eplb_state.prepare_forward(\n"
    "                self.model_config,\n"
    "                num_tokens_unpadded,\n"
    "                ubatch_slices_padded,\n"
    "            )\n",
    "        if self.eplb_state is not None:\n"
    "            self.eplb_state.prepare_forward(\n"
    "                self.model_config,\n"
    "                num_tokens_unpadded,\n"
    "                ubatch_slices_padded,\n"
    "            )\n"
    "        # radiance: RADIANCE_MOE_PAD_ROUTE - real rows of this (possibly padded) target forward, as a device value\n"
    "        if _RADIANCE_PAD_ROUTE:\n"
    "            _radiance_padroute.set_real(num_tokens_unpadded, num_tokens_padded, \"target\")\n",
    "radiance: RADIANCE_MOE_PAD_ROUTE - real rows of this (possibly padded) target",
    "moe pad-route: real-row count before the target forward",
)

# 3) the drafter's passes
P = SP / "vllm/v1/spec_decode/llm_base_proposer.py"
apply(P, "\nimport torch\n", "\nimport torch\n" + FLAG,
      "radiance: RADIANCE_MOE_PAD_ROUTE module flag", "moe pad-route: module flag in llm_base_proposer.py")
apply(
    P,
    "        if self.eplb_state is not None:\n"
    "            self.eplb_state.prepare_forward(\n"
    "                self.draft_model_config,\n"
    "                num_tokens,\n"
    "            )\n",
    "        if self.eplb_state is not None:\n"
    "            self.eplb_state.prepare_forward(\n"
    "                self.draft_model_config,\n"
    "                num_tokens,\n"
    "            )\n"
    "        # radiance: RADIANCE_MOE_PAD_ROUTE - drafter first pass real rows\n"
    "        if _RADIANCE_PAD_ROUTE:\n"
    "            _radiance_padroute.set_real(num_tokens, num_input_tokens, \"drafter\")\n",
    "radiance: RADIANCE_MOE_PAD_ROUTE - drafter first pass",
    "moe pad-route: real-row count before the drafter's first pass",
)
apply(
    P,
    "            if self.eplb_state is not None:\n"
    "                self.eplb_state.prepare_forward(\n"
    "                    self.draft_model_config,\n"
    "                    batch_size,\n"
    "                )\n",
    "            if self.eplb_state is not None:\n"
    "                self.eplb_state.prepare_forward(\n"
    "                    self.draft_model_config,\n"
    "                    batch_size,\n"
    "                )\n"
    "            # radiance: RADIANCE_MOE_PAD_ROUTE - drafter loop pass real rows\n"
    "            if _RADIANCE_PAD_ROUTE:\n"
    "                _radiance_padroute.set_real(batch_size, input_batch_size, \"drafter\")\n",
    "radiance: RADIANCE_MOE_PAD_ROUTE - drafter loop pass",
    "moe pad-route: real-row count before each drafter loop pass",
)
