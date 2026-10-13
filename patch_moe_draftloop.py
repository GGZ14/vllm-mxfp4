#!/usr/bin/env python3
"""Less host time in the MTP drafter loop (radiance-gaps/, RADIANCE_MOE_DRAFT_GRAPH=1, RADIANCE_MOE_DRAFT_OVERLAP=1).

Copies moe-draftloop/radiance_moe_draftloop.py into site-packages and patches:
  1. v1/spec_decode/llm_base_proposer.py  propose loop: right before each loop pass's forward, resolve the deferred
     dynamic-draft decision (break on stop), and run the forward through radiance_moe_draftloop.forward_ctx, which
     replays one captured graph per batch size (GRAPH) or falls through to the stock set_forward_context + model call.
  2. radiance_draft.py (image site-packages, the dynamic-draft controller): greedy_sample's per-slot decision becomes
     _rad_decide(); with OVERLAP the blocking packed.cpu() becomes a pinned non-blocking copy and the decision is
     deferred to the loop hook above or the draft postprocess.
Inert unless one of the two env knobs is 1 (module flags read at import; nothing imported otherwise): with both at 0
the loop runs the stock code and greedy_sample calls _rad_decide(self, packed.cpu().numpy(), j, B) inline, the same
statements in the same order. Usage: python3 patch_moe_draftloop.py [SITE_PACKAGES]. Idempotent (sentinels); fails
before writing on drift (_patchlib.apply: unique anchor, ast-checked). Apply after patch_moe_padroute.py.
"""
import shutil
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])

src = HERE / "moe-draftloop" / "radiance_moe_draftloop.py"
if not src.exists():
    raise SystemExit(f"  FAIL  moe draft loop: {src} missing")
shutil.copyfile(src, SP / "radiance_moe_draftloop.py")
print(f"  OK    moe draft loop: installed {SP / 'radiance_moe_draftloop.py'}")

FLAG = ("\n# radiance-gaps: RADIANCE_MOE_DRAFT_GRAPH / RADIANCE_MOE_DRAFT_OVERLAP module flag (patch_moe_draftloop.py)\n"
        "_RADIANCE_DRAFTLOOP = any(__import__(\"os\").environ.get(_k, \"0\").strip() == \"1\"\n"
        "                          for _k in (\"RADIANCE_MOE_DRAFT_GRAPH\", \"RADIANCE_MOE_DRAFT_OVERLAP\"))\n"
        "if _RADIANCE_DRAFTLOOP:\n"
        "    import radiance_moe_draftloop as _radiance_draftloop  # fail loudly: asked for, must be present\n")

P = SP / "vllm/v1/spec_decode/llm_base_proposer.py"
apply(P, "\nimport torch\n", "\nimport torch\n" + FLAG,
      "radiance-gaps: RADIANCE_MOE_DRAFT_GRAPH / RADIANCE_MOE_DRAFT_OVERLAP module flag",
      "moe draft loop: module flag in llm_base_proposer.py")
apply(
    P,
    "            with set_forward_context(\n"
    "                per_layer_attn_metadata,\n"
    "                self.vllm_config,\n"
    "                num_tokens=input_batch_size,\n"
    "                num_tokens_across_dp=batch_size_across_dp,\n"
    "                cudagraph_runtime_mode=cudagraph_runtime_mode,\n"
    "                slot_mapping=self._get_slot_mapping(input_batch_size),\n"
    "            ):\n"
    "                ret_hidden_states = self.model(**model_kwargs)\n",
    "            # radiance-gaps: deferred dynamic-draft decision (OVERLAP) and one graph per loop pass (GRAPH)\n"
    "            if _RADIANCE_DRAFTLOOP and _radiance_draftloop.before_forward(self):\n"
    "                break\n"
    "            with (_radiance_draftloop.forward_ctx(\n"
    "                    self, per_layer_attn_metadata, common_attn_metadata, batch_size,\n"
    "                    input_batch_size, batch_size_across_dp, cudagraph_runtime_mode)\n"
    "                  if _RADIANCE_DRAFTLOOP else set_forward_context(\n"
    "                per_layer_attn_metadata,\n"
    "                self.vllm_config,\n"
    "                num_tokens=input_batch_size,\n"
    "                num_tokens_across_dp=batch_size_across_dp,\n"
    "                cudagraph_runtime_mode=cudagraph_runtime_mode,\n"
    "                slot_mapping=self._get_slot_mapping(input_batch_size),\n"
    "            )) as _rad_fw:\n"
    "                ret_hidden_states = (_rad_fw(model_kwargs) if _RADIANCE_DRAFTLOOP\n"
    "                                     else self.model(**model_kwargs))\n",
    "radiance-gaps: deferred dynamic-draft decision",
    "moe draft loop: loop pass hook in llm_base_proposer.py",
)

D = SP / "radiance_draft.py"
apply(
    D,
    "import numpy as np\n",
    "import numpy as np\n"
    "# radiance-gaps: RADIANCE_MOE_DRAFT_OVERLAP (patch_moe_draftloop.py)\n"
    "_rad_dl = None\n"
    "if os.environ.get(\"RADIANCE_MOE_DRAFT_OVERLAP\", \"0\").strip() == \"1\":\n"
    "    import radiance_moe_draftloop as _rad_dl  # fail loudly: asked for, must be present\n",
    "radiance-gaps: RADIANCE_MOE_DRAFT_OVERLAP (patch_moe_draftloop.py)",
    "moe draft loop: overlap flag in radiance_draft.py",
)
apply(
    D,
    "        packed[1] = draft_token_ids.to(torch.float32)\n"
    "        arr = packed.cpu().numpy()\n"
    "        cfn = arr[0]; mtpn = arr[1].astype(np.int64)\n",
    "        packed[1] = draft_token_ids.to(torch.float32)\n"
    "        # radiance-gaps: with RADIANCE_MOE_DRAFT_OVERLAP the D2H does not block; the decision runs later\n"
    "        if _rad_dl is not None:\n"
    "            _rad_dl.defer_d2h(self, packed, lambda arr, _j=j, _B=B: _rad_decide(self, arr, _j, _B))\n"
    "            return draft_token_ids\n"
    "        _rad_decide(self, packed.cpu().numpy(), j, B)\n"
    "        return draft_token_ids\n"
    "\n"
    "    def _rad_decide(self, arr, j, B):\n"
    "        cfn = arr[0]; mtpn = arr[1].astype(np.int64)\n",
    "def _rad_decide(self, arr, j, B):",
    "moe draft loop: per-slot decision as _rad_decide in radiance_draft.py",
)
apply(
    D,
    "        if st[\"stopped\"].all():\n"
    "            self._radiance_stop = True\n"
    "        return draft_token_ids\n",
    "        if st[\"stopped\"].all():\n"
    "            self._radiance_stop = True\n"
    "        return None  # radiance-gaps: end of _rad_decide\n",
    "return None  # radiance-gaps: end of _rad_decide",
    "moe draft loop: _rad_decide tail in radiance_draft.py",
)
apply(
    D,
    "        self._radiance_gate = {}                          # per-slot gating state (filled at j==0)\n",
    "        self._radiance_gate = {}                          # per-slot gating state (filled at j==0)\n"
    "        self._radiance_resolve = None                     # radiance-gaps: no deferred decision carries over\n",
    "radiance-gaps: no deferred decision carries over",
    "moe draft loop: reset the deferred decision in propose (radiance_draft.py)",
)
apply(
    D,
    "    d = runner.drafter\n"
    "    st = getattr(d, \"_radiance_gate\", None)\n",
    "    d = runner.drafter\n"
    "    if _rad_dl is not None:\n"
    "        _rad_dl.resolve_pending(d)  # radiance-gaps: the last slot's deferred decision\n"
    "    st = getattr(d, \"_radiance_gate\", None)\n",
    "radiance-gaps: the last slot's deferred decision",
    "moe draft loop: resolve the last decision in _postprocess_gpu (radiance_draft.py)",
)
