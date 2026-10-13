#!/usr/bin/env python3
"""W4A8 MoE experts on gfx1201: optional fp8-WMMA expert GEMMs for prefill-sized calls (MOE-GFX1201.md).

With RADIANCE_MOE_W4A8=1, vLLM's aiter_triton_kernel_w4a16_moe_forward (the a16w4 lane that
patch_gfx12_aiter_a16w4.py enables on gfx12) hands calls with >= RADIANCE_MOE_W4A8_MIN_TOKENS tokens
(default 1025, where AITER routing's block_m reaches 64) to radiance_moe_w4a8.moe_forward: per-token fp8
activations, the grouped MXFP4 x FP8 WMMA kernel for w13 (+ SiLU*up) and w2 (+ gammas) on the same routing,
weights and sorted-order buffers, then the same reduce_grouped. Smaller calls -- decode, MTP verify, short
prompts, every CUDA-graph capture size -- stay on a16w4 (at those sizes the expert GEMMs are weight-bandwidth
bound and fp8 activations buy nothing). Only plain SiLU without clamp, bias or router-weight-on-input goes
there, which is what the Qwen3.5/3.6-35B-A3B Quark checkpoints are.

Default off: with the knob unset nothing is imported and the lane runs exactly as before. Asked for and missing
(radiance_moe_w4a8 not built into site-packages by moe-w4a8/build.sh) is fatal at import, never a silent fallback.

Requires patch_gfx12_aiter_a16w4.py first (anchors on code it inserts). Usage: python3 patch_moe_w4a8.py
Idempotent (two applies, each with its own sentinel); fails before writing on source drift.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _patchlib import apply  # noqa: E402

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sysconfig.get_paths()["purelib"])
F = SP / "vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py"

# 1. module-level knob + import, next to the gfx12 tile-table flag that patch_gfx12_aiter_a16w4.py inserts
apply(
    F,
    "_RADIANCE_A16W4_TILES = False\n",
    "_RADIANCE_A16W4_TILES = False\n"
    "\n"
    "# radiance: RADIANCE_MOE_W4A8=1 (patch_moe_w4a8.py, moe-w4a8/) routes prefill-sized expert GEMM\n"
    "# calls to the fp8-WMMA W4A8 grouped kernel. Default off: nothing imported.\n"
    "_RADIANCE_W4A8 = None\n"
    "if __import__(\"os\").environ.get(\"RADIANCE_MOE_W4A8\", \"0\") == \"1\":\n"
    "    import radiance_moe_w4a8 as _RADIANCE_W4A8  # fail loudly: asked for, must be present\n"
    "\n"
    "    __import__(\"sys\").stderr.write(\n"
    "        \"[radiance.moe] W4A8 expert GEMMs ON (fp8 WMMA, calls >= %d tokens; cfg w13 %s w2 %s)\\n\"\n"
    "        % (_RADIANCE_W4A8.MIN_TOKENS, _RADIANCE_W4A8.CFG13, _RADIANCE_W4A8.CFG2)\n"
    "    )\n",
    "_RADIANCE_W4A8 = None",
    "moe w4a8: knob + import",
)

# 2. dispatch, right before the a16w4 w13 call (after routing, weights, gammas and the gfx12 limit are set)
apply(
    F,
    "    # SILU: silu(gate) * up — same kernel, just no \"+1\" residual in swiglu.\n"
    "    swiglu_add_residual = activation != MoEActivation.SILU\n"
    "\n"
    "    intermediate = moe_gemm_a16w4(\n",
    "    # SILU: silu(gate) * up — same kernel, just no \"+1\" residual in swiglu.\n"
    "    swiglu_add_residual = activation != MoEActivation.SILU\n"
    "\n"
    "    # radiance: RADIANCE_MOE_W4A8 (patch_moe_w4a8.py) - prefill-sized calls on the fp8-WMMA W4A8 kernel\n"
    "    if (\n"
    "        _RADIANCE_W4A8 is not None\n"
    "        and activation == MoEActivation.SILU\n"
    "        and _RADIANCE_W4A8.supported(\n"
    "            hidden_states,\n"
    "            routing_data,\n"
    "            quant_config.w1_bias,\n"
    "            quant_config.w2_bias,\n"
    "            swiglu_limit,\n"
    "            apply_router_weight_on_input,\n"
    "        )\n"
    "    ):\n"
    "        return _RADIANCE_W4A8.moe_forward(\n"
    "            hidden_states, w1_data, w2_data, w1_wscale, w2_wscale,\n"
    "            routing_data, gather_idx, scatter_idx, gammas,\n"
    "        )\n"
    "\n"
    "    intermediate = moe_gemm_a16w4(\n",
    "radiance: RADIANCE_MOE_W4A8 (patch_moe_w4a8.py) - prefill-sized calls",
    "moe w4a8: dispatch in aiter_triton_kernel_w4a16_moe_forward",
)
