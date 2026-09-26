#!/usr/bin/env python3
"""Serve Quark W4A4 (MXFP4 weights + dynamic MXFP4 activations) MoE experts as weight-only on gfx12.

Every native W4A4 MoE kernel vLLM knows (AITER CK) is CDNA-only, so on gfx1201 the selector finds
nothing for `w_mxfp4_a_mxfp4` and QuarkOCP_MX_MoEMethod drops to EMULATION, which dequantizes the
expert weights on every forward (measured ~16 tok/s decode on amd/Qwen3.5-35B-A3B-MXFP4, one R9700).

With RADIANCE_MOE_W4A16=1 the scheme is rewritten to `w_mxfp4` right after it is derived: the
activation quant is dropped, so experts run with bf16 activations on the weight-only lanes
(`--moe-backend triton_unfused`, or the AITER a16w4 Triton lane). That is at least as accurate as
W4A4 (activations are no longer rounded to 4 bits), but not bit-identical to the emulation path.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
path = SP / "vllm/model_executor/layers/quantization/quark/quark_moe.py"

apply(
    path,
    "        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(\n"
    "            self.input_dtype, self.weight_dtype\n"
    "        )\n",
    "        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(\n"
    "            self.input_dtype, self.weight_dtype\n"
    "        )\n"
    "        # radiance: gfx12 has no native W4A4 MoE kernel; serve the experts weight-only\n"
    "        import os as _os\n"
    "        if (_os.environ.get('RADIANCE_MOE_W4A16', '0') == '1'\n"
    "                and self.ocp_mx_scheme == 'w_mxfp4_a_mxfp4'):\n"
    "            self.input_quant = None\n"
    "            self.input_dtype = None\n"
    "            self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(None, self.weight_dtype)\n"
    "            logger.info('radiance: W4A4 MoE experts served weight-only (%s)',\n"
    "                        self.ocp_mx_scheme)\n",
    "radiance: gfx12 has no native W4A4 MoE kernel",
    "quark moe: W4A4 experts served weight-only on gfx12",
)

# RADIANCE_MOE_BACKEND picks the MXFP4 MoE lane for these quantized expert layers only. The global
# `--moe-backend` flag also reaches unquantized MoE layers (the bf16 MTP drafter's), which reject
# the MXFP4 lane names (`triton_unfused` is "not supported for unquantized MoE").
apply(
    path,
    "        if self.ocp_mx_scheme == \"w_mxfp4\":\n"
    "            # W4A16: weight-only MXFP4\n"
    "            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)\n",
    "        if self.ocp_mx_scheme == \"w_mxfp4\":\n"
    "            # W4A16: weight-only MXFP4\n"
    "            # radiance: per-layer MXFP4 MoE lane, independent of --moe-backend\n"
    "            import dataclasses as _dc\n"
    "            _rb = _os.environ.get('RADIANCE_MOE_BACKEND', '')\n"
    "            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(\n"
    "                _dc.replace(moe, moe_backend=_rb) if _rb else moe)\n",
    "radiance: per-layer MXFP4 MoE lane",
    "quark moe: RADIANCE_MOE_BACKEND selects the MXFP4 lane",
)
