#!/usr/bin/env python3
"""Native MXFP4 (OCP micro-scaling) linear GEMM on gfx1201, for Quark W4A4 checkpoints
such as amd/Qwen3.8-27B-Quark-AWQ-MXFP4.

Stock vLLM emulates MXFP4 on this card: QuarkOCP_MX.apply_weights materialises the whole weight
tensor in bf16 (dequant_mxfp4) on every forward, then calls F.linear. Two things stand in the way
of the real kernel, and neither of them is the compiler:

  1. `RocmPlatform.supports_mx()` allowlists gfx95 (CDNA4), so `self.emulate` is forced True.
     But Triton 3.6 *does* lower `tl.dot_scaled` on gfx12x -- it upconverts the e2m1 operands and
     runs bf16 WMMA -- which is exactly what aiter's `gemm_afp4wfp4` rides on. Measured on gfx1201:
     bit-identical output to the emulated path (the activation quant is the same either way) and
     ~2.4-4x faster at decode shapes.
  2. vLLM 0.26.0 imports the aiter fp4 GEMM from `aiter.ops.triton.gemm_afp4wfp4`, a module path
     that aiter 0.1.17 moved to `aiter.ops.triton.gemm.basic.gemm_afp4wfp4`. Left alone, the native
     branch would raise ImportError the moment it was reached. (`aiter.ops.triton.quant` still
     re-exports dynamic_mxfp4_quant, so that import is left as-is.)

aiter's own `arch_info.is_fp4_avail()` allowlists gfx950/gfx1250 too; it is relaxed in the same
place, lazily, inside the op body -- so no aiter import is added at plugin-load time, where it
would initialise HIP early and force the engine core to spawn instead of fork.

The native kernel is not a win at every batch size: emulation amortises its single bf16 dequant
across many rows and then runs a fast bf16 GEMM, so it overtakes the fp4 kernel past M~256.
Measured (gate_up 17408x5120, speedup vs emulation): 6.11x at M=16, 4.71x at M=32, 2.52x at M=64,
1.88x at M=128, 1.13x at M=256, 0.85x at M=1024. So apply_weights dispatches on M and hands large
batches back to emulation, which keeps prefill from regressing. RADIANCE_MXFP4_MAX_M tunes the
crossover (default 256).

A second, separate opt-in (RADIANCE_MXFP4_W4A8=1) routes large-M linears to a hand-written fp8-WMMA
kernel. Triton will not emit gfx1201's fp8 matrix instruction -- measured register-resident, fp8 WMMA
runs 325.2 TFLOP/s vs f16's 160.2, while Triton's own fp8 tl.dot manages only 43.3 because it
upconverts -- so the 2x is only reachable by hand. Measured against the tuned aiter path it replaces:
1.47-2.26x faster AND 4.2x more accurate (relative error 0.0265 vs 0.1119 against exact arithmetic),
because fp8 activations beat the mxfp4 ones aiter quantizes to. It is opt-in anyway, since it makes
the layer W4A8 rather than the checkpoint's declared W4A4.

Gated by RADIANCE_MXFP4=1 (default off). With it unset the checkpoint still loads and serves,
just on the stock emulated path, and no other quantization scheme is touched.

Tiles come from mxfp4-configs/gfx1201-GEMM-AFP4WFP4.json, which the Dockerfile drops into aiter's
config dir: every band there pins matrix_instr_nonkdim to 16. aiter's gfx1250 table uses 32 for
M>=64, and gfx1201's WMMA is 16x16x16 only, so those bands fail to compile with
"no matching matrix core intrinsic due to unsupported element type: A='bf16' B='bf16' C='f32'".
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
Q = SP / "vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py"

IMPORT_ANCHOR = (
    "        from aiter.ops.triton.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffled_weight_scales,\n"
    "        )\n"
)
IMPORT_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): aiter 0.1.17 moved this module ---\n"
    "        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffled_weight_scales,\n"
    "        )\n"
    "        # aiter allowlists gfx950/gfx1250 for fp4; gfx1201 lowers tl.dot_scaled correctly\n"
    "        # (verified bit-identical against dequant_mxfp4 + F.linear), so relax the assert.\n"
    "        # Done here, lazily, to keep aiter out of the plugin-load import graph.\n"
    "        import aiter.ops.triton.utils._triton.arch_info as _radiance_arch\n"
    "        if not _radiance_arch.is_fp4_avail():\n"
    "            _radiance_arch.is_fp4_avail = lambda: True\n"
)

EMULATE_ANCHOR = (
    "        # TODO: integrate (or test) mixed-precision kernel.\n"
    "        self.emulate = not current_platform.supports_mx() or (\n"
    '            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"\n'
    "        )\n"
)
EMULATE_NEW = (
    "        # TODO: integrate (or test) mixed-precision kernel.\n"
    "        # --- radiance (patch_quark_mxfp4.py): gfx1201 native MXFP4, RADIANCE_MXFP4=1 ---\n"
    "        # supports_mx() is CDNA4-only, but Triton 3.6 lowers tl.dot_scaled on gfx12x.\n"
    "        # Only the mxfp4 x mxfp4 case is claimed; mxfp6 and mixed dtypes stay emulated.\n"
    "        _radiance_mx = current_platform.supports_mx()\n"
    "        if not _radiance_mx:\n"
    "            import os as _os\n"
    '            if _os.environ.get("RADIANCE_MXFP4", "0") == "1":\n'
    "                from vllm.platforms.rocm import on_gfx12x\n"
    "                _radiance_mx = bool(on_gfx12x())\n"
    "                if _radiance_mx:\n"
    "                    logger.warning_once(\n"
    '                        "[radiance] native MXFP4 enabled on gfx12x "\n'
    '                        "(aiter gemm_afp4wfp4 via tl.dot_scaled); "\n'
    '                        "the emulation notice below does not apply to mxfp4 x mxfp4 layers"\n'
    "                    )\n"
    "        self.emulate = not _radiance_mx or (\n"
    '            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"\n'
    "        )\n"
)

W4A8_ANCHOR = (
    "        self.rocm_use_aiter_fp4_asm_gemm = (\n"
    "            rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()\n"
    "        )\n"
)
W4A8_NEW = (
    "        self.rocm_use_aiter_fp4_asm_gemm = (\n"
    "            rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()\n"
    "        )\n"
    "\n"
    "        # --- radiance (patch_quark_mxfp4.py): resolve the W4A8 fp8-WMMA path once ---\n"
    "        # Importing the HIP extension inside apply_weights would break the torch.compile\n"
    "        # graph; importing at module scope would initialise HIP during config parsing in\n"
    "        # the parent process, forcing the engine core to spawn instead of fork.\n"
    "        self._radiance_w4a8 = None\n"
    "        if not self.emulate:\n"
    "            try:\n"
    "                import radiance_mxfp4 as _rmx\n"
    "                if _rmx.ENABLED:\n"
    "                    self._radiance_w4a8 = _rmx\n"
    "            except Exception as _e:\n"
    "                import sys as _s\n"
    '                _s.stderr.write(f"[radiance.mxfp4] w4a8 unavailable: {_e!r}\\n")\n'
)


CONST_ANCHOR = 'logger = init_logger(__name__)\n'
CONST_NEW = 'logger = init_logger(__name__)\n\n# radiance (patch_quark_mxfp4.py): batch size above which the emulated path is faster.\nimport os as _radiance_os\n_RADIANCE_MXFP4_MAX_M = int(_radiance_os.environ.get("RADIANCE_MXFP4_MAX_M", "256"))\n_RADIANCE_W4A8_MIN_M = int(\n    _radiance_os.environ.get("RADIANCE_MXFP4_W4A8_MIN_M", "256")\n)\n_RADIANCE_W4A8_ON = _radiance_os.environ.get("RADIANCE_MXFP4_W4A8", "0") == "1"\n'

PW_ANCHOR = (
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)\n"
)
PW_NEW = (
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)\n"
    "        # --- radiance (patch_quark_mxfp4.py): decide W4A8 eligibility ONCE, here ---\n"
    "        # The forward runs inside torch.compile, so the gate in front of the kernel must be\n"
    "        # a plain bool plus an integer shape compare. Doing the shape validation per call\n"
    "        # through a Python helper broke the graph at every linear and cost 33% of decode.\n"
    "        self._radiance_w4a8_ok = False\n"\
    "        # every layer needs the attribute: apply_weights passes it unconditionally\n"\
    "        # when RADIANCE_MXFP4_W4A8=1, and the op itself decides whether to use it\n"\
    "        layer.radiance_wref = torch.nn.Parameter(\n"\
    "            torch.zeros(1, dtype=torch.uint8, device=layer.weight.device),\n"\
    "            requires_grad=False,\n"\
    "        )\n"
)
PW_TAIL_ANCHOR = (
    "            else:\n"
    "                layer.weight_scale = torch.nn.Parameter(\n"
    "                    layer.weight_scale.data.T.contiguous(), requires_grad=False\n"
    "                )\n"
)
PW_TAIL_NEW = (
    "            else:\n"
    "                layer.weight_scale = torch.nn.Parameter(\n"
    "                    layer.weight_scale.data.T.contiguous(), requires_grad=False\n"
    "                )\n"
    "        if self._radiance_w4a8 is not None:\n"
    "            self._radiance_w4a8_ok = self._radiance_w4a8.layer_is_supported(\n"
    "                layer, int(layer.weight.shape[1]) * 2\n"
    "            )\n"
    "            if self._radiance_w4a8_ok:\n"
    "                layer.radiance_wref = torch.nn.Parameter(\n"
    "                    self._radiance_w4a8.make_row_ref(layer.weight_scale.data),\n"
    "                    requires_grad=False,\n"
    "                )\n"
)

APPLY_ANCHOR = '        if self.emulate:\n            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)\n            qdq_x = self.quant_dequant_func(x)\n            return F.linear(qdq_x, dq_w, bias)\n        y = torch.ops.vllm.gemm_with_dynamic_quant(\n'
APPLY_NEW = "        if self.emulate:\n            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)\n            qdq_x = self.quant_dequant_func(x)\n            return F.linear(qdq_x, dq_w, bias)\n        # --- radiance (patch_quark_mxfp4.py): M-keyed dispatch ---\n        # The fp4 kernel wins 2.5-6x at decode shapes and loses past M~256, where\n        # emulation's one-off bf16 dequant is amortised over enough rows to pay for\n        # itself. Hand large batches back so prefill cannot regress. A plain shape\n        # branch, which is dynamo-traceable (same pattern as radiance_kernels.py).\n        if _RADIANCE_W4A8_ON:\n            # the M threshold lives INSIDE the op: a shape branch here is\n            # data-dependent under vLLM's dynamic token dim and splits the graph\n            # at every linear, costing ~30% of decode\n            _y = torch.ops.radiance.mxfp4_linear(\n                x, layer.weight, layer.weight_scale, layer.radiance_wref,\n                self._radiance_w4a8_ok, self.rocm_use_aiter_fp4_asm_gemm,\n            )\n            return _y + bias if bias is not None else _y\n        if x.shape[0] > _RADIANCE_MXFP4_MAX_M:\n            # process_weights_after_loading stored weight_scale TRANSPOSED for the native\n            # path ([K/32, N]); dequant_mxfp4 wants [N, K/32]. .contiguous() is required,\n            # not tidiness: given a non-contiguous transposed view dequant_mxfp4 returns\n            # silently wrong values rather than raising (measured max abs err 4.7e3).\n            dq_w = self.dequant_func(\n                layer.weight, layer.weight_scale.T.contiguous(), x.dtype\n            )\n            return F.linear(self.quant_dequant_func(x), dq_w, bias)\n        y = torch.ops.vllm.gemm_with_dynamic_quant(\n"


def main():
    apply(Q, IMPORT_ANCHOR, IMPORT_NEW,
          "aiter 0.1.17 moved this module", "quark mxfp4 aiter import path + arch gate")
    apply(Q, EMULATE_ANCHOR, EMULATE_NEW,
          "gfx1201 native MXFP4", "quark mxfp4 emulate gate -> RADIANCE_MXFP4")
    apply(Q, CONST_ANCHOR, CONST_NEW,
          "_RADIANCE_MXFP4_MAX_M", "quark mxfp4 large-M threshold constant")
    apply(Q, PW_ANCHOR, PW_NEW,
          "decide W4A8 eligibility ONCE", "quark mxfp4 W4A8 eligibility default")
    apply(Q, PW_TAIL_ANCHOR, PW_TAIL_NEW,
          "layer.radiance_w4a8_ok = self._radiance_w4a8.layer_is_supported",
          "quark mxfp4 W4A8 eligibility computed at load")
    apply(Q, W4A8_ANCHOR, W4A8_NEW,
          "resolve the W4A8 fp8-WMMA path once", "quark mxfp4 W4A8 module resolution")
    apply(Q, APPLY_ANCHOR, APPLY_NEW,
          "M-keyed dispatch", "quark mxfp4 M-keyed dispatch + W4A8 fp8-WMMA route")


if __name__ == "__main__":
    main()
