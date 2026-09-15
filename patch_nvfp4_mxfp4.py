"""NVFP4 checkpoints on gfx1201: route compressed-tensors NVFP4 (and optionally FP8 per-channel)
linears to radiance_nvfp4's load-time MXFP4 requantization, which lands on the radiance W4A8 kernel.

Two hunks in compressed_tensors.py's _get_scheme_from_parts, both gated on RADIANCE_NVFP4_MXFP4=1
at call time so the stock scheme selection is untouched otherwise:
  1. the NVFP4 branch returns RadianceNvfp4ToMxfp4 instead of CompressedTensorsW4A4Fp4 (whose
     kernel list on ROCm ends in an emulation that would materialise bf16 weights per forward);
  2. the FP8 W8A8 branch returns RadianceFp8ChannelToMxfp4 when RADIANCE_NVFP4_FP8_LAYERS=mxfp4
     (never for lm_head), else the stock CompressedTensorsW8A8Fp8.
Also a one-shot conversion summary after weight loading, hooked next to the existing radiance
install in radiance_kernels (best effort).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
CT = SP / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"

NV_ANCHOR = (
    "            return CompressedTensorsW4A4Fp4()\n"
    "\n"
    "        if self._is_mxfp4(weight_quant):\n"
)
NV_NEW = (
    "            # --- radiance (patch_nvfp4_mxfp4.py): NVFP4 -> MXFP4 at load on gfx12x ---\n"
    "            import os as _radiance_os\n"
    "            if _radiance_os.environ.get(\"RADIANCE_NVFP4_MXFP4\", \"0\") == \"1\":\n"
    "                import radiance_nvfp4 as _radiance_nvfp4\n"
    "\n"
    "                _radiance_cls = _radiance_nvfp4.scheme_class()\n"
    "                if _radiance_cls is not None:\n"
    "                    return _radiance_cls()\n"
    "            return CompressedTensorsW4A4Fp4()\n"
    "\n"
    "        if self._is_mxfp4(weight_quant):\n"
)

FP8_ANCHOR = (
    "                if is_fp8_w8a8_supported:\n"
    "                    return CompressedTensorsW8A8Fp8(\n"
)
FP8_NEW = (
    "                if is_fp8_w8a8_supported:\n"
    "                    # --- radiance (patch_nvfp4_mxfp4.py): optional FP8 -> MXFP4 at load ---\n"
    "                    import os as _radiance_os\n"
    "                    if _radiance_os.environ.get(\"RADIANCE_NVFP4_MXFP4\", \"0\") == \"1\":\n"
    "                        import radiance_nvfp4 as _radiance_nvfp4\n"
    "\n"
    "                        _radiance_cls = _radiance_nvfp4.fp8_scheme_class(layer_name)\n"
    "                        if _radiance_cls is not None:\n"
    "                            return _radiance_cls()\n"
    "                    return CompressedTensorsW8A8Fp8(\n"
)


BF16_ANCHOR = (
    "            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n"
    "            input_tfms, output_tfms = get_linear_transform_schemes(\n"
)
BF16_NEW = (
    "            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n"
    "            # --- radiance (patch_nvfp4_mxfp4.py): optional bf16 -> MXFP4 for unmatched linears ---\n"
    "            if quant_scheme is None:\n"
    "                import os as _radiance_os\n"
    "                if _radiance_os.environ.get(\"RADIANCE_NVFP4_MXFP4\", \"0\") == \"1\":\n"
    "                    import radiance_nvfp4 as _radiance_nvfp4\n"
    "\n"
    "                    _radiance_cls = _radiance_nvfp4.bf16_scheme_class(prefix)\n"
    "                    if _radiance_cls is not None:\n"
    "                        quant_scheme = _radiance_cls()\n"
    "            input_tfms, output_tfms = get_linear_transform_schemes(\n"
)


def main():
    apply(CT, NV_ANCHOR, NV_NEW, "NVFP4 -> MXFP4 at load on gfx12x",
          "nvfp4: NVFP4 branch -> radiance requant scheme")
    apply(CT, BF16_ANCHOR, BF16_NEW, "optional bf16 -> MXFP4 for unmatched linears",
          "nvfp4: unmatched bf16 linears -> optional radiance requant scheme")
    apply(CT, FP8_ANCHOR, FP8_NEW, "optional FP8 -> MXFP4 at load",
          "nvfp4: FP8 branch -> optional radiance requant scheme")


if __name__ == "__main__":
    main()
