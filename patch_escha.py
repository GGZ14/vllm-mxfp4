#!/usr/bin/env python3
"""Register the radiance `escha` quantization config early enough for ModelConfig to see it.

Identical timing problem to patch_autoround.py, and identical solution: the decorator has to run
before `ModelConfig._verify_quantization` resolves the checkpoint's `quant_method`, and after vLLM
is importable. A throwaway import beforehand is lost with the process; sitecustomize.py is shadowed
by the image's own copy; and vLLM's plugin entry points load during engine init, which is after
ModelConfig is built during argument parsing.

Unlike auto-round there is no built-in method fighting for the name -- "escha" is unknown to vLLM
entirely, so without this the engine simply reports an unsupported quantization method and exits.

Gated on RADIANCE_ESCHA so a build not serving an escha checkpoint imports nothing.
"""
import pathlib

from _patchlib import apply

SP = pathlib.Path("/opt/vllm/lib/python3.12/site-packages")
QI = SP / "vllm/model_executor/layers/quantization/__init__.py"

ANCHOR = '''__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]'''

NEW = '''__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

# radiance: register the gfx1201 escha (EXL3 trellis) W2 config. Must happen before ModelConfig
# resolves the checkpoint's quant_method; "escha" is unknown to vLLM, so without this the engine
# exits during argument parsing. Import failures are reported and swallowed: a broken side module
# must not take down every other quantization method.
import os as _radiance_escha_os  # noqa: E402

if _radiance_escha_os.environ.get("RADIANCE_ESCHA", "0") == "1":
    try:
        import radiance_escha as _radiance_escha_mod  # noqa: E402
        _radiance_escha_mod.register()
    except Exception as _radiance_escha_exc:  # pragma: no cover
        import sys as _radiance_escha_sys  # noqa: E402
        _radiance_escha_sys.stderr.write(
            f"[radiance.escha] registration FAILED: {_radiance_escha_exc!r}\\n")'''

SENTINEL = "radiance: register the gfx1201 escha (EXL3 trellis) W2 config"


def main():
    apply(QI, ANCHOR, NEW, SENTINEL, "escha: register the quantization config")


if __name__ == "__main__":
    main()
