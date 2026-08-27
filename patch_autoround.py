#!/usr/bin/env python3
"""Register the radiance `auto-round` quantization config early enough for ModelConfig to see it.

The config itself lives in radiance_autoround.py and registers via
`@register_quantization_config("auto-round")`. The only hard part is WHEN that decorator runs: it
has to be before `ModelConfig._verify_quantization` resolves the checkpoint's `quant_method`, and
after vLLM is importable.

Three earlier hooks do not work, which is why this exists:

  * a throwaway `python3 -c "import radiance_autoround"` before the engine starts. The registry is
    process state, so it is gone by the time `vllm serve` runs.
  * `sitecustomize.py` dropped in site-packages. The radiance image already ships
    /usr/lib/python3.12/sitecustomize.py, which comes FIRST on sys.path and shadows it -- the file
    is simply never imported, silently.
  * vLLM's general-plugin entry points. Those load during engine/worker init, but ModelConfig is
    built during argument parsing, which is earlier.

Patching the end of the quantization package's __init__ is both early enough and late enough:
`register_quantization_config` is defined by that point, and anything importing the registry has
necessarily executed this line first.

Without it the engine dies before building a single layer. vLLM ships an INC (Intel Neural
Compressor) config whose `override_quantization_method` maps quant_method "auto-round" straight to
"inc", and INC refuses to run on ROCm:

    Value error, inc quantization is currently not supported in rocm.

With the config registered, ModelConfig probes CUSTOM methods before the built-in override list
("auto-round" is not in the QuantizationMethods literal), so AutoRoundConfig's own
`override_quantization_method` claims the checkpoint first -- and only claims 4-bit group-128
symmetric, which is what the kernel actually supports.

Gated on RADIANCE_AUTOROUND so a build that is not serving an AutoRound checkpoint imports nothing.
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

# radiance: register the gfx1201 AutoRound int4 W4A8 config. Must happen before ModelConfig
# resolves the checkpoint's quant_method, otherwise vLLM's INC config claims "auto-round" and the
# engine dies with "inc quantization is currently not supported in rocm". Import failures are
# reported and swallowed: a broken side module must not take down every other quantization method.
import os as _radiance_os  # noqa: E402

if _radiance_os.environ.get("RADIANCE_AUTOROUND", "0") == "1":
    try:
        import radiance_autoround  # noqa: F401,E402
    except Exception as _radiance_exc:  # pragma: no cover
        import sys as _radiance_sys  # noqa: E402
        _radiance_sys.stderr.write(
            f"[radiance.autoround] registration FAILED: {_radiance_exc!r}\\n")'''

SENTINEL = "radiance: register the gfx1201 AutoRound int4 W4A8 config"


def main():
    apply(QI, ANCHOR, NEW, SENTINEL, "autoround: register the quantization config")


if __name__ == "__main__":
    main()
