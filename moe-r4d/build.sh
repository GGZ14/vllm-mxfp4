#!/bin/bash
# Build the r4d_moe pybind module (libr4d v0.5.0's paged prefill attention, instantiated at GQA 8 for
# Qwen3.5/3.6-35B-A3B: 16 q / 2 kv heads, head 256) inside the vllm-radiance image, with the hipcc
# the venv loads it against. serve-moe-mxfp4.sh runs this at container start (MOE_PREFILL_ATTN=r4d).
#   R4D_SRC  libr4d v0.5.0 checkout (required; r4d_moe.hip includes its headers and prefill kernel)
#   OUT      output file (default r4d_moe<EXT> here)
# Flags match libr4d v0.5.0's build.sh (-O3 -ffp-contract=off, gfx1201).
set -euo pipefail
cd "$(dirname "$0")"
R4D_SRC=${R4D_SRC:?R4D_SRC must point at a libr4d v0.5.0 checkout}
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT=${OUT:-r4d_moe$EXT}
${HIPCC:-hipcc} -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off \
  -I "$R4D_SRC" $(python3 -m pybind11 --includes) r4d_moe.hip -o "$OUT"
echo "  OK    r4d_moe: built $OUT against $R4D_SRC"
