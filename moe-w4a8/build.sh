#!/bin/bash
# Build the radiance_moe_w4a8 HIP module (grouped MXFP4 x FP8 WMMA MoE GEMM + fp8 row quant) inside the
# vllm-radiance image, with the hipcc the venv loads it against, and install the Python wrapper next to it.
# serve-moe-mxfp4.sh runs this at container start (RADIANCE_MOE_W4A8=1).
#   OUT  output directory (default: the current directory), e.g. the image's site-packages
# Flags match the dense W4A8 kernel's image build (radiance_mxfp4_fp8.hip): -O3, gfx1201.
# radiance_moe_w4a8.hip is derived from this repo's radiance_mxfp4_fp8.hip. No libr4d code is involved.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-.}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cd "$HERE"
# shellcheck disable=SC2086
${HIPCC:-hipcc} -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 -Wno-unused-result \
  $(python3 -m pybind11 --includes) ${W4A8_EXTRA:-} radiance_moe_w4a8.hip -o "$OUT/radiance_moe_w4a8_hip$EXT"
[ "$OUT" = "$HERE" ] || cp radiance_moe_w4a8.py "$OUT/radiance_moe_w4a8.py"
echo "  OK    radiance_moe_w4a8: built $OUT/radiance_moe_w4a8_hip$EXT"
