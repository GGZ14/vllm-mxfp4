#!/bin/bash
# Build the radiance_fp8w HIP module (skinny W8A16 GEMM for the fp8 dense layers, M <= 16) inside the
# vllm-radiance image, with the hipcc the venv loads it against.
# serve-moe-mxfp4.sh runs this at container start (RADIANCE_MOE_DENSE_FP8=1). The Python side,
# radiance_moe_densefp8.py, is installed by patch_moe_densefp8.py, not here; the fused expert gate
# (RADIANCE_MOE_GATE_FIX) needs no HIP module.
#   OUT  output directory (default: the current directory), e.g. the image's site-packages
# Flags match moe-w4a8/build.sh: -O3, gfx1201. radiance_fp8w.hip is written in this repo, but its skeleton
# (accumulate, reduce, launch dispatch) follows libr4d v0.5.0's r4d_gemm_bf16_nt_m16.hip and a few boilerplate
# lines are identical to it; the fp8 weight load, decode and scale epilogue are new. Nothing of libr4d is
# needed at build time.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-.}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# shellcheck disable=SC2086
${HIPCC:-hipcc} -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 \
  $(python3 -m pybind11 --includes) ${FP8W_EXTRA:-} "$HERE/radiance_fp8w.hip" -o "$OUT/radiance_fp8w$EXT"
echo "  OK    radiance_fp8w: built $OUT/radiance_fp8w$EXT"
