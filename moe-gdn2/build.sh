#!/bin/bash
# Build the radiance_gdn2 pybind module inside the vllm-radiance image, with the hipcc the venv loads it
# against: an exact GDN chunk scan (libr4d #4 fix) with the same ABI as r4d.gdn_chunk_scan_k128_v128_c64_bf16,
# plus libr4d v0.5.0's own scan from the same source for A/B. serve-moe-mxfp4.sh runs this at container
# start (RADIANCE_GDN_SCAN_FIX=1).
#   R4D_SRC  libr4d v0.5.0 checkout (required, read-only is fine)
#   OUT      output file (default radiance_gdn2<EXT> in the current directory)
#   GDN2_EXTRA  extra hipcc flags (e.g. -Rpass-analysis=kernel-resource-usage)
# No libr4d source is stored in this repo. radiance_gdn2_vs_v050.patch is a diff against libr4d v0.5.0's
# r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip; this script applies it to a temporary copy of that file taken
# from R4D_SRC (apply_patch.py: the image has no `patch` or `git`), checks the result byte for byte, and
# compiles it there. The translation unit includes r4d_gdn_wmma.h from R4D_SRC.
# Flags match libr4d v0.5.0's build.sh for this translation unit: -O3 -ffp-contract=off, gfx1201, and
# -mcumode (the scan asks for ~60 KB of LDS per workgroup and was tuned with one workgroup per CU).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R4D_SRC=${R4D_SRC:?R4D_SRC must point at a libr4d v0.5.0 checkout}
fail() { echo "  FAIL  radiance_gdn2: $*" >&2; exit 1; }
md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }
SCAN=r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip
grep -q '#define R4D_VERSION "0.5.0"' "$R4D_SRC/r4d.h" 2>/dev/null || fail "$R4D_SRC is not libr4d 0.5.0"
# The patch is a diff against this exact upstream file and the result includes this header: refuse to
# build against anything else rather than silently mixing versions.
[ "$(md5 "$R4D_SRC/$SCAN")" = 3c1f87fbc9598b06c0b452f1f6edf09f ] \
  || fail "upstream chunk scan in $R4D_SRC differs from the v0.5.0 file this fix was derived from"
[ "$(md5 "$R4D_SRC/r4d_gdn_wmma.h")" = dcb10fd659237905e75e19acdbc6c050 ] \
  || fail "r4d_gdn_wmma.h in $R4D_SRC differs from v0.5.0"
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT=${OUT:-radiance_gdn2$EXT}
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$R4D_SRC/$SCAN" "$WORK/radiance_gdn2.hip"
python3 "$HERE/apply_patch.py" "$WORK/radiance_gdn2.hip" "$HERE/radiance_gdn2_vs_v050.patch" \
  || fail "radiance_gdn2_vs_v050.patch does not apply to the upstream chunk scan"
[ "$(md5 "$WORK/radiance_gdn2.hip")" = db41219a6486d0262ea8eac007fd7713 ] \
  || fail "patched radiance_gdn2.hip is not the validated file"
cd "$WORK"
# shellcheck disable=SC2086
${HIPCC:-hipcc} -O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off \
  -mcumode -I "$R4D_SRC" $(python3 -m pybind11 --includes) ${GDN2_EXTRA:-} radiance_gdn2.hip -o "$OUT"
echo "  OK    radiance_gdn2: built $OUT against $R4D_SRC"
