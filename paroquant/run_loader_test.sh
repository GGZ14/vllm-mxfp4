#!/bin/bash
# GPU unit test for the MXFP4 ParoQuant serving loader. Compiles both kernel modules into the
# image's site-packages the way run_paroquant.sh does, then runs test_mxfp4_loader.py.
# Needs a free GPU (HIP_VISIBLE_DEVICES) and the hybrid checkpoint at $MODELS/Qwen3.8-27B-PARO-MXFP4.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-$HOME/models}
GPU=${GPU:-0}
exec podman run --rm --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU" -e RADIANCE_PAROQUANT=1 \
  -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-1}" -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_PQ_WPERM="${RADIANCE_PQ_WPERM:-1}" -e RADIANCE_PQ_ROT_V2=1 \
  -v "$REPO":/patches:z -v "$MODELS":/models \
  --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    cp radiance_mxfp4.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    cd /patches/paroquant
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_paroquant.hip -o "$SP"/radiance_paroquant_kernel.so
    cp radiance_paroquant.py radiance_paroquant_mxfp4.py "$SP"/
    cd /
    python3 /patches/paroquant/test_mxfp4_loader.py'
