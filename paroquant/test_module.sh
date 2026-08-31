#!/bin/bash
# Run the radiance_paroquant.py integration test inside the radiance image.
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e RADIANCE_PQ_CHECKALL="17408:5120,34816:5120,14336:5120,16384:5120,5120:17408" -e RADIANCE_PQ_CHECK_MAX_M=512 \
  -v /home/brian/mxfp4_work/paro:/paro:z -v /home/brian/models:/models:ro \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
SP=/opt/vllm/lib/python3.12/site-packages
cd /paro
hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 \
  \$(python3 -m pybind11 --includes) radiance_paroquant.hip -o \"\$SP\"/radiance_paroquant_kernel.so \
  > /paro/modbuild.log 2>&1 || { echo BUILD FAILED; tail -30 /paro/modbuild.log; exit 1; }
cd /
python3 /paro/test_module.py"
