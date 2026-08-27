#!/bin/bash
# Ablation + head-to-head against the SHIPPED kernel header (../radiance_autoround_kernels.h),
# not a local copy -- tilebench/preopt/decopt each hold a copy that can drift.
set -e
WHICH=${WHICH:-abl}
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage \
  $WHICH.hip -o /tmp/$WHICH > /repo/autoround-tests/${WHICH}_build.log 2>&1 || {
    echo BUILD FAILED; tail -30 /repo/autoround-tests/${WHICH}_build.log; exit 1; }
/tmp/$WHICH $*"
