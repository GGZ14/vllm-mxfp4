#!/bin/bash
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 arsweep.hip -o /tmp/arsweep > arsweep_build.log 2>&1 || {
  echo BUILD FAILED; tail -30 arsweep_build.log; exit 1; }
/tmp/arsweep"
