#!/bin/bash
# Build AND run. Needs the GPU (--device /dev/kfd, /dev/dri); build.sh is the compile-only path.
set -e
exec podman run --rm --privileged --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/mxfp4_work/escha/kernel:/work:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc '
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LD_LIBRARY_PATH
cd /work
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 '"$1"' -o /tmp/t > /work/run_build.log 2>&1 || {
  echo BUILD FAILED; tail -25 /work/run_build.log; exit 1; }
/tmp/t'
