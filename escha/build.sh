#!/bin/bash
# COMPILE ONLY -- no GPU devices are mounted, so this cannot touch the GPUs.
# Running the harness needs --device /dev/kfd --device /dev/dri and is a separate, explicit step.
set -e
exec podman run --rm -v /home/brian/mxfp4_work/escha/kernel:/work:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc '
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LD_LIBRARY_PATH
cd /work
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage \
  '"$1"' -o /tmp/out > /work/build.log 2>&1 || { echo BUILD FAILED; tail -30 /work/build.log; exit 1; }
echo "BUILD OK: '"$1"'"
grep -E "VGPRs:|VGPRs Spill:|Occupancy" /work/build.log | sort -u | head -6'
