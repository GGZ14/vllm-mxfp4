#!/bin/bash
# Correctness gate (and --bench) for the AutoRound int4 W4A8 kernels, built from the REPO copies
# in the parent directory -- the same radiance_autoround_kernels.h the serving module compiles, so
# a change cannot be benchmarked without also being gated.
#
# AR_IMAJOR selects the prefill variant under test; both are gated.
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage \
  ar_harness.hip -o /tmp/ar > /repo/autoround-tests/build.log 2>&1 || {
    echo BUILD FAILED; tail -40 /repo/autoround-tests/build.log; exit 1; }
grep -E 'VGPRs:|VGPRs Spill:|Occupancy' /repo/autoround-tests/build.log | sort -u | head -8
echo
AR_IMAJOR=1 /tmp/ar $*"
