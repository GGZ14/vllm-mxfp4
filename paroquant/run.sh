#!/bin/bash
# Build and run the ParoQuant W4A8 kernel harness inside the radiance image.
# Compiler output is redirected to a file: SIGPIPE from a `head` on the pipeline kills the compile.
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/mxfp4_work/paro:/work:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /work
hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 \
  -Rpass-analysis=kernel-resource-usage par_harness.hip -o /tmp/par > /work/build.log 2>&1 || {
    echo BUILD FAILED; tail -40 /work/build.log; exit 1; }
grep -A9 'Function Name.*gemm_decode\|Function Name.*rotate_quant' /work/build.log \
  | grep -E 'VGPRs|SGPRs|Occupancy|Spill|LDS' | sed 's/^/  /' | head -40
echo
PQ_ABLATE_FOLD=${PQ_ABLATE_FOLD:-} PQ_BENCH_PTOK=${PQ_BENCH_PTOK:-} /tmp/par $*"
