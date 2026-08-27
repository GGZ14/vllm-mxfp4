#!/bin/bash
# Decode weight-staging width A/B: 8 bytes/thread (AR_DEC_WU=2, shipped) vs 16 (=4).
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
for WU in 2 4; do
  hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -DAR_DEC_WU=\$WU cmp.hip -o /tmp/c\$WU \
    > /repo/autoround-tests/wu_\$WU.log 2>&1 || { echo \"WU=\$WU BUILD FAIL\"; tail -5 /repo/autoround-tests/wu_\$WU.log; continue; }
  echo \"=== AR_DEC_WU=\$WU (\$((WU*4)) bytes/thread) ===\"
  /tmp/c\$WU 6 2>/dev/null | sed -n '2,8p'
done"
