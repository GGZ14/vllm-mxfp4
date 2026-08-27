#!/bin/bash
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
for V in '' '-DAR_BF16_TRUNC'; do
  hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 \$V tilebench.hip -o /tmp/bf > /dev/null 2>&1
  echo \"=== \${V:-shipped RNE cast} ===\"
  /tmp/bf 6 2>/dev/null | sed -n '2,8p' | awk '{print \$1, \$2, \$3, \$5}'
done"
