#!/bin/bash
# Prefill BK sweep at the shipped tile. LDS = (AR_BMF + WN*TN*16) * (AR_BK + 8); at TN=4 that is
# 15 KB / 27 KB / 51 KB for BK 32 / 64 / 128, so BK=128 drops to one resident block per CU.
# IMAJOR folds the group scale per slab, which stays correct for any BK dividing the 128 group.
set -e
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v /home/brian/deadcode-vllm:/repo:z \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
cd /repo/autoround-tests
for BK in 32 64 128; do
  hipcc -O3 -w -std=c++17 --offload-arch=gfx1201 -DAR_BK=\$BK -Rpass-analysis=kernel-resource-usage \
    tilebench.hip -o /tmp/b\$BK > /repo/autoround-tests/bk_\$BK.log 2>&1 || {
      echo \"BK=\$BK BUILD FAIL\"; grep -m2 error: /repo/autoround-tests/bk_\$BK.log; continue; }
  echo \"=== AR_BK=\$BK ===\"
  /tmp/b\$BK 6 2>/dev/null | sed -n '2,8p'
done"
