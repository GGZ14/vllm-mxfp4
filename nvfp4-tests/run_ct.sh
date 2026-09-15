#!/bin/bash
# Run a python snippet inside the radiance image with /patches (deadcode-vllm), /models and this dir mounted.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
exec podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=${GPU:-0} \
  -e HF_HUB_OFFLINE=1 -v "$HERE":/work:z -v "$(cd "$HERE/.." && pwd)":/patches:ro -v "${MODELS:-$HOME/models}":/models:ro \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH; cd /work; $*"
