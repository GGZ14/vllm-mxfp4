#!/bin/bash
# Serve Qwen3.8-27B-int4-AutoRound through the radiance W4A8 kernel.
#
# Mirrors run_mxfp4_074.sh's structure: build the kernel into the image's site-packages on every
# run (each `podman run --rm` starts from the pristine image), drop the quant method next to it,
# then exec the radiance entrypoint.
#
# NOTE: the production MXFP4 server holds --gpu-memory-utilization 0.98 on both GPUs. Stop it
# before running this.
set -euo pipefail

MODEL=${MODEL:-/models/Qwen3.8-27B-AutoRound-int4}
PORT=${PORT:-8080}
TP=${TP:-2}
MAXLEN=${MAXLEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}
MAXSEQS=${MAXSEQS:-8}
CHUNK=${CHUNK:-8192}
EAGER=${EAGER:-0}
CHECKALL=${CHECKALL:-}
NAME=${NAME:-vllmautoround}

HOSTMODEL=${MODEL/\/models/$HOME\/models}
if [ ! -d "$HOSTMODEL" ]; then
  echo "model not found: $HOSTMODEL" >&2
  exit 1
fi

# podman will not create a bind-mount source, it errors with statfs ENOENT.
mkdir -p "$HOME/.radiance-cache-autoround"

EXTRA=""
[ "$EAGER" = "1" ] && EXTRA="$EXTRA --enforce-eager"

exec podman run --replace --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e RADIANCE_USE_R4D=1 \
  -e RADIANCE_AUTOROUND=1 \
  -e RADIANCE_AR_DECODE_MAX_M=${AR_DECODE_MAX_M:-64} \
  -e RADIANCE_AR_CHECKALL="$CHECKALL" \
  -e RADIANCE_AR_CHECK_MAX_M=${AR_CHECK_MAX_M:-128} \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$HOME/models:/models:ro" \
  -v "$HOME/.radiance-cache-autoround:/cache" \
  -v "$HOME/deadcode-vllm:/ar:z" \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
SP=/opt/vllm/lib/python3.12/site-packages
cd /ar
hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 \
  \$(python3 -m pybind11 --includes) radiance_autoround.hip -o \"\$SP\"/radiance_autoround_kernel.so
cp radiance_autoround.py \"\$SP\"/
python3 patch_autoround.py
# Leave /ar before exec: it is a bind mount and precedes site-packages on sys.path, so a stale
# .so built there would shadow the one just compiled into site-packages.
cd /
exec /opt/radiance_entrypoint.sh $MODEL \
  --served-model-name Qwen3.8-AutoRound \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAXLEN \
  --max-num-seqs $MAXSEQS \
  --max-num-batched-tokens $CHUNK \
  --attention-backend R4D \
  --kv-cache-dtype fp8 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --no-async-scheduling $EXTRA"
