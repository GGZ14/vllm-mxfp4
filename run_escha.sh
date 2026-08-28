#!/bin/bash
# Serve EschaLabs/Qwen3.8-27B-Escha-W2 through the radiance escha (EXL3 trellis) W2 kernels.
#
# Mirrors run_autoround.sh: build the kernel into the image's site-packages on every run (each
# `podman run --rm` starts from the pristine image), drop the quant method next to it, then exec
# the radiance entrypoint.
#
# NOTE: the production MXFP4 server holds --gpu-memory-utilization 0.98 on both GPUs. Stop it
# before running this.
#
# SPECULATION IS OFF by default: this checkpoint ships no MTP head (400 coded projections, all in
# the main decoder; no mtp.* tensors), so there is nothing to draft with unless a separate drafter
# is pointed at it. The DFlash2 FP8 drafter matches this target on hidden size and vocab and
# carries its own fp8 quant config, so it does not touch the escha path -- set SPEC_METHOD=dflash
# SPEC=7 to try it.
set -euo pipefail

MODEL=${MODEL:-/models/Qwen3.8-27B-Escha-W2}
PORT=${PORT:-8080}
TP=${TP:-2}
MAXLEN=${MAXLEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}
MAXSEQS=${MAXSEQS:-8}
CHUNK=${CHUNK:-8192}
# Derived from CHUNK, not hardcoded: the cap has to clear one prefill all-reduce, whose size is
# CHUNK x hidden x 2 B. Same expression the other launchers use.
AR_MAX_KB=${AR_MAX_KB:-$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))}
EAGER=${EAGER:-0}
SPEC_METHOD=${SPEC_METHOD:-none}
SPEC=${SPEC:-0}
DRAFTER=${DRAFTER:-/models/Qwen3.8-27B-DFlash2-FP8}
DRAFT_ATTN=${DRAFT_ATTN:-TRITON_ATTN}
UNPAD=${UNPAD:-true}
NAME=${NAME:-vllmescha}

HOSTMODEL=${MODEL/\/models/$HOME\/models}
if [ ! -d "$HOSTMODEL" ]; then
  echo "model not found: $HOSTMODEL" >&2
  exit 1
fi

# podman will not create a bind-mount source; it errors with statfs ENOENT.
mkdir -p "$HOME/.radiance-cache-escha"

EXTRA=""
[ "$EAGER" = "1" ] && EXTRA="$EXTRA --enforce-eager"
# Hand the JSON over as an environment variable and rebuild the argument INSIDE the container:
# interpolating it into the `bash -lc` string does not survive, the quotes are eaten and vLLM sees
# `method:dflash`, which argparse rejects.
SPEC_CFG=""
if [ "$SPEC" -gt 0 ] && [ "$SPEC_METHOD" = dflash ]; then
  HOSTDRAFTER=${DRAFTER/\/models/$HOME\/models}
  if [ ! -d "$HOSTDRAFTER" ] && [ ! -L "$HOSTDRAFTER" ]; then
    echo "drafter not found: $HOSTDRAFTER" >&2; exit 1
  fi
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
fi

exec podman run --replace --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_AR=1 -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_R4D_REPORT=1 \
  -e RADIANCE_AR_MAX_KB="$AR_MAX_KB" \
  -e RADIANCE_PRESHUFFLE="${RADIANCE_PRESHUFFLE:-1}" \
  -e RADIANCE_FUSE_RMS_QUANT="${RADIANCE_FUSE_RMS_QUANT:-1}" \
  -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e ESCHA_SPEC_CFG="$SPEC_CFG" \
  -e RADIANCE_ESCHA=1 \
  -e RADIANCE_ESCHA_TRACE="${RADIANCE_ESCHA_TRACE:-}" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "$HOME/models:/models:ro" \
  -v "$HOME/.radiance-cache-escha:/cache" \
  -v "$HOME/deadcode-vllm:/esc:z" \
  --entrypoint bash docker.io/stilldeadcode/vllm-radiance:0.9.3 -lc "
set -e
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:\$LD_LIBRARY_PATH
SP=/opt/vllm/lib/python3.12/site-packages
cd /esc
hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 \
  \$(python3 -m pybind11 --includes) radiance_escha.hip -o \"\$SP\"/radiance_escha_kernel.so
cp radiance_escha.py \"\$SP\"/
python3 patch_escha.py
# Leave /esc before exec: it is a bind mount and precedes site-packages on sys.path, so a stale
# .so built there would shadow the one just compiled into site-packages.
cd /
SPECARGS=()
if [ -n \"\${ESCHA_SPEC_CFG:-}\" ]; then SPECARGS=(--speculative-config \"\$ESCHA_SPEC_CFG\"); fi
exec /opt/radiance_entrypoint.sh $MODEL \
  --served-model-name Qwen3.8-Escha \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAXLEN \
  --max-num-seqs $MAXSEQS \
  --max-num-batched-tokens $CHUNK \
  --trust-remote-code \
  --kv-cache-dtype fp8 --mamba-cache-mode align --enable-prefix-caching \
  --attention-backend R4D --no-async-scheduling \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  \"\${SPECARGS[@]}\" $EXTRA
"
