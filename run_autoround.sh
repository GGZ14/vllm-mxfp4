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
# The default RADIANCE_AR_MAX_KB (32768) is below a prefill all-reduce, so every one of them goes
# to NCCL instead of the radiance all-reduce. Raising it is measured at +3-13% prefill and free.
# 86016 is what the production MXFP4 serve runs.
# Derived from CHUNK, not hardcoded: the cap has to clear one prefill all-reduce, whose size is
# CHUNK x hidden x 2 B. run_mxfp4_074.sh uses the same expression.
AR_MAX_KB=${AR_MAX_KB:-$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))}
EAGER=${EAGER:-0}
# The checkpoint ships a single-layer MTP head (mtp_num_hidden_layers: 1) and AutoRound quantized
# it too -- mtp.layers.0 has qweight/scales for q/k/v/o and the MLP -- so the drafter runs on this
# same int4 kernel rather than needing a separate checkpoint. SPEC=0 disables speculation.
#
# The head is standard self-attention (q_proj/k_proj/o_proj with q_norm/k_norm), unlike the GDN
# main model, so it gets TRITON_ATTN rather than the R4D backend the target uses.
# SPEC_METHOD=mtp uses the head built into the AutoRound checkpoint (int4, served by this same
# kernel). SPEC_METHOD=dflash uses a SEPARATE drafter checkpoint -- the same FP8 DFlash2 drafter
# the MXFP4 production build runs, which matches this target on hidden size and vocab and carries
# its own fp8 quant config, so it does not touch the auto-round path at all.
# Depth defaults differ: dflash drafts a whole block and peaks at 7; mtp is settled at 4.
SPEC_METHOD=${SPEC_METHOD:-mtp}
if [ "$SPEC_METHOD" = dflash ]; then SPEC=${SPEC:-7}; else SPEC=${SPEC:-0}; fi
DRAFTER=${DRAFTER:-/models/Qwen3.8-27B-DFlash2-FP8}
DRAFT_ATTN=${DRAFT_ATTN:-TRITON_ATTN}
# disable_padded_drafter_batch: the unpad fix is what enables the single-stream drafter path.
UNPAD=${UNPAD:-true}
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
# Hand the JSON over as an environment variable and rebuild the argument INSIDE the container.
# Interpolating it into the `bash -lc` string does not survive: the quotes are eaten and vLLM sees
# `method:mtp`, which argparse rejects with "cannot be converted to <function loads>".
SPEC_CFG=""
if [ "$SPEC" -gt 0 ]; then
  if [ "$SPEC_METHOD" = dflash ]; then
    HOSTDRAFTER=${DRAFTER/\/models/$HOME\/models}
    if [ ! -d "$HOSTDRAFTER" ] && [ ! -L "$HOSTDRAFTER" ]; then
      echo "drafter not found: $HOSTDRAFTER" >&2; exit 1
    fi
    SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
  else
    SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
  fi
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
  -e AR_SPEC_CFG="$SPEC_CFG" \
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
SPECARGS=()
if [ -n \"\${AR_SPEC_CFG:-}\" ]; then SPECARGS=(--speculative-config \"\$AR_SPEC_CFG\"); fi
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
  --no-async-scheduling $EXTRA \"\${SPECARGS[@]}\""
