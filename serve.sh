#!/usr/bin/env bash
# serve.sh — start the radiance server.
#
# Picks the right --model path + serve-model-name + drafter for the chosen
# quant format, and `docker run`s the published image. The image is the
# SAME regardless of QUANT — dispatch is by the checkpoint's quant_method.
#
# Usage:
#   ./serve.sh                          # MXFP4 (quark_mxfp4)
#   QUANT=int4 ./serve.sh               # ParoQuant int4 W4A8
#   QUANT=int5 ./serve.sh               # ParoQuant int5 W5A8
#
# Environment:
#   IMAGE=radiance:latest                # image to run (the prod-bake image)
#   MODELS=~/models                      # where the checkpoints are
#   PORT=8080                            # host port to expose
#   GPU_UTIL=0.98                        # vLLM gpu-memory-utilization
#   KV_MEM=17317374464                   # --kv-cache-memory (par pinning; mxfp4
#                                        #   lets vLLM auto-size at GPU_UTIL)
#   MAXLEN=262144                        # --max-model-len
#   MAXSEQS=8                            # --max-num-seqs
#   SPEC=7                               # dflash num_speculative_tokens
#   ASYNC=0                              # 1 -> --async-scheduling, 0 -> --no-async-scheduling
#   TEMP=0.7                             # override-generation-config temperature
#   TOP_P=0.95                           # override-generation-config top_p
#   TOP_K=20                             # override-generation-config top_k
#   NO_DRAFTER=1                         # skip the drafter (then SPEC_METHOD=mtp)

set -euo pipefail

QUANT=${QUANT:-mxfp4}
MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-radiance:latest}
PORT=${PORT:-8080}
GPU_UTIL=${GPU_UTIL:-0.98}
KV_MEM=${KV_MEM:-}
MAXLEN=${MAXLEN:-262144}
MAXSEQS=${MAXSEQS:-8}
SPEC=${SPEC:-7}
ASYNC=${ASYNC:-0}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}

case "$QUANT" in
  mxfp4) SNAP_REL=Qwen3.8-27B-MXFP4-mtpfp8
         SERVED=Qwen3.8
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8 ;;
  int4)  SNAP_REL=Qwen3.8-27B-PARO
         SERVED=Qwen3.8-PARO
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8 ;;
  int5)  SNAP_REL=Qwen3.8-27B-PARO-int5
         SERVED=Qwen3.8-PARO
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8 ;;
  *) echo "unknown QUANT=$QUANT (mxfp4 / int4 / int5)" >&2; exit 2 ;;
esac

SNAP="$MODELS/$SNAP_REL"
[ -d "$SNAP" ] || { echo "ERROR: checkpoint not at $SNAP (run setup.sh)" >&2; exit 1; }

# ---- the gpu group ids this stack expects on gfx1201 ----------------------------
# 993 = render (libr4d-rx6 needs write access to render nodes), 44 = video.
GROUP_ARGS=(--group-add 993 --group-add 44)

# ---- the same security_opts the in-tree launcher uses --------------------------
SECURITY_ARGS=(--security-opt seccomp=unconfined
               --security-opt apparmor=unconfined)

# ---- async scheduling ---------------------------------------------------------
if [ "$ASYNC" = 1 ]; then
  ASYNC_FLAG=--async-scheduling; UNPAD=false
else
  ASYNC_FLAG=--no-async-scheduling; UNPAD=true
fi

# ---- drafter spec config -------------------------------------------------------
SPEC_ARGS=()
if [ -z "${NO_DRAFTER:-}" ]; then
  DRAFTER="$MODELS/$DRAFTER_REL"
  [ -d "$DRAFTER" ] || { echo "ERROR: drafter not at $DRAFTER (setup.sh)" >&2; exit 1; }
  SPEC_ARGS=(--speculative-config \
    "{\"method\":\"dflash\",\"model\":\"/models/$DRAFTER_REL\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":$UNPAD,\"draft_sample_method\":\"greedy\"}")
fi

# ---- KV cache memory (opt-in: pin or auto at GPU_UTIL) -------------------------
KV_ARGS=()
[ -n "$KV_MEM" ] && KV_ARGS=(--kv-cache-memory "$KV_MEM")

# ---- compose the vllm-serve args -----------------------------------------------
VLLM_ARGS=(
  "$SNAP"
  --served-model-name "$SERVED"
  --host 0.0.0.0 --port "$PORT"
  --kv-cache-dtype fp8
  --tensor-parallel-size "${TP:-2}"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$MAXSEQS"
  --max-num-batched-tokens 8192
  --attention-backend R4D
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  "$ASYNC_FLAG"
  --override-generation-config "{\"temperature\":$TEMP,\"top_p\":$TOP_P,\"top_k\":$TOP_K}"
  "${SPEC_ARGS[@]}"
  "${KV_ARGS[@]}"
)

RUNTIME=${RUNTIME:-}
[ -z "$RUNTIME" ] && command -v docker >/dev/null 2>&1 && RUNTIME=docker
[ -z "$RUNTIME" ] && command -v podman >/dev/null 2>&1 && RUNTIME=podman
[ -n "$RUNTIME" ] || { echo "ERROR: no container runtime" >&2; exit 1; }

# ---- the run ------------------------------------------------------------------
# `--ipc=host` is needed for vLLM's tensor-parallel shared memory; `--network=host`
# keeps `localhost:8080` working without publishing a port separately, which is
# what every local setup wants. Drop both if you need a stricter sandbox.
exec "$RUNTIME" run --rm \
  --network=host --ipc=host \
  --device /dev/kfd --device /dev/dri \
  "${GROUP_ARGS[@]}" "${SECURITY_ARGS[@]}" \
  -v "$MODELS":/models \
  "$IMAGE" "${VLLM_ARGS[@]}"