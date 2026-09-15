#!/usr/bin/env bash
# serve.sh — start the radiance server from the prod-bake image.
#
# The image is the SAME for every quant format; the checkpoint's
# quantization_config selects the kernel (quark_mxfp4 / amd_mxfp4 -> the mxfp4
# fp8-WMMA GEMM; paroquant_mxfp4 -> that plus ParoQuant rotations; paroquant ->
# the ParoQuant W4A8/W5A8 kernel). This script mirrors what the launchers do
# at run time:
#
#   - the per-quant env profile. The image ENV is the MXFP4 measured profile
#     (the serve-mxfp4.sh "-e" defaults, baked in). The ParoQuant launches run
#     with a different profile (GDN_MERGE_INPROJ=0, RADIANCE_PAROQUANT=1, the
#     rotation/producers flags, the per-bit-width I8/PG/ZPE read off the
#     checkpoint) — exported here per QUANT so a bare `./serve.sh` of a
#     ParoQuant checkpoint reproduces the launcher.
#   - the --compilation-config pass_config entry the MXFP4 launcher appends
#     with NQF=1 (the fuse_norm_quant / fuse_act_quant vLLM passes).
#   - the measured default GPU_UTIL per quant (0.98 mxfp4 / 0.92 ParoQuant).
#   - the shipped chat template (baked into the image at /opt).
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
#   GPU_UTIL=(unset: 0.98 mxfp4, 0.92 ParoQuant)
#   KV_MEM=                              # optional --kv-cache-memory pin
#   MAXLEN=262144                        # --max-model-len
#   MAXSEQS=8                            # --max-num-seqs
#   SPEC=7                               # dflash num_speculative_tokens
#   SPEC_METHOD=dflash                   # dflash | mtp (then NODRAFTER)
#   ASYNC=0                              # 1 -> --async-scheduling
#   NO_DRAFTER=1                         # skip the drafter (SPEC_METHOD=mtp)
#   TP=2                                 # --tensor-parallel-size
#   RADIANCE*=<any>                      # any RADIANCE_* / VLLM_* here overrides
#                                       # the per-quant profile (env wins)

set -euo pipefail

QUANT=${QUANT:-mxfp4}
MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-radiance:latest}
PORT=${PORT:-8080}
KV_MEM=${KV_MEM:-}
CHUNK=${CHUNK:-8192}
MAXLEN=${MAXLEN:-262144}
MAXSEQS=${MAXSEQS:-8}
SPEC=${SPEC:-7}
SPEC_METHOD=${SPEC_METHOD:-dflash}
ASYNC=${ASYNC:-0}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}

case "$QUANT" in
  mxfp4) SNAP_REL=Qwen3.8-27B-MXFP4-mtpfp8
         SERVED=Qwen3.8
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8
         GPU_UTIL=${GPU_UTIL:-0.98} ;;
  int4)  SNAP_REL=Qwen3.8-27B-PARO
         SERVED=Qwen3.8-PARO
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8
         GPU_UTIL=${GPU_UTIL:-0.92} ;;
  int5)  SNAP_REL=Qwen3.8-27B-PARO-int5
         SERVED=Qwen3.8-PARO
         DRAFTER_REL=Qwen3.8-27B-DFlash2-FP8
         GPU_UTIL=${GPU_UTIL:-0.92} ;;
  *) echo "unknown QUANT=$QUANT (mxfp4 / int4 / int5)" >&2; exit 2 ;;
esac

SNAP="$MODELS/$SNAP_REL"
[ -d "$SNAP" ] || { echo "ERROR: checkpoint not at $SNAP (run setup.sh)" >&2; exit 1; }

# --------------------------------------------------------- the env profile -----
# RADIANCE_ENV=(--env ...): the per-quant profile, mirrored from the launchers.
# An explicit shell RADIANCE_* always wins (we only export what is unset).
ENV_ARGS=()
e() { # e NAME VALUE  -> add -e only if NAME is not already set in the shell
  if [ -z "${!1:-}" ]; then ENV_ARGS+=(-e "$1=$2"); fi
}

case "$QUANT" in
  int4|int5)
    # The ParoQuant profile (run_paroquant.sh, MODE=prod): the rotation_supply
    # producers on, the int4-default I8/PG/ZPE (int5 reads them off the
    # checkpoint, exactly as the launcher does), GDN merge off (in_proj_a/b are
    # fp16 there; the mxfp4-baked profile has it on).
    e RADIANCE_PAROQUANT 1
    e RADIANCE_GDN_MERGE_INPROJ 0
    e RADIANCE_PQ_WPERM 1
    e RADIANCE_PQ_DECODE_NT 1
    e RADIANCE_PQ_AT_LBK 128
    e RADIANCE_PQ_AT_HOIST 1
    e RADIANCE_PQ_PTOK 1
    e RADIANCE_PQ_PG_PRODUCER 3
    e RADIANCE_PQ_FUSED_TOKQ 1
    e RADIANCE_PQ_CHECK_MAX_M 128
    e RADIANCE_PQ_DECODE_MAX_M 64
    e RADIANCE_PQ_ROT_STREAM 1
    e RADIANCE_PQ_ROT_STREAM2 1
    # Per-bit-width defaults (the launcher reads the width off the checkpoint).
    if [ "$QUANT" = int5 ]; then
      PQ_BITS=$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("quantization_config") or {}).get("bits",4))' \
               "$SNAP/config.json" 2>/dev/null || echo 4)
      [ "$PQ_BITS" = 5 ] || { echo "NOTE: $SNAP config does not declare bits=5 (got ${PQ_BITS}); serving int4 defaults" >&2; }
      e RADIANCE_PQ_I8 1
      e RADIANCE_PQ_PG 1
      e RADIANCE_PQ_ZPE 1
    else
      e RADIANCE_PQ_I8 0
      e RADIANCE_PQ_PG 0
      e RADIANCE_PQ_ZPE 0
    fi
    ;;
  mxfp4)
    # The measured profile is the image ENV; nothing extra to export. (Override
    # with an explicit RADIANCE_* in the shell if you want a bisect arm.)
    ;;
esac

# --------------------------------------------------------- the gpu groups ------
GROUP_ARGS=(--group-add 993 --group-add 44)

# --------------------------------------------------------- async / unpad -------
if [ "$ASYNC" = 1 ]; then
  ASYNC_FLAG=--async-scheduling; UNPAD=false
else
  ASYNC_FLAG=--no-async-scheduling; UNPAD=true
fi

# --------------------------------------------------------- drafter spec --------
SPEC_ARGS=()
if [ "$SPEC_METHOD" = dflash ] && [ -z "${NO_DRAFTER:-}" ]; then
  DRAFTER="$MODELS/$DRAFTER_REL"
  [ -d "$DRAFTER" ] || { echo "ERROR: drafter not at $DRAFTER (setup.sh)" >&2; exit 1; }
  SPEC_ARGS=(--speculative-config \
    "{\"method\":\"dflash\",\"model\":\"/models/$DRAFTER_REL\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":$UNPAD,\"draft_sample_method\":\"greedy\"}")
elif [ "$SPEC_METHOD" = mtp ]; then
  # mtp uses the head inside the target (no extra download); measured default 4.
  mspec=${SPEC:-4}
  SPEC_ARGS=(--speculative-config \
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${mspec},\"attention_backend\":\"R4D\",\"disable_padded_drafter_batch\":$UNPAD}")
fi

# --------------------------------------------------------- KV pin (opt-in) -----
KV_ARGS=()
[ -n "$KV_MEM" ] && KV_ARGS=(--kv-cache-memory "$KV_MEM")

# --------------------------------------------------------- compilation config --
# The MXFP4 launcher appends the pass_config entry when NQF=1 (the traced-quant
# passes). Keep the measured graph key -nqft; the ParoQuant profile leaves the
# passes off (the launcher there does not set them).
CC_ARGS=()
if [ "$QUANT" = mxfp4 ] && [ "${RADIANCE_NORMQUANT_FUSION:-1}" = 1 ]; then
  CC_ARGS=(--compilation-config '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}')
fi

# --------------------------------------------------------- vllm serve args -----
VLLM_ARGS=(
  "$SNAP"
  --served-model-name "$SERVED"
  --host 0.0.0.0 --port "$PORT"
  --kv-cache-dtype fp8
  --tensor-parallel-size "${TP:-2}"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$MAXSEQS"
  --max-num-batched-tokens "$CHUNK"
  --attention-backend R4D
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --chat-template /opt/qwen-fixed-v22.3.jinja
  "$ASYNC_FLAG"
  --override-generation-config "{\"temperature\":$TEMP,\"top_p\":$TOP_P,\"top_k\":$TOP_K}"
  "${SPEC_ARGS[@]}"
  "${KV_ARGS[@]}"
  "${CC_ARGS[@]}"
)

# --------------------------------------------------------- runtime -------------
RUNTIME=${RUNTIME:-}
[ -z "$RUNTIME" ] && command -v docker >/dev/null 2>&1 && RUNTIME=docker
[ -z "$RUNTIME" ] && command -v podman >/dev/null 2>&1 && RUNTIME=podman
[ -n "$RUNTIME" ] || { echo "ERROR: no container runtime" >&2; exit 1; }

exec "$RUNTIME" run --rm \
  --network=host --ipc=host \
  --device /dev/kfd --device /dev/dri \
  "${GROUP_ARGS[@]}" \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  "${ENV_ARGS[@]}" \
  -v "$MODELS":/models \
  "$IMAGE" "${VLLM_ARGS[@]}"