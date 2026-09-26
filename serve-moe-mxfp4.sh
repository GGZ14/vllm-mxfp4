#!/bin/bash
# Serve a Qwen3.5/3.6-35B-A3B-class MXFP4 MoE checkpoint on ONE gfx1201 card: plain `vllm serve` on the
# Radiance image plus the gfx1201 MoE fixes. serve-mxfp4.sh is built around Qwen3.8-27B (R4D attention
# needs GQA 6, the DFlash2 drafter, measured KV pins); an A3B MoE (GQA 8, 256 experts, its own MTP
# head) runs here instead. Details and measurements: MOE-GFX1201.md.
#
#   SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 ./serve-moe-mxfp4.sh [extra vllm serve args]
#
# Knobs (all ${VAR:-default}):
#   SNAP              checkpoint directory (required). Measured: amd/Qwen3.5-35B-A3B-MXFP4 (Quark W4A4,
#                     experts only). A Hugging Face cache snapshot path works (its hub dir is mounted).
#   MOE_FIXES=1       1: experts weight-only on the AITER a16w4 lane (patch_quark_moe_w4a16.py,
#                     patch_gfx12_aiter_a16w4.py) + split-KV verify attention (patch_attn_3d_multiq.py).
#                     0: stock vLLM, i.e. the EMULATION MoE backend on gfx1201 (~5x slower decode).
#   RAM_TIER_BYTES=0  host-RAM KV cache tier in bytes (OffloadingConnector); 0 = off. Lives in /dev/shm.
#   SPEC=8            MTP draft tokens; 0 disables speculation
#   MAXLEN=65536  MAXSEQS=8  CHUNK=4096  GPU_UTIL=0.95  PORT=8080  NAME=radiance-moe  GPUS=0
#   IMAGE=stilldeadcode/vllm-radiance:0.9.3   RUNTIME=podman|docker (auto)
#   CACHE=~/.radiance-cache-moe-<model>-f<fixes>[-rt]   compile cache; never share one across knobs
#   SERVED_NAMES=<basename of the model>   DRY_RUN=1 print the command   DETACH=1 run in background
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
die() { echo "[serve-moe] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

SNAP=${SNAP:-}
[ -n "$SNAP" ] || die "set SNAP to the checkpoint directory" "e.g. SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 $0"
[ -f "$SNAP/config.json" ] || die "no config.json in $SNAP"
SNAP=$(cd "$SNAP" && pwd)
KIND=$(python3 - "$SNAP/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c)
moe = any((t.get(k) or 0) > 0 for k in ("num_experts", "n_routed_experts", "num_local_experts"))
print("moe" if moe or str(t.get("model_type", "")).endswith("_moe") else "dense")
PY
)
[ "$KIND" = moe ] || echo "[serve-moe] WARNING: $SNAP does not look like an MoE checkpoint; serve-mxfp4.sh is the dense launcher" >&2

MOE_FIXES=${MOE_FIXES:-1}; RAM_TIER_BYTES=${RAM_TIER_BYTES:-0}; SPEC=${SPEC:-8}
MAXLEN=${MAXLEN:-65536}; MAXSEQS=${MAXSEQS:-8}; CHUNK=${CHUNK:-4096}; GPU_UTIL=${GPU_UTIL:-0.95}
PORT=${PORT:-8080}; NAME=${NAME:-radiance-moe}; GPUS=${GPUS:-0}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}

# A Hugging Face cache snapshot holds symlinks into ../../blobs: mount the whole hub dir, at the same
# path, so they resolve. Anything else mounts just the checkpoint.
case "$SNAP" in */models--*/snapshots/*) MOUNT=${SNAP%%/models--*} ;; *) MOUNT=$SNAP ;; esac
MODEL_ID=$(basename "$SNAP")
case "$SNAP" in */snapshots/*) MODEL_ID=$(basename "$(dirname "$(dirname "$SNAP")")" | sed 's/^models--//; s#--#/#') ;; esac
SERVED_NAMES=${SERVED_NAMES:-$(basename "$MODEL_ID")}

CACHE_SUF="-f$MOE_FIXES"; [ "$RAM_TIER_BYTES" != 0 ] && CACHE_SUF="$CACHE_SUF-rt"
CACHE=${CACHE:-$HOME/.radiance-cache-moe-$(echo "$MODEL_ID" | tr '/' '_')$CACHE_SUF}
mkdir -p "$CACHE"/vllm "$CACHE"/inductor "$CACHE"/triton "$CACHE"/aiter

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker, then re-run"; fi
fi
RT_FLAGS=(); GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  RT_FLAGS+=(--replace); GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    [ -n "$gid" ] && GROUP_FLAGS+=(--group-add "$gid")
  done
  [ -z "${DRY_RUN:-}" ] && "$RUNTIME" rm -f "$NAME" >/dev/null 2>&1
fi
[ "${DETACH:-0}" = 1 ] && RT_FLAGS+=(-d)

TIER=()
if [ "$RAM_TIER_BYTES" != 0 ]; then
  shm_avail=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
  if [ -n "$shm_avail" ] && [ "$shm_avail" -le "$RAM_TIER_BYTES" ]; then
    die "RAM_TIER_BYTES=$RAM_TIER_BYTES does not fit in /dev/shm ($shm_avail bytes free)"
  fi
  TIER=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$RAM_TIER_BYTES}}")
fi
SPEC_ARGS=()
if [ "$SPEC" != 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":true}")
fi
# patch_offload_mamba_eagle.py is what makes the RAM tier load on a hybrid model under speculation,
# so it runs with or without the MoE fixes (inert when no connector is configured).
if [ "$MOE_FIXES" = 1 ]; then
  PRE="python3 patch_quark_moe_w4a16.py && python3 patch_gfx12_aiter_a16w4.py && python3 patch_attn_3d_multiq.py && python3 patch_offload_mamba_eagle.py"
  FIX_ENV=(-e RADIANCE_MOE_W4A16=1 -e RADIANCE_MOE_BACKEND=aiter -e RADIANCE_ATTN_3D_MAX_Q=16)
else
  PRE="python3 patch_offload_mamba_eagle.py"
  FIX_ENV=(-e RADIANCE_MOE_W4A16=0)
fi

echo "[serve-moe] $MODEL_ID ($KIND) fixes=$MOE_FIXES ram_tier=$RAM_TIER_BYTES spec=$SPEC runtime=$RUNTIME cache=$CACHE"
exec ${DRY_RUN:+echo} "$RUNTIME" run --rm "${RT_FLAGS[@]}" --name "$NAME" --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
  -v "$MOUNT":"$MOUNT":ro -v "$CACHE":/cache -v "$SCRIPT_DIR":/patches:ro \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e ROCR_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_ROCM_USE_AITER_MHA=0 \
  -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_ROCM_USE_AITER_LINEAR=0 \
  -e VLLM_ROCM_USE_AITER_FP8BMM=0 -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  "${FIX_ENV[@]}" -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e RADIANCE_PRE="$PRE" --entrypoint bash "$IMAGE" \
  -c 'cd /patches && eval "$RADIANCE_PRE" && cd / && exec /opt/radiance_entrypoint.sh "$@"' _ \
  "$SNAP" --served-model-name $SERVED_NAMES --host 0.0.0.0 --port "$PORT" \
  --max-num-seqs "$MAXSEQS" --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-batched-tokens "$CHUNK" --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
  --enable-prefix-caching --mamba-cache-mode align ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72]}' \
  --no-async-scheduling --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code ${TIER[@]+"${TIER[@]}"} "$@"
