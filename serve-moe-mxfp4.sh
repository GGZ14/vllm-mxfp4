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
#   MOE_PREFILL_ATTN=r4d  prefill attention with MOE_FIXES=1 (patch_moe_prefill_attn.py):
#                     r4d: R4D backend for the model and the MTP drafter; libr4d's prefill kernel built at
#                          GQA 8 (moe-r4d/) serves runs >= 17 tokens, stock Triton split-KV the rest
#                          (decode, verify). ~6x lower TTFT at 16k. Needs a libr4d checkout (below).
#                     off: TRITON_ATTN for everything, as before.
#   RADIANCE_GDN_SCAN_FIX=1  1: exact GDN chunk scan (moe-gdn2/, patch_gdn_scan_fix.py) in place of libr4d
#                     v0.5.0's, which is wrong for any (sequence, head) whose in-chunk decay span exceeds
#                     160 (libr4d #4; about 7% of GDN heads on real prompts). Same prefill speed. Built at
#                     container start from a patch against the libr4d checkout (R4D_SRC), so it needs that
#                     checkout with or without MOE_PREFILL_ATTN=r4d, and applies with or without MOE_FIXES.
#                     0: libr4d's scan as is.
#   RADIANCE_MOE_W4A8=1  1: expert GEMM calls of >= RADIANCE_MOE_W4A8_MIN_TOKENS tokens (default 1025: prefill
#                     chunks) run on an fp8-WMMA W4A8 grouped kernel (moe-w4a8/, patch_moe_w4a8.py); decode,
#                     MTP verify and CUDA-graph calls stay on a16w4. ~+25% prefill. Rides on the a16w4 lane,
#                     so it needs MOE_FIXES=1 and is ignored with MOE_FIXES=0. 0: a16w4 only.
#   R4D_SRC=~/.radiance-libr4d-<R4D_VERSION>  libr4d checkout for r4d and the GDN scan fix; cloned from
#                     R4D_REPO at R4D_VERSION (v0.5.0, the tag the image's r4d.so is built from) when missing
#   RADIANCE_HW_QUEUES=1  GPU_MAX_HW_QUEUES for the container (see serve-mxfp4.sh); 0 = HIP default
#   SPEC=4            MTP draft tokens; 0 disables speculation. In align mode each request pins 2 + SPEC GDN
#                     state blocks, so SPEC sets how many requests fit: SPEC=4 fits 15 short ones at
#                     MAXSEQS=16 and GPU_UTIL=0.97, SPEC=8 fits 7. Sections 1-3 of MOE-GFX1201.md were
#                     measured at SPEC=8, MAXSEQS=8, GPU_UTIL=0.95.
#   MAXLEN=65536  MAXSEQS=16  CHUNK=4096  GPU_UTIL=0.97  PORT=8080  NAME=radiance-moe  GPUS=0
#   IMAGE=stilldeadcode/vllm-radiance:0.9.3   RUNTIME=podman|docker (auto)
#   CACHE=~/.radiance-cache-moe-<model>-f<fixes>[-pa<mode>][-gdn2][-w4a8]   compile cache; never share one across knobs
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

MOE_FIXES=${MOE_FIXES:-1}; SPEC=${SPEC:-4}
MAXLEN=${MAXLEN:-65536}; MAXSEQS=${MAXSEQS:-16}; CHUNK=${CHUNK:-4096}; GPU_UTIL=${GPU_UTIL:-0.97}
PORT=${PORT:-8080}; NAME=${NAME:-radiance-moe}; GPUS=${GPUS:-0}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
PA=${MOE_PREFILL_ATTN:-r4d}; HWQ=${RADIANCE_HW_QUEUES:-1}
GDNFIX=${RADIANCE_GDN_SCAN_FIX:-1}; W4A8=${RADIANCE_MOE_W4A8:-1}; W4A8_MIN=${RADIANCE_MOE_W4A8_MIN_TOKENS:-}
case "$PA" in off|r4d) ;; *) die "MOE_PREFILL_ATTN must be r4d or off (got $PA)" ;; esac
case "$GDNFIX" in 0|1) ;; *) die "RADIANCE_GDN_SCAN_FIX must be 0 or 1 (got $GDNFIX)" ;; esac
case "$W4A8" in 0|1) ;; *) die "RADIANCE_MOE_W4A8 must be 0 or 1 (got $W4A8)" ;; esac
case "$W4A8_MIN" in ''|*[!0-9]*) [ -z "$W4A8_MIN" ] || die "RADIANCE_MOE_W4A8_MIN_TOKENS must be an integer (got $W4A8_MIN)" ;; esac
case "$SPEC" in ''|*[!0-9]*) die "SPEC must be a non-negative integer (got $SPEC)" ;; esac
case "$MAXSEQS" in ''|*[!0-9]*|0) die "MAXSEQS must be a positive integer (got $MAXSEQS)" ;; esac
[ "$MOE_FIXES" = 1 ] || PA=off   # measured only on top of the MoE fixes
[ "$MOE_FIXES" = 1 ] || W4A8=0   # W4A8 rides on the a16w4 lane that MOE_FIXES=1 enables
R4D_REPO=${R4D_REPO:-https://codeberg.org/StillDeadcode/libr4d.git}; R4D_VERSION=${R4D_VERSION:-v0.5.0}
R4D_SRC=${R4D_SRC:-$HOME/.radiance-libr4d-$R4D_VERSION}

# A Hugging Face cache snapshot holds symlinks into ../../blobs: mount the whole hub dir, at the same
# path, so they resolve. Anything else mounts just the checkpoint.
case "$SNAP" in */models--*/snapshots/*) MOUNT=${SNAP%%/models--*} ;; *) MOUNT=$SNAP ;; esac
MODEL_ID=$(basename "$SNAP")
case "$SNAP" in */snapshots/*) MODEL_ID=$(basename "$(dirname "$(dirname "$SNAP")")" | sed 's/^models--//; s#--#/#') ;; esac
SERVED_NAMES=${SERVED_NAMES:-$(basename "$MODEL_ID")}

CACHE_SUF="-f$MOE_FIXES"; [ "$PA" != off ] && CACHE_SUF="$CACHE_SUF-pa$PA"
[ "$GDNFIX" = 1 ] && CACHE_SUF="$CACHE_SUF-gdn2"
[ "$W4A8" = 1 ] && CACHE_SUF="$CACHE_SUF-w4a8"
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

BACKEND=TRITON_ATTN; R4D_MNT=()
[ "$PA" = r4d ] && BACKEND=R4D
# The libr4d checkout is read by both the GQA-8 prefill build (r4d) and the exact GDN scan build.
if [ "$PA" = r4d ] || [ "$GDNFIX" = 1 ]; then
  if [ ! -f "$R4D_SRC/r4d_attn_prefill_h256_gqa6.hip" ] || [ ! -f "$R4D_SRC/r4d_gdn_wmma.h" ]; then
    echo "[serve-moe] cloning libr4d $R4D_VERSION into $R4D_SRC (once)"
    git clone -q --depth 1 -b "$R4D_VERSION" "$R4D_REPO" "$R4D_SRC" \
      || die "could not clone $R4D_REPO at $R4D_VERSION" "point R4D_SRC at a libr4d $R4D_VERSION checkout, or set MOE_PREFILL_ATTN=off RADIANCE_GDN_SCAN_FIX=0"
  fi
  R4D_MNT=(-v "$(cd "$R4D_SRC" && pwd)":/r4dsrc:ro)
fi
SPEC_ARGS=()
if [ "$SPEC" != 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$BACKEND\",\"disable_padded_drafter_batch\":true}")
fi
PRE=""; pre_add() { PRE="${PRE:+$PRE && }$1"; }
if [ "$MOE_FIXES" = 1 ]; then
  PRE="python3 patch_quark_moe_w4a16.py && python3 patch_gfx12_aiter_a16w4.py && python3 patch_attn_3d_multiq.py"
  FIX_ENV=(-e RADIANCE_MOE_W4A16=1 -e RADIANCE_MOE_BACKEND=aiter -e RADIANCE_ATTN_3D_MAX_Q=16)
  if [ "$PA" = r4d ]; then
    # build the GQA-8 prefill module with the image's hipcc (a few seconds), then install it
    PRE="R4D_SRC=/r4dsrc OUT=/cache/r4d_moe.so bash moe-r4d/build.sh && $PRE && python3 patch_moe_prefill_attn.py"
    FIX_ENV+=(-e RADIANCE_MOE_PREFILL_ATTN=r4d -e RADIANCE_R4D_MOE_SO=/cache/r4d_moe.so)
  fi
else
  FIX_ENV=(-e RADIANCE_MOE_W4A16=0)
fi
# Both new kernels are built straight into the image's site-packages, which is also where the patches
# below edit the installed vLLM / radiance sources (never a copy that sits next to the patch scripts).
if [ "$GDNFIX" = 1 ] || [ "$W4A8" = 1 ]; then
  pre_add "SP=\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
fi
if [ "$GDNFIX" = 1 ]; then
  # libr4d #4: libr4d v0.5.0's GDN chunk scan without the midpoint decay split. moe-gdn2/build.sh patches a
  # temporary copy of R4D_SRC's scan (no libr4d source in this repo), builds it, and the patch binds it.
  pre_add "R4D_SRC=/r4dsrc OUT=\$SP/radiance_gdn2.so bash moe-gdn2/build.sh && python3 patch_gdn_scan_fix.py"
  FIX_ENV+=(-e RADIANCE_GDN_SCAN_FIX=1)
fi
if [ "$W4A8" = 1 ]; then
  # fp8-WMMA W4A8 expert GEMMs for prefill-sized calls; anchors on what patch_gfx12_aiter_a16w4.py inserts
  pre_add "OUT=\$SP bash moe-w4a8/build.sh && python3 patch_moe_w4a8.py"
  FIX_ENV+=(-e RADIANCE_MOE_W4A8=1)
  [ -n "$W4A8_MIN" ] && FIX_ENV+=(-e RADIANCE_MOE_W4A8_MIN_TOKENS="$W4A8_MIN")
fi

HWQ_ENV=(); [ "$HWQ" != 0 ] && HWQ_ENV=(-e GPU_MAX_HW_QUEUES="$HWQ")
echo "[serve-moe] $MODEL_ID ($KIND) fixes=$MOE_FIXES prefill_attn=$PA backend=$BACKEND gdn_scan_fix=$GDNFIX moe_w4a8=$W4A8 hw_queues=$HWQ spec=$SPEC max_seqs=$MAXSEQS gpu_util=$GPU_UTIL runtime=$RUNTIME cache=$CACHE"
exec ${DRY_RUN:+echo} "$RUNTIME" run --rm "${RT_FLAGS[@]}" --name "$NAME" --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
  -v "$MOUNT":"$MOUNT":ro -v "$CACHE":/cache -v "$SCRIPT_DIR":/patches:ro ${R4D_MNT[@]+"${R4D_MNT[@]}"} \
  ${HWQ_ENV[@]+"${HWQ_ENV[@]}"} -e HIP_VISIBLE_DEVICES="$GPUS" -e ROCR_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
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
  --max-num-batched-tokens "$CHUNK" --kv-cache-dtype fp8 --attention-backend "$BACKEND" \
  --enable-prefix-caching --mamba-cache-mode align ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72]}' \
  --no-async-scheduling --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code "$@"
