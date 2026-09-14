#!/bin/bash
# docker-compose-setup.sh -- make `docker compose up -d` work here without editing anything.
#
#   ./docker-compose-setup.sh              write .env for this host, then print the next command
#   ./docker-compose-setup.sh --download   also fetch the checkpoint the compose file serves
#   ./docker-compose-setup.sh --show       print what it would write and change nothing
#
# docker-compose.yml serves the plain FP8 checkpoint (no MXFP4, no patch prelude) and is the
# compose-shaped path for the FP8 and Gemma models. It needs three things that are properties of
# YOUR host and cannot be defaults in a committed file: the numeric GIDs of the render and video
# groups (docker, unlike rootless podman, has no `--group-add keep-groups`, so the GPU device
# nodes are unreachable without them), which HIP indices to serve on, and where the checkpoints
# live. This writes all three to .env, which docker compose reads on its own.
#
# For MXFP4 -- the fast path, and what this repo is about -- use ./docker-quickstart.sh instead.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.8-27B-FP8}
ENV_FILE=${ENV_FILE:-$SCRIPT_DIR/.env}
RUNTIME=${RUNTIME:-}

DOWNLOAD=0
SHOW_ONLY=0
for a in "$@"; do
  case "$a" in
    --download) DOWNLOAD=1 ;;
    --show)     SHOW_ONLY=1 ;;
    -h|--help)
      cat <<'USAGE'
docker-compose-setup.sh -- prepare docker-compose.yml for this host

  ./docker-compose-setup.sh             write .env (GPU group ids, HIP indices, TP, paths)
  ./docker-compose-setup.sh --download  also download the FP8 checkpoint (tens of GiB)
  ./docker-compose-setup.sh --show      print what it would write, change nothing

Then:
  docker compose up -d          start        docker compose logs -f     follow
  docker compose down           stop         docker compose restart     restart

Environment:
  MODELS=~/models                  where checkpoints live (mounted at /models)
  MODEL_REPO=Qwen/Qwen3.8-27B-FP8  which checkpoint --download fetches
  IMAGE=stilldeadcode/vllm-radiance:0.9.3
  ENV_FILE=./.env                  where to write

The compose file serves FP8, not MXFP4. For MXFP4 use ./docker-quickstart.sh.
USAGE
      exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then B=$'\033[1m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
else B=""; GRN=""; YEL=""; RST=""; fi
ok()   { echo "  ${GRN}ok${RST}  $*"; }
warn() { echo "  ${YEL}note${RST}  $*"; }
die()  { echo "ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

# ---------------------------------------------------------------- GPU group ids
# Without these the container sees /dev/kfd and /dev/dri but cannot open them, and the failure
# is an HSA error from deep inside torch rather than a permission message.
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3 || true)
VIDEO_GID=$(getent group video  2>/dev/null | cut -d: -f3 || true)
[ -n "$RENDER_GID" ] || die "this host has no 'render' group" \
    "that group owns /dev/dri/renderD*; on most distributions it comes with the amdgpu driver" \
    "check with: ls -l /dev/dri/  and  getent group"
[ -n "$VIDEO_GID" ] || VIDEO_GID=$RENDER_GID

# ---------------------------------------------------------------- GPUs and TP
# Same scan the MXFP4 launcher uses, so both paths agree about the hardware.
# shellcheck source=gpu-detect.sh
. "$SCRIPT_DIR/gpu-detect.sh"
[ "$RAD_GPU_COUNT" -ge 1 ] || die "no AMD GPU with at least ${RAD_MIN_GPU_MIB} MiB of VRAM" \
    "found:${RAD_GPU_SKIPPED:- nothing on the amdgpu driver}" \
    "this image is compiled for gfx1201 (RDNA4) only"

NEW_ENV=$(cat <<EOF
# Written by docker-compose-setup.sh on $(date -Iseconds) for this host.
# docker compose reads this file automatically; edit it freely, or re-run the script to redo it.

# GPU device access. docker needs numeric group ids here -- rootless podman does not.
RENDER_GID=$RENDER_GID
VIDEO_GID=$VIDEO_GID

# Which cards to serve on, and across how many. $RAD_GPU_COUNT usable card(s) detected
# ($RAD_GPU_NAME, $RAD_GPU_MIB MiB each); tensor parallel must divide the model's head counts.
GPUS=$RAD_GPU_INDICES
TP=$RAD_TP

# Paths. MODELS is mounted at /models, so MODEL_PATH is the checkpoint seen from inside the
# container; point both at a different model and nothing in docker-compose.yml has to change.
# The cache keeps compiled kernels between restarts, which is what makes every start after the
# first one a short one.
MODELS=$MODELS
MODEL_PATH=/models/$MODEL_REPO
VLLM_CACHE=$SCRIPT_DIR/vllm-cache
WORK=$SCRIPT_DIR

IMAGE=$IMAGE
EOF
)

echo "${B}docker-compose-setup${RST} -- compose configuration for this host"
echo
ok "render group id $RENDER_GID, video group id $VIDEO_GID"
ok "$RAD_GPU_COUNT x $RAD_GPU_NAME ($RAD_GPU_MIB MiB each) -> GPUS=$RAD_GPU_INDICES, TP=$RAD_TP"
[ -n "$RAD_GPU_SKIPPED" ] && warn "skipped as too small:$RAD_GPU_SKIPPED"

if [ "$SHOW_ONLY" = 1 ]; then
  echo
  echo "$NEW_ENV"
  exit 0
fi

# Never silently overwrite: a .env in this directory is deploy state someone may have tuned.
if [ -f "$ENV_FILE" ]; then
  cp "$ENV_FILE" "$ENV_FILE.bak"
  warn "kept your previous .env as .env.bak"
fi
printf '%s\n' "$NEW_ENV" > "$ENV_FILE"
ok "wrote $ENV_FILE"

# ---------------------------------------------------------------- the checkpoint
MODEL_DIR=$MODELS/$MODEL_REPO
if [ -f "$MODEL_DIR/config.json" ]; then
  ok "checkpoint present at $MODEL_DIR"
elif [ "$DOWNLOAD" = 1 ]; then
  if [ -z "$RUNTIME" ]; then
    if   command -v docker >/dev/null 2>&1; then RUNTIME=docker
    elif command -v podman >/dev/null 2>&1; then RUNTIME=podman
    else die "no container runtime found to download with" "install docker or podman"
    fi
  fi
  echo
  echo "  downloading $MODEL_REPO into $MODEL_DIR (tens of GiB; resumes if interrupted)"
  mkdir -p "$MODELS" "$HF_CACHE"
  # Downloaded inside the image, which already has huggingface_hub -- the host needs no Python
  # environment of its own. Same approach as setup-mxfp4.sh.
  "$RUNTIME" run --rm --network=host \
    -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$HF_CACHE":/root/.cache/huggingface -v "$MODELS":/models \
    --entrypoint python3 "$IMAGE" -c '
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2]))
' "$MODEL_REPO" "/models/$MODEL_REPO" >/dev/null
  [ -f "$MODEL_DIR/config.json" ] || die "download finished but $MODEL_DIR/config.json is missing"
  ok "downloaded"
else
  warn "no checkpoint at $MODEL_DIR yet -- compose will not start without it"
  echo "        fetch it with:  ./docker-compose-setup.sh --download"
  echo "        or serve a checkpoint you already have: MODEL_REPO=org/name ./docker-compose-setup.sh"
fi

cat <<EOF

${B}=== ready ===${RST}

  docker compose up -d        start it (detached)
  docker compose logs -f      follow the log -- a first start compiles kernels for a while
  docker compose down         stop and remove
  docker compose restart      restart in place

  It listens on http://localhost:8000/v1 as "qwen3.8-27b". Override any knob in .env rather
  than editing docker-compose.yml; the file's comments say what each one costs.

  For MXFP4 -- 4-bit weights, the speculative drafter, and every number in README.md --
  use ./docker-quickstart.sh instead. The two serve different checkpoints.
EOF
