#!/bin/bash
# One-time setup for the MXFP4 serve. Checks the host, pulls the image, fetches AMD's checkpoint,
# builds the loadable form of it, fetches the drafter, and compiles libr4d -- then tells you the
# one command that starts the server.
#
# Safe to re-run: every step checks whether its output already exists and skips it. Nothing here
# is destructive, and nothing writes outside $MODELS, the HF cache and ~/.cache/radiance-libr4d.
set -euo pipefail

MODELS=${MODELS:-$HOME/models}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
SRC_REPO=${SRC_REPO:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}
DRAFT_REPO=${DRAFT_REPO:-tcclaviger/Qwen3.8-27B-DFlash2-FP8}
SNAP=${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-FP8}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

WANT_DRAFTER=1
ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    -h|--help)
      cat <<'USAGE'
setup-mxfp4.sh -- one-time setup for the MXFP4 serve

  ./setup-mxfp4.sh              run every step that is not already done
  ./setup-mxfp4.sh --yes        don't ask before downloading (~40 GiB)
  ./setup-mxfp4.sh --no-drafter skip the DFlash2 drafter (then serve with SPEC_METHOD=mtp)

Environment:
  MODELS=~/models               where the checkpoints are written
  HF_CACHE=~/.cache/huggingface where the source download lands
  IMAGE=...:0.9.3               container image to use
  RUNTIME=podman|docker         container runtime (auto-detected)

Disk: ~19 GiB for AMD's release, ~19 GiB for the checkpoint built from it, 2 GiB for the
drafter and ~10 GiB for the image. The source download can be deleted afterwards.
USAGE
      exit 0 ;;
    --yes|-y)      ASSUME_YES=1 ;;
    --no-drafter)  WANT_DRAFTER=0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

step() { echo; echo "=== $* ==="; }
ok()   { echo "  ok: $*"; }
die()  { echo "ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

# ------------------------------------------------------------------ 1. host
step "1/6  host"

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker, then re-run"
  fi
fi
ok "container runtime: $RUNTIME"

[ -e /dev/kfd ] || die "/dev/kfd is missing -- the amdgpu kernel driver is not loaded" \
    "ROCm userspace ships inside the image, but the kernel driver must be on the host"
[ -d /dev/dri ] || die "/dev/dri is missing -- no GPU render nodes on this host"

gpus=0
for d in /sys/class/drm/renderD*; do
  if [ "$(cat "$d/device/vendor" 2>/dev/null)" = "0x1002" ]; then gpus=$((gpus+1)); fi
done
if [ "$gpus" -lt 2 ]; then
  echo "  WARNING: found $gpus AMD GPU(s). The serve runs tensor-parallel across two R9700s and"
  echo "  will not start with fewer. Setup can still prepare everything else."
else
  ok "$gpus AMD GPUs visible"
fi

command -v git >/dev/null 2>&1 || die "git is required (the libr4d build clones it)"

free_gib=$(df -BG --output=avail "$(dirname "$MODELS")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gib" ] && [ "$free_gib" -lt 60 ]; then
  echo "  WARNING: ${free_gib} GiB free on $(dirname "$MODELS"); a full setup wants about 60 GiB"
  echo "  (19 source + 19 built + 2 drafter + ~10 image). Delete the source download when done."
else
  ok "${free_gib:-?} GiB free"
fi

mkdir -p "$MODELS" "$HF_CACHE"

if [ "$ASSUME_YES" = 0 ] && { [ ! -d "$SNAP" ] || { [ "$WANT_DRAFTER" = 1 ] && [ ! -d "$DRAFTER" ]; }; }; then
  echo
  echo "This will download roughly 40 GiB into $HF_CACHE and $MODELS."
  read -r -p "Continue? [y/N] " reply
  case "$reply" in y|Y|yes|YES) ;; *) echo "aborted"; exit 1 ;; esac
fi

# ------------------------------------------------------------------ 2. image
step "2/6  container image"
if "$RUNTIME" image exists "$IMAGE" >/dev/null 2>&1 || "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "$IMAGE already present"
else
  echo "  pulling $IMAGE (a few GiB)"
  "$RUNTIME" pull "$IMAGE"
fi

# Download inside the image rather than on the host: it already has huggingface_hub, so the host
# needs no Python environment of its own for any of this.
hf_get() { # repo [local-dir]
  local repo="$1" dest="${2:-}"
  "$RUNTIME" run --rm --network=host \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$MODELS":/models \
    --entrypoint python3 "$IMAGE" -c '
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], (sys.argv[2] or None)
p = snapshot_download(repo_id=repo, local_dir=dest)
print(p)
' "$repo" "${dest:-}"
}

# ------------------------------------------------------------------ 3. source checkpoint
step "3/6  AMD's MXFP4 release ($SRC_REPO)"
if [ -d "$SNAP" ] && [ -f "$SNAP/config.json" ]; then
  ok "skipped -- the built checkpoint already exists at $SNAP"
  SRC=""
else
  echo "  downloading ~19 GiB into $HF_CACHE (resumes if interrupted)"
  hf_get "$SRC_REPO" >/dev/null
  SRC=$(ls -d "$HF_CACHE"/hub/models--${SRC_REPO//\//--}/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$SRC" ] || die "download finished but no snapshot directory under $HF_CACHE/hub"
  ok "source snapshot: $SRC"
fi

# ------------------------------------------------------------------ 4. build the loadable checkpoint
step "4/6  build $SNAP"
if [ -f "$SNAP/config.json" ]; then
  ok "already built"
else
  echo "  requantizing the MTP head to fp8 (~15 minutes, one file rewritten)."
  echo "  This is not optional: AMD's release leaves mtp.* out of both 'exclude' and"
  echo "  'layer_quant_config', so vLLM applies the mxfp4 scheme to a bf16 head and asserts"
  echo "  on a half-width parameter at load."
  case "$SNAP" in
    "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
    *) die "SNAP ($SNAP) must live under MODELS ($MODELS)" ;;
  esac
  CSRC="/root/.cache/huggingface/${SRC#"$HF_CACHE"/}"
  if python3 -c "import torch" >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/fp8_mtp.py" "$SRC" "$SNAP"
  else
    "$RUNTIME" run --rm \
      -v "$HF_CACHE":/root/.cache/huggingface \
      -v "$MODELS":/models \
      -v "$SCRIPT_DIR":/repo:z \
      --entrypoint python3 "$IMAGE" /repo/fp8_mtp.py "$CSRC" "$CSNAP"
  fi
  [ -f "$SNAP/config.json" ] || die "fp8_mtp.py did not produce $SNAP/config.json"
  ok "built"
fi

# ------------------------------------------------------------------ 5. drafter
step "5/6  speculative drafter"
if [ "$WANT_DRAFTER" = 0 ]; then
  echo "  skipped (--no-drafter). Serve with: SPEC_METHOD=mtp ./serve-mxfp4.sh"
elif [ -f "$DRAFTER/config.json" ]; then
  ok "already present at $DRAFTER"
else
  echo "  downloading $DRAFT_REPO (2 GiB)"
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) die "DRAFTER ($DRAFTER) must live under MODELS ($MODELS)" ;;
  esac
  hf_get "$DRAFT_REPO" "$CDRAFTER" >/dev/null
  [ -f "$DRAFTER/config.json" ] || die "drafter download did not produce $DRAFTER/config.json"
  ok "downloaded"
fi

# ------------------------------------------------------------------ 6. kernels
step "6/6  libr4d kernels"
echo "  building the pinned libr4d inside the image (once, a few minutes; cached afterwards)."
echo "  The kernel shipped in the image predates the gated-delta-net overflow fix and NaNs this"
echo "  model's output, so this build is load-bearing, not an optimization."
MODELS="$MODELS" IMAGE="$IMAGE" RUNTIME="$RUNTIME" PREPARE_ONLY=1 "$SCRIPT_DIR/serve-mxfp4.sh"

cat <<EOF

=== setup complete ===

Start the server:

    ./serve-mxfp4.sh

It listens on http://localhost:8080/v1 as "Qwen3.8". The first start compiles Triton and inductor
kernels and takes several extra minutes; later starts reuse that cache. Then:

    curl http://localhost:8080/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'

Every knob:  ./serve-mxfp4.sh --help
EOF
if [ -n "${SRC:-}" ]; then
  echo
  echo "You can reclaim ~19 GiB now -- the source download is no longer needed:"
  echo "    rm -rf $HF_CACHE/hub/models--${SRC_REPO//\//--}"
fi
