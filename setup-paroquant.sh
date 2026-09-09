#!/bin/bash
# One-time setup for the ParoQuant serve. Checks the host, pulls the image, fetches the PARO
# checkpoint, fetches the drafter, and compiles libr4d -- then tells you the one command that
# starts the server.
#
# Shares the image, the DFlash2 drafter and libr4d with setup-mxfp4.sh, so running both costs one
# download of each. Safe to re-run: every step checks whether its output already exists and skips
# it. Nothing here is destructive, and nothing writes outside $MODELS, the HF cache and
# ~/.cache/radiance-libr4d.
#
# Unlike AMD's MXFP4 release, the PARO checkpoint needs no rewrite to be loadable -- there is no
# build step here. It also ships no MTP/drafter tensors, so the external DFlash2-FP8 drafter is
# what provides speculative decoding.
set -euo pipefail

MODELS=${MODELS:-$HOME/models}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
SRC_REPO=${SRC_REPO:-z-lab/Qwen3.8-27B-PARO}
DRAFT_REPO=${DRAFT_REPO:-tcclaviger/Qwen3.8-27B-DFlash2-FP8}
SNAP=${SNAP:-$MODELS/Qwen3.8-27B-PARO}
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-FP8}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

WANT_DRAFTER=1
ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    -h|--help)
      cat <<'USAGE'
setup-paroquant.sh -- one-time setup for the ParoQuant W4A8 serve

  ./setup-paroquant.sh              run every step that is not already done
  ./setup-paroquant.sh --yes        don't ask before downloading (~21 GiB)
  ./setup-paroquant.sh --no-drafter skip the DFlash2 drafter (then serve with MODE=eval)

Environment:
  MODELS=~/models               where the checkpoint is written
  HF_CACHE=~/.cache/huggingface where huggingface_hub keeps its cache
  IMAGE=...:0.9.3               container image to use
  RUNTIME=podman|docker         container runtime (auto-detected)

Disk: ~19 GiB for the PARO checkpoint, 2 GiB for the drafter and ~10 GiB for the image. The
checkpoint is downloaded straight into $MODELS, so there is no second copy to delete afterwards.
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
step "1/5  host"

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

# Same scan the launchers use, so setup and serve cannot disagree about what hardware is here.
# shellcheck source=gpu-detect.sh
. "$SCRIPT_DIR/gpu-detect.sh"
if [ "$RAD_GPU_COUNT" -lt 1 ]; then
  echo "  WARNING: no AMD GPU with at least ${RAD_MIN_GPU_MIB} MiB of VRAM."
  echo "  Found:${RAD_GPU_SKIPPED:- nothing on the amdgpu driver}. Setup can still prepare everything else."
else
  ok "$RAD_GPU_COUNT AMD GPU(s) usable: $RAD_GPU_NAME, $RAD_GPU_MIB MiB each -> tensor-parallel $RAD_TP"
  if [ -n "$RAD_GPU_SKIPPED" ]; then
    echo "  (skipped as too small:$RAD_GPU_SKIPPED)"
  fi
fi

command -v git >/dev/null 2>&1 || die "git is required (the libr4d build clones it)"

free_gib=$(df -BG --output=avail "$(dirname "$MODELS")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gib" ] && [ "$free_gib" -lt 32 ]; then
  echo "  WARNING: ${free_gib} GiB free on $(dirname "$MODELS"); a full setup wants about 32 GiB"
  echo "  (19 checkpoint + 2 drafter + ~10 image)."
else
  ok "${free_gib:-?} GiB free"
fi

mkdir -p "$MODELS" "$HF_CACHE"

if [ "$ASSUME_YES" = 0 ] && { [ ! -f "$SNAP/config.json" ] || { [ "$WANT_DRAFTER" = 1 ] && [ ! -f "$DRAFTER/config.json" ]; }; }; then
  echo
  echo "This will download roughly 21 GiB into $MODELS."
  read -r -p "Continue? [y/N] " reply
  case "$reply" in y|Y|yes|YES) ;; *) echo "aborted"; exit 1 ;; esac
fi

# ------------------------------------------------------------------ 2. image
step "2/5  container image"
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

# ------------------------------------------------------------------ 3. checkpoint
step "3/5  PARO checkpoint ($SRC_REPO)"
case "$SNAP" in
  "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
  *) die "SNAP ($SNAP) must live under MODELS ($MODELS)" ;;
esac
if [ -f "$SNAP/config.json" ]; then
  ok "already present at $SNAP"
else
  echo "  downloading ~19 GiB straight into $SNAP (resumes if interrupted)"
  hf_get "$SRC_REPO" "$CSNAP" >/dev/null
  [ -f "$SNAP/config.json" ] || die "download did not produce $SNAP/config.json"
  ok "downloaded"
fi

# The kernels are built for exactly this shape: group 128 is baked into the slab structure, and
# the prologue's rotation table is sized for krot <= 8. Fail here with the reason rather than at
# model load with an assertion, or worse, at the first token.
python3 - "$SNAP/config.json" <<'PY' || die "checkpoint is not servable by this stack" \
    "the radiance ParoQuant kernels are built for quant_method=paroquant, bits=4, group_size=128, krot<=8"
import json, sys
q = (json.load(open(sys.argv[1])).get("quantization_config") or {})
m, b, g, k = q.get("quant_method"), q.get("bits"), q.get("group_size"), q.get("krot")
print(f"  quantization_config: quant_method={m} bits={b} group_size={g} krot={k}")
bad = [n for n, v, want in (("quant_method", m, "paroquant"), ("bits", b, 4), ("group_size", g, 128))
       if v != want]
if not isinstance(k, int) or not 1 <= k <= 8:
    bad.append("krot")
if bad:
    print("  unsupported: " + ", ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
ok "checkpoint declares a servable paroquant config"

# ------------------------------------------------------------------ 4. drafter
step "4/5  speculative drafter"
if [ "$WANT_DRAFTER" = 0 ]; then
  echo "  skipped (--no-drafter). Serve without speculative decoding: MODE=eval ./paroquant/run_paroquant.sh"
elif [ -f "$DRAFTER/config.json" ]; then
  ok "already present at $DRAFTER (shared with the MXFP4 setup)"
else
  echo "  downloading $DRAFT_REPO (2 GiB)"
  echo "  The PARO checkpoint ships no MTP tensors, so this external drafter is the only"
  echo "  speculative path -- MODE=prod needs it."
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) die "DRAFTER ($DRAFTER) must live under MODELS ($MODELS)" ;;
  esac
  hf_get "$DRAFT_REPO" "$CDRAFTER" >/dev/null
  [ -f "$DRAFTER/config.json" ] || die "drafter download did not produce $DRAFTER/config.json"
  ok "downloaded"
fi

# ------------------------------------------------------------------ 5. kernels
step "5/5  libr4d kernels"
echo "  building the pinned libr4d inside the image (once, a few minutes; cached afterwards)."
echo "  The kernel shipped in the image predates the gated-delta-net overflow fix and NaNs this"
echo "  model's output, so this build is load-bearing, not an optimization."
echo "  The ParoQuant kernel module itself is compiled at container start by run_paroquant.sh,"
echo "  so there is nothing to build for it here."
MODELS="$MODELS" IMAGE="$IMAGE" RUNTIME="$RUNTIME" PREPARE_ONLY=1 "$SCRIPT_DIR/serve-mxfp4.sh"

cat <<EOF

=== setup complete ===

Start the server:

    MODE=prod SPEC=7 ./paroquant/run_paroquant.sh

It listens on http://localhost:8080/v1 as "Qwen3.8-PARO". The first start compiles the ParoQuant
kernel, then Triton and inductor kernels, and takes several extra minutes; later starts reuse that
cache. Then:

    curl http://localhost:8080/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"Qwen3.8-PARO","messages":[{"role":"user","content":"Hello!"}]}'

To gate numerics instead of serving, MODE=eval runs eager with the in-serve kernel-vs-fp32
comparison on every quantized shape:

    MODE=eval ./paroquant/run_paroquant.sh

The format, the kernels and every knob:  PAROQUANT.md
EOF
