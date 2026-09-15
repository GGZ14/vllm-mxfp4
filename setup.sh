#!/usr/bin/env bash
# setup.sh — downstream-facing one-shot setup for the radiance image.
#
# Pulls the published radiance image, fetches the checkpoint for the chosen
# quant format, and (optionally) the DFlash2-FP8 drafter. That's it. No git
# clone, no host build, no libr4d, no hipcc. Everything is baked into the
# image at build time (see build.sh + Dockerfile.ggz14 on the
# maintainer side).
#
# Usage:
#   ./setup.sh                          # default: MXFP4 (quark_mxfp4)
#   ./setup.sh --int4                   # ParoQuant int4 W4A8 (z-lab)
#   ./setup.sh --int5                   # ParoQuant int5 W5A8 (Launch80, ~21 GiB)
#   ./setup.sh --no-drafter             # skip the drafter (then --mtp-mode)
#
# Environment:
#   IMAGE=...                           # image to pull (default: radiance:latest)
#   MODELS=~/models                     # where checkpoints land
#
# Disk: 19-21 GiB checkpoint + 2 GiB drafter + ~10 GiB image. The
# downloaded-from-HuggingFace directories are full snapshots, so they are
# reusable for `serve.sh` and any future rerun.

set -euo pipefail

QUANT=mxfp4
WANT_DRAFTER=1
ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    --int4)         QUANT=int4 ;;
    --int5)         QUANT=int5 ;;
    --mxfp4)        QUANT=mxfp4 ;;
    --no-drafter)   WANT_DRAFTER=0 ;;
    --yes|-y)       ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

MODELS=${MODELS:-$HOME/models}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
IMAGE=${IMAGE:-radiance:latest}
DRAFT_REPO=${DRAFT_REPO:-tcclaviger/Qwen3.8-27B-DFlash2-FP8}
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-FP8}

case "$QUANT" in
  mxfp4) SRC_REPO=${SRC_REPO:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}
         SNAP=${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}
         NEED_BUILD=1 ; SIZE="~38 GiB (19 source + 19 built)" ;;
  int4)  SRC_REPO=${SRC_REPO:-z-lab/Qwen3.8-27B-PARO}
         SNAP=${SNAP:-$MODELS/Qwen3.8-27B-PARO}
         NEED_BUILD=0 ; SIZE="~19 GiB" ;;
  int5)  SRC_REPO=${SRC_REPO:-Launch80/Qwen3.8-27B-PARO-int5}
         SNAP=${SNAP:-$MODELS/Qwen3.8-27B-PARO-int5}
         NEED_BUILD=0 ; SIZE="~21 GiB" ;;
  *) echo "unknown QUANT=$QUANT (--mxfp4 / --int4 / --int5)" >&2; exit 2 ;;
esac

step() { echo; echo "=== $* ==="; }
ok()   { echo "  ok: $*"; }
die()  { echo "ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

# --------------------------------------------------------------- 1. host check
step "1/4  host"
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v docker >/dev/null 2>&1; then RUNTIME=docker
  elif command -v podman >/dev/null 2>&1; then RUNTIME=podman
  else die "no container runtime found" "install docker (preferred) or podman, then re-run"
  fi
fi
ok "container runtime: $RUNTIME"
[ -e /dev/kfd ] || die "/dev/kfd is missing -- amdgpu kernel driver not loaded"
[ -d /dev/dri ] || die "/dev/dri is missing -- no GPU render nodes"

free_gib=$(df -BG --output=avail "$(dirname "$MODELS")" 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$free_gib" ] && [ "$free_gib" -lt 40 ] && \
  echo "  WARNING: ${free_gib} GiB free on $(dirname "$MODELS"); setup wants ~$SIZE + drafter + image"

mkdir -p "$MODELS" "$HF_CACHE"

# --------------------------------------------------------------- 2. image pull
step "2/4  pull ${IMAGE}"
if "$RUNTIME" image exists "$IMAGE" >/dev/null 2>&1 || "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "$IMAGE already present locally"
else
  echo "  pulling $IMAGE (a few GiB)"
  "$RUNTIME" pull "$IMAGE"
fi

# --------------------------------------------------------------- 3. checkpoint
step "3/4  ${QUANT} checkpoint"
case "$SNAP" in
  "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
  *) die "SNAP ($SNAP) must live under MODELS ($MODELS)" ;;
esac

# Download via the image's own python — it ships huggingface_hub, so the
# host needs no Python environment for this.
hf_get() {
  local repo="$1" dest="$2"
  "$RUNTIME" run --rm --network=host \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$MODELS":/models \
    --entrypoint python3 "$IMAGE" -c '
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2]))
' "$repo" "$dest"
}

if [ "$NEED_BUILD" = 1 ]; then
  # MXFP4: download source + rewrite to a loadable form via the repo's
  # fp8_mtp.py (in the image at /opt/radiance_* — actually fp8_mtp.py lives
  # in the upstream repo, not this one, so we mount it from the build
  # context). For most users, --mxfp4 is a one-time write; use --no-build
  # if you already have a built checkpoint.
  if [ -f "$SNAP/config.json" ]; then
    ok "skipped -- $SNAP already built"
  else
    SRC=$(ls -d "$HF_CACHE"/hub/models--${SRC_REPO//\//--}/snapshots/*/ 2>/dev/null | head -1 || true)
    if [ -z "$SRC" ]; then
      echo "  downloading $SRC_REPO into HF cache"
      hf_get "$SRC_REPO" >/dev/null
      SRC=$(ls -d "$HF_CACHE"/hub/models--${SRC_REPO//\//--}/snapshots/*/ | head -1)
    fi
    ok "source: $SRC"
    echo "  rewriting the MTP head to fp8 (~15 minutes, one-time)."
    echo "  AMD's release lists the bf16 mtp.* layers as tensor names (mtp.fc.weight)"
    echo "  in its exclude list, so quark's module match never fires; vLLM then"
    echo "  asserts on a half-width parameter at load. This rewrite is not optional."
    CSRC="/root/.cache/huggingface/${SRC#"$HF_CACHE"/}"
    "$RUNTIME" run --rm \
      -v "$HF_CACHE":/root/.cache/huggingface \
      -v "$MODELS":/models \
      -v "$(pwd)":/repo:z \
      --entrypoint python3 "$IMAGE" /repo/fp8_mtp.py "$CSRC" "$CSNAP"
    [ -f "$SNAP/config.json" ] || die "fp8_mtp.py did not produce $SNAP/config.json"
    ok "built: $SNAP"
  fi
else
  if [ -f "$SNAP/config.json" ]; then
    ok "skipped -- already at $SNAP"
  else
    echo "  downloading $SRC_REPO straight into $SNAP"
    hf_get "$SRC_REPO" "$CSNAP" >/dev/null
    [ -f "$SNAP/config.json" ] || die "download did not produce $SNAP/config.json"
    ok "downloaded: $SNAP"
  fi
  # ParoQuant: validate the bits/group_size/krot contract before serving.
  case "$QUANT" in
    int4) WANT_BITS=4 ;;
    int5) WANT_BITS=5 ;;
  esac
  python3 - "$SNAP/config.json" "$WANT_BITS" <<'PY' || die "checkpoint is not servable by this stack" \
      "the radiance ParoQuant kernels are built for quant_method=paroquant, bits=$WANT_BITS, group_size=128, krot<=8"
import json, sys
q = (json.load(open(sys.argv[1])).get("quantization_config") or {})
m, b, g, k = q.get("quant_method"), q.get("bits"), q.get("group_size"), q.get("krot")
print(f"  quantization_config: quant_method={m} bits={b} group_size={g} krot={k}")
want_bits = int(sys.argv[2])
bad = [n for n, v, want in (("quant_method", m, "paroquant"), ("bits", b, want_bits),
                            ("group_size", g, 128)) if v != want]
if not isinstance(k, int) or not 1 <= k <= 8:
    bad.append("krot")
if bad:
    print("  unsupported: " + ", ".join(bad), file=sys.stderr); sys.exit(1)
PY
  ok "checkpoint declares a servable paroquant config"
fi

# --------------------------------------------------------------- 4. drafter
step "4/4  speculative drafter"
if [ "$WANT_DRAFTER" = 0 ]; then
  echo "  skipped (--no-drafter)"
else
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) die "DRAFTER ($DRAFTER) must live under MODELS ($MODELS)" ;;
  esac
  if [ -f "$DRAFTER/config.json" ]; then
    ok "skipped -- already at $DRAFTER"
  else
    echo "  downloading $DRAFT_REPO (~2 GiB)"
    hf_get "$DRAFT_REPO" "$CDRAFTER" >/dev/null
    [ -f "$DRAFTER/config.json" ] || die "drafter download did not produce $DRAFTER/config.json"
    ok "downloaded: $DRAFTER"
  fi
fi

cat <<EOF

=== setup complete ===

Image:        $IMAGE
Checkpoint:   $SNAP
Drafter:      ${DRAFTER:-none}

Start the server:

    ./serve.sh                          # MXFP4
    QUANT=int4 ./serve.sh               # ParoQuant int4
    QUANT=int5 ./serve.sh               # ParoQuant int5

It listens on http://localhost:8080/v1. The first start compiles inductor
and Triton kernels and takes several extra minutes; later starts reuse the
cache. See PAROQUANT.md / DOCKERHUB.md for every knob.
EOF