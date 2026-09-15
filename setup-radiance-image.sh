#!/usr/bin/env bash
# setup-radiance-image.sh — build the prod-bake image.
#
# Reads VERSION from this repo, builds the Dockerfile (multi-stage: libr4d
# builder + main with the 20-patch chain, both kernels, and sitecustomize
# registration), and tags:
#
#     radiance:vX.Y.Z          (version tag — primary)
#     radiance:latest          (latest pointer)
#     radiance-mxfp4:vX.Y.Z    (alias — same id)
#     radiance-mxfp4:latest    (alias — same id)
#     radiance-paro:vX.Y.Z     (alias — same id)
#     radiance-paro:latest     (alias — same id)
#
# The aliases exist so downstream tooling that already references
# `radiance-mxfp4:latest` / `radiance-paro:latest` (tamad's docker_config
# .image field, the old README examples, etc.) keeps working unchanged.
# The image itself is the same regardless of which alias you pull — the
# dispatch is purely by checkpoint quant_method, see Dockerfile + README.
#
# Optional --push uploads to a registry (default: no push, local only). The
# registry is taken from $REGISTRY (or `--registry=codeberg.org/ggz14`).
#
# Idempotent: re-running rebuilds (the cache-keyed layers skip when nothing
# changed) and re-tags. A Dockerfile change forces the affected stages.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

REGISTRY=${REGISTRY:-}
PUSH=0
BASE_DIGEST_OVERRIDE=
for a in "$@"; do
  case "$a" in
    --push)            PUSH=1 ;;
    --registry=*)      REGISTRY="${a#*=}" ;;
    --base-digest=*)   BASE_DIGEST_OVERRIDE="${a#*=}" ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

# ---- runtime + version --------------------------------------------------------
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v docker  >/dev/null 2>&1; then RUNTIME=docker
  elif command -v podman  >/dev/null 2>&1; then RUNTIME=podman
  else echo "ERROR: no container runtime (docker or podman) found" >&2; exit 1
  fi
fi

VERSION=$(cat VERSION 2>/dev/null | tr -d '[:space:]')
[ -n "$VERSION" ] || { echo "ERROR: VERSION file missing or empty" >&2; exit 1; }

# ---- base-image digest -------------------------------------------------------
# Pull the base so we have a local copy (and so the build doesn't hit the
# registry mid-build), then read the repo digest and pin it in the Dockerfile
# via --build-arg. This is the repeatability hook: same digest + same repo
# commit = same image, regardless of upstream tag repaints.
BASE_REPO=${BASE_REPO:-stilldeadcode/vllm-radiance}
BASE_TAG=${BASE_TAG:-0.9.3}
echo "=== pulling base ${BASE_REPO}:${BASE_TAG}"
"$RUNTIME" pull "${BASE_REPO}:${BASE_TAG}" >/dev/null

BASE_DIGEST=$("$RUNTIME" inspect --format '{{index .RepoDigests 0}}' "${BASE_REPO}:${BASE_TAG}" \
               | sed "s|^.*@||" | sed "s|^sha256:||")
if [ -z "$BASE_DIGEST" ] || ! [[ "$BASE_DIGEST" =~ ^[a-fA-F0-9]{64}$ ]]; then
  # No usable manifest digest (podman edge case). Hard-fail rather than
  # silently pinning a second or a fallback, so a bad pin is visible.
  echo "ERROR: could not read a 64-hex manifest digest for ${BASE_REPO}:${BASE_TAG}" >&2
  echo "  (docker after a pull always has one; if you are on podman, try RUNTIME=docker)" >&2
  exit 1
fi
echo "=== base digest: ${BASE_DIGEST}"
if [ -n "$BASE_DIGEST_OVERRIDE" ]; then
  echo "    (overridden via --base-digest)"
  BASE_DIGEST="$BASE_DIGEST_OVERRIDE"
fi

# ---- build -------------------------------------------------------------------
echo "=== building radiance:v${VERSION}"
"$RUNTIME" build \
  --build-arg "BASE_DIGEST=${BASE_DIGEST}" \
  --file Dockerfile.radiance \
  --tag "radiance:v${VERSION}" \
  --tag "radiance:latest" \
  --label "org.opencontainers.image.title=radiance-vllm-mxfp4" \
  --label "org.opencontainers.image.version=${VERSION}" \
  --label "org.opencontainers.image.source=https://codeberg.org/ggz14/radiance-vllm-mxfp4" \
  --label "org.opencontainers.image.licenses=MIT" \
  "$SCRIPT_DIR"

# Tag the aliases. Podman sometimes refuses a re-tag onto a different name
# when the image has multiple tags already; use --tag instead of docker tag.
NEW_ID=$("$RUNTIME" inspect --format '{{.Id}}' "radiance:v${VERSION}")
echo "=== image id: ${NEW_ID}"
for alias in radiance-mxfp4:v${VERSION} radiance-mxfp4:latest \
             radiance-paro:v${VERSION}  radiance-paro:latest; do
  "$RUNTIME" tag "radiance:v${VERSION}" "$alias"
done

# ---- optional push -----------------------------------------------------------
if [ "$PUSH" = 1 ]; then
  [ -n "$REGISTRY" ] || { echo "ERROR: --push needs --registry=host/path" >&2; exit 1; }
  echo "=== pushing to ${REGISTRY}"
  for repo in radiance radiance-mxfp4 radiance-paro; do
    "$RUNTIME" push "${REGISTRY}/${repo}:v${VERSION}"
    "$RUNTIME" push "${REGISTRY}/${repo}:latest"
  done
fi

cat <<EOF

=== built ===

Image:      ${NEW_ID}
Tags:       radiance:v${VERSION}  radiance:latest
            radiance-mxfp4:v${VERSION}  radiance-mxfp4:latest
            radiance-paro:v${VERSION}   radiance-paro:latest
Base:       ${BASE_REPO}:${BASE_TAG}@${BASE_DIGEST}

Smoke test (selects the kernel by checkpoint quant_method — no env switch):

    docker run --rm --device /dev/kfd --device /dev/dri \\
      --group-add 993 --group-add 44 \\
      --security-opt seccomp=unconfined --security-opt apparmor=unconfined \\
      -v /path/to/Qwen3.8-27B-MXFP4-mtpfp8:/models \\
      -p 8080:8080 radiance:v${VERSION} \\
      --model /models --port 8080

If you don't have a checkpoint yet, run setup.sh first.
EOF