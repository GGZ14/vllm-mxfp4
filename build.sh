#!/usr/bin/env bash
# build.sh -- the radiance build orchestrator. One file builds both images:
#
#   ./build.sh                          bake the release image (base + ggz14 layer)
#   ./build.sh --base-only              publish the platform base only
#                                       (docker build --target base)
#   ./build.sh --push                  also push (needs --registry=host/path, or $REGISTRY)
#   ./build.sh --jobs=N                MAX_JOBS for the from-source compile stage
#   ./build.sh --help                  this text
#
# Reads VERSION from this repo, builds Dockerfile.ggz14, and tags:
#
#   radiance:vX.Y.Z           version tag (primary)
#   radiance:latest           latest pointer
#   radiance-mxfp4:vX.Y.Z     alias, same id (downstream tooling that references
#   radiance-mxfp4:latest     the older name keeps working)
#   radiance-paro:vX.Y.Z
#   radiance-paro:latest
#
# The base stages (1-4) are cached between runs on the same machine: a patch or
# kernel change re-runs only the bake stage. On a cold machine the full build
# includes the from-source stack compile (hours) -- the same reality as every
# base build in this repo's history. Dev-loop against the PUBLISHED base only:
# Dockerfile.ggz14.top (a few minutes, no stack rebuild).
#
# NB: the base recipe in Dockerfile.ggz14 is a verbatim copy of this repo's
# Dockerfile. When the upstream file changes, re-sync that copy before
# publishing --target base output.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

REGISTRY=${REGISTRY:-}
PUSH=0
BASE_ONLY=0
JOBS=
for a in "$@"; do
  case "$a" in
    --base-only)        BASE_ONLY=1 ;;
    --push)             PUSH=1 ;;
    --registry=*)       REGISTRY="${a#*=}" ;;
    --jobs=*)           JOBS="${a#*=}" ;;
    -h|--help)
      sed -n '2,28p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------- runtime + version
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v docker >/dev/null 2>&1; then RUNTIME=docker
  elif command -v podman >/dev/null 2>&1; then RUNTIME=podman
  else echo "ERROR: no container runtime (docker or podman) found" >&2; exit 1
  fi
fi

VERSION=$(tr -d '[:space:]' <VERSION 2>/dev/null || true)
if [ -z "$VERSION" ]; then echo "ERROR: no VERSION file" >&2; exit 1; fi

IMAGE_ARGS=(--file Dockerfile.ggz14)
if [ -n "$JOBS" ]; then IMAGE_ARGS+=(--build-arg "MAX_JOBS=$JOBS"); fi

# --------------------------------------------------------------- build
if [ "$BASE_ONLY" = 1 ]; then
  TARGET_ARG=(--target base)
  TAGS=(-b -t base)
else
  TARGET_ARG=()
  TAGS=(-t "radiance:v${VERSION}" -t "radiance:latest")
fi

echo "=== ${RUNTIME} build${TARGET_ARG[*]+" (${TARGET_ARG[*]})"} ==="
"$RUNTIME" build "${IMAGE_ARGS[@]}" "${TARGET_ARG[@]+"${TARGET_ARG[@]}"}" "${TAGS[@]}"

if [ "$BASE_ONLY" = 1 ]; then
  cat <<EOF

=== base published ===
  base   (the platform base, --target base)
To build the release image:  ./build.sh
EOF
  exit 0
fi

NEW_ID=$("$RUNTIME" inspect --format '{{.Id}}' "radiance:v${VERSION}")
for alias in radiance-mxfp4:v${VERSION} radiance-mxfp4:latest \
             radiance-paro:v${VERSION}  radiance-paro:latest; do
  "$RUNTIME" tag "radiance:v${VERSION}" "$alias"
done

# --------------------------------------------------------------- optional push
if [ "$PUSH" = 1 ]; then
  [ -n "$REGISTRY" ] || { echo "ERROR: --push needs --registry=host/path (or \$REGISTRY)" >&2; exit 1; }
  echo "=== pushing to ${REGISTRY}"
  for repo in radiance radiance-mxfp4 radiance-paro; do
    "$RUNTIME" push "${REGISTRY}/${repo}:v${VERSION}"
    "$RUNTIME" push "${REGISTRY}/${repo}:latest"
  done
fi

cat <<EOF

=== baked ===
  image   ${NEW_ID}
  tags    radiance:v${VERSION}  radiance:latest
          radiance-mxfp4:v${VERSION}  radiance-mxfp4:latest
          radiance-paro:v${VERSION}   radiance-paro:latest

Smoke test (the checkpoint must be on the host; ./setup.sh fetches it):

    ./serve.sh
    # single-card host:  TP=1 ./serve.sh
EOF