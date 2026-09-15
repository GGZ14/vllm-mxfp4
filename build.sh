#!/usr/bin/env bash
# build.sh -- build the radiance ggz14 images.
#
# One build for both images, from one file (Dockerfile.ggz14):
#
#   ./build.sh               build the radiance image (base + the ggz14 bake layer)
#                           -> ggz14/vllm-radiance-mxfp4:$VERSION-$SHA
#                              e.g.  ggz14/vllm-radiance-mxfp4:0.13.0-3368c48
#   ./build.sh --base-only  build the platform base only (--target base)
#                           -> ggz14/vllm-radiance:$VERSION-$SHA
#   ./build.sh --push       also push (needs --registry=host/path or $REGISTRY)
#   ./build.sh --jobs=N     MAX_JOBS for the from-source compile stage
#
# Tags (every build):
#   $VERSION-$SHA    primary (VERSION from the VERSION file, SHA7 of the commit
#                    being built -- the full recipe is in the file, so the tag
#                    fully identifies the image)
#   latest          moved to the build, every time
#   v$VERSION       the version alias (0.13.0 rebase can be found without the SHA)
#
# A dirty tree warns; the tag does not change (the SHA is the recipe id, a
# dirty build is a dev-only thing and the tag lies either way).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v docker >/dev/null 2>&1; then RUNTIME=docker
  elif command -v podman >/dev/null 2>&1; then RUNTIME=podman
  else echo "ERROR: no container runtime (docker or podman) found" >&2; exit 1
  fi
fi

# docker >= 29 aliases `docker build` to the buildkit gateway frontend, which on
# 29.1.3 fails this file at the parse stage with a swallowed "exit code: 1".
# The classic builder parses and runs it fine; pin it. BUILDKIT=1 opts back in.
if [ "$RUNTIME" = docker ] && [ "${BUILDKIT:-0}" != 1 ]; then
  export DOCKER_BUILDKIT=0
fi

PUSH=0; JOBS=; WANT_BASE=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --base-only) WANT_BASE=1 ;;
    --jobs=*) JOBS="${a#*=}" ;;
    --registry=*) REGISTRY="${a#*=}" ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//' ; exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------- version + commit ----
VERSION=$(tr -d '[:space:]' <VERSION 2>/dev/null || true)
[ -n "$VERSION" ] || { echo "ERROR: no VERSION file" >&2; exit 1; }
SHA=$(git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
[ -z "$(git status --porcelain 2>/dev/null)" ] || \
  echo "NOTE: the working tree is dirty; the tag $VERSION-$SHA will refer to the committed recipe, not the worktree"

if [ "$WANT_BASE" = 1 ]; then
  NAME=${BASE_NAME:-ggz14/vllm-radiance}
  TARGET=(--target base)
else
  NAME=${NAME:-ggz14/vllm-radiance-mxfp4}
  TARGET=()
fi

BUILD_ARGS=(--file Dockerfile.ggz14
  -t "${NAME}:${VERSION}-${SHA}"
  -t "${NAME}:latest"
  -t "${NAME}:v${VERSION}")
[ -n "$JOBS" ] && BUILD_ARGS+=(--build-arg "MAX_JOBS=$JOBS")

# --------------------------------------------------------- the build -----------
echo "=== $RUNTIME build: $( [ "$WANT_BASE" = 1 ] && echo "--target base" || echo "full" ) -t $NAME:$VERSION-$SHA ==="
"$RUNTIME" build "${BUILD_ARGS[@]}" ${TARGET[@]+"${TARGET[@]}"} .

# --------------------------------------------------------- the base byproduct --
# The base is a build product of the same file: re-tag the stage-4 output so
# the next bake can ride on it without a fresh from-source compile (same for
# a standalone --base-only, just not re-run).
if [ "$WANT_BASE" != 1 ]; then
  "$RUNTIME" build --file Dockerfile.ggz14 --target base \
    -t "${BASE_NAME:-ggz14/vllm-radiance}:${VERSION}-${SHA}" \
    -t "${BASE_NAME:-ggz14/vllm-radiance}:latest" . >/dev/null 2>&1 || \
  echo "  note: the base is not separately available with $RUNTIME; the full build is the source of truth"
fi

# --------------------------------------------------------- optional push -------
if [ "$PUSH" = 1 ]; then
  [ -n "${REGISTRY:-}" ] || { echo "ERROR: --push needs --registry=host/path (or \$REGISTRY)" >&2; exit 1; }
  echo "=== pushing to ${REGISTRY}"
  for n in "$NAME" "${BASE_NAME:-ggz14/vllm-radiance}"; do
    for t in "${VERSION}-${SHA}" latest "v${VERSION}"; do
      if "$RUNTIME" inspect "$n:$t" >/dev/null 2>&1; then
        "$RUNTIME" tag "$n:$t" "${REGISTRY}/${n}:$t"
        "$RUNTIME" push "${REGISTRY}/${n}:$t"
      fi
    done
  done
fi

cat <<EOF

=== done ===
  primary  ${NAME}:${VERSION}-${SHA}
  latest   ${NAME}:latest
  version  ${NAME}:v${VERSION}

Compose consumes:

  services:
    vllm:
      image: ${NAME}:${VERSION}-${SHA}
EOF