#!/usr/bin/env bash
# build.sh -- build the radiance ggz14 images.
#
# Default (the normal one):
#
#   ./build.sh         quick boot image = the PUBLISHED deadcode base
#                     (stilldeadcode/vllm-radiance:0.9.3) + the ggz14 bake layer
#                     (Dockerfile.ggz14.top: ~a few minutes -- the patch chain,
#                     2 hipcc kernel compiles, dispatch registration)
#                     -> ggz14/vllm-radiance-mxfp4:$VERSION-$SHA
#                        e.g.  ggz14/vllm-radiance-mxfp4:0.13.0-3f8e21a
#
#   ./build.sh --paro        the ParoQuant prod-bake image: the SAME bake on the
#                           same published base, but the ParoQuant measured profile
#                           baked (paroquant/Dockerfile: run_paroquant.sh's "-e" list)
#                           -> ggz14/vllm-radiance-paroquant:$VERSION-$SHA
#
# Building the base from source (only when the upstream base moves, e.g. a
# new vLLM/torch - that is the base recipe's purpose, not a bake input):
#
#   ./build.sh --base-only   the platform base, from source (--target base)
#                           -> ggz14/vllm-radiance:$VERSION-$SHA
#   ./build.sh --full       base from source AND the bake layer (hours)
#
# Other:
#   ./build.sh --push       also push (needs --registry=host/path or $REGISTRY)
#   ./build.sh --jobs=N     MAX_JOBS for the from-source compile stage
#   ./build.sh --base-repo=stilldeadcode/vllm-radiance --base-tag=0.9.3
#   ./build.sh --base-digest=HEX64   pin explicitly instead of the pulled one
#
# Tags (every build):
#   $VERSION-$SHA    primary (VERSION from the VERSION file, SHA7 of the commit
#                    being built -- the full recipe is in this repo, so the tag
#                    identifies the image)
#   latest          moved to the build, every time
#   v$VERSION       the version alias (a 0.13.0 rebuild can be found by eye)
#
# A dirty tree warns; the tag does not change (the SHA is the recipe id).

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

# docker >= 29 aliases `docker build` to the buildkit gateway front-end, which on 29.1.3
# fails this file at parse time with a swallowed "exit code: 1". The classic builder
# parses and runs fine. Pin it; BUILDKIT=1 opts back in.
if [ "$RUNTIME" = docker ] && [ "${BUILDKIT:-0}" != 1 ]; then
  export DOCKER_BUILDKIT=0
fi

WANT_BASE=0; WANT_FULL=0; WANT_PARO=0; PUSH=0; JOBS=
BASE_REPO=${BASE_REPO:-stilldeadcode/vllm-radiance}
BASE_TAG=${BASE_TAG:-0.9.3}
BASE_DIGEST=
BASE_PIN_OVERRIDE=1
for a in "$@"; do
  case "$a" in
    --base-only) WANT_BASE=1 ;;
    --full) WANT_FULL=1 ;;
    --paro) WANT_PARO=1 ;;
    --push) PUSH=1 ;;
    --jobs=*) JOBS="${a#*=}" ;;
    --registry=*) REGISTRY="${a#*=}" ;;
    --base-repo=*) BASE_REPO="${a#*=}" ;;
    --base-tag=*) BASE_TAG="${a#*=}" ;;
    --base-digest=*) BASE_DIGEST="${a#*=}" ;;
    -h|--help) sed -n '2,34p' "$0" | sed 's/^# \?//' ; exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------- version + commit ----
VERSION=$(tr -d '[:space:]' < VERSION 2>/dev/null || true)
[ -n "$VERSION" ] || { echo "ERROR: no VERSION file" >&2; exit 1; }
SHA=$(git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
[ -z "$(git status --porcelain 2>/dev/null)" ] || \
  echo "NOTE: the working tree is dirty; the tag $VERSION-$SHA will refer to the committed recipe, not the worktree"

# --------------------------------------------------------- which recipe --------
# default = quick boot on the published base. base from source only when asked.
if [ "$WANT_BASE" = 1 ]; then
  NAME=${BASE_NAME:-ggz14/vllm-radiance}
  DF=Dockerfile.ggz14
  TARGET=(--target base)
elif [ "$WANT_FULL" = 1 ]; then
  NAME=${NAME:-ggz14/vllm-radiance-mxfp4}
  DF=Dockerfile.ggz14
  TARGET=()
elif [ "$WANT_PARO" = 1 ]; then
  NAME=${PARO_NAME:-ggz14/vllm-radiance-paroquant}
  DF=paroquant/Dockerfile
  TARGET=()
else
  NAME=${NAME:-ggz14/vllm-radiance-mxfp4}
  DF=Dockerfile.ggz14.top
  TARGET=()
fi

# --------------------------------------------------------- base pin (top/full) -
# The top rides the published base: pull it, pin by live digest, warn on drift
# from the recipe's default. (For --full the base is source by definition; the
# pin is a reference only.)
BUILDARG=()
if [ "$WANT_BASE" != 1 ] && [ -z "$BASE_DIGEST" ]; then
  echo "=== pulling base ${BASE_REPO}:${BASE_TAG}"
  "$RUNTIME" pull "${BASE_REPO}:${BASE_TAG}" >/dev/null
  BASE_DIGEST=$("$RUNTIME" inspect --format '{{index .RepoDigests 0}}' "${BASE_REPO}:${BASE_TAG}" 2>/dev/null | sed 's|^.*@||' | sed 's|^sha256:||')
  if ! [[ "$BASE_DIGEST" =~ ^[a-f0-9]{64}$ ]]; then
    echo "NOTE: no manifest digest readable for ${BASE_REPO}:${BASE_TAG}; using the recipe's default pin" >&2
  else
    DEFPIN=$(grep -m1 -oP 'ARG BASE_DIGEST=\K[0-9a-f]{64}' $DF)
    [ -n "$DEFPIN" ] && [ "$DEFPIN" != "$BASE_DIGEST" ] && \
      echo "NOTE: the published base has moved since the recipe was written (default pin ${DEFPIN} != live ${BASE_DIGEST}); building against the live base"
  fi
fi
[ -n "$BASE_DIGEST" ] && BUILDARG=(--build-arg "BASE_DIGEST=${BASE_DIGEST}")

BUILD_ARGS=(--file $DF
  -t "${NAME}:${VERSION}-${SHA}"
  -t "${NAME}:latest"
  -t "${NAME}:v${VERSION}"
  "${BUILDARG[@]}")
[ -n "$JOBS" ] && BUILD_ARGS+=(--build-arg "MAX_JOBS=$JOBS")

# --------------------------------------------------------- the build -----------
echo "=== $RUNTIME build: [ $DF ${TARGET[*]:-no --target} ] -t $NAME:$VERSION-$SHA ==="
"$RUNTIME" build "${BUILD_ARGS[@]}" ${TARGET[@]+"${TARGET[@]}"} .

# --------------------------------------------------------- optional push -------
if [ "$PUSH" = 1 ]; then
  [ -n "${REGISTRY:-}" ] || { echo "ERROR: --push needs --registry=host/path (or \$REGISTRY)" >&2; exit 1; }
  echo "=== pushing to ${REGISTRY}"
  for t in "${VERSION}-${SHA}" latest "v${VERSION}"; do
    if "$RUNTIME" inspect "$NAME:$t" >/dev/null 2>&1; then
      "$RUNTIME" tag "$NAME:$t" "${REGISTRY}/${NAME}:$t"
      "$RUNTIME" push "${REGISTRY}/${NAME}:$t"
    fi
  done
fi

base_digest="$( [ -n "${BASE_DIGEST}" ] && echo "${BASE_DIGEST}" || echo "recipe default" )"
cat <<EOF

=== done ===
  primary  ${NAME}:${VERSION}-${SHA}
  latest   ${NAME}:latest
  version  ${NAME}:v${VERSION}
$( [ "$WANT_BASE" = 0 ] && [ "$WANT_FULL" = 0 ] && echo "  base     ${BASE_REPO}:${BASE_TAG} (published, pinned at $base_digest)" )

Compose consumes:

  services:
    vllm:
      image: ${NAME}:${VERSION}-${SHA}
EOF