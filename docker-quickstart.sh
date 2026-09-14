#!/bin/bash
# docker-quickstart.sh -- one command from a fresh clone to a working endpoint, on Docker.
#
#   ./docker-quickstart.sh            check the host, fetch everything, start the server, test it
#   ./docker-quickstart.sh status     is it up, and what is it doing
#   ./docker-quickstart.sh logs       follow the server log
#   ./docker-quickstart.sh test       send a chat request and print the reply
#   ./docker-quickstart.sh stop       stop the server
#   ./docker-quickstart.sh restart    start it again (nothing is downloaded twice)
#   ./docker-quickstart.sh clean      remove the container; checkpoints and caches stay
#
# It does nothing setup-mxfp4.sh and serve-mxfp4.sh do not already do. It runs them with
# RUNTIME=docker, checks the things that only bite Docker users (the daemon socket, docker-group
# membership, a small /var/lib/docker, root-owned files in a bind mount), and then waits for
# /health so a first start ends with a working curl instead of a wall of compile output.
# Every message it prints names the command that fixes it.
#
# podman users do not need this script -- ./setup-mxfp4.sh && ./serve-mxfp4.sh is the same thing --
# but it takes RUNTIME=podman if you want the same guided run.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------- knobs
# All of these are the launcher's own variables, so anything set here is also what serve-mxfp4.sh
# sees. Nothing in this file is a second source of truth for a default.
RUNTIME=${RUNTIME:-docker}
PORT=${PORT:-8080}
MODELS=${MODELS:-$HOME/models}
NAME=${NAME:-vllmmxfp4074}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
# A first start compiles Triton and inductor kernels before the engine comes up, which is several
# minutes of looking idle; later starts reuse that cache. This is a ceiling on how long we WATCH,
# not an expected duration -- nothing is killed when it expires, we just stop polling.
WAIT_SECS=${WAIT:-2400}

ASSUME_YES=0
FOREGROUND=0
SETUP_ARGS=()
ACTION=start

while [ "$#" -gt 0 ]; do
  case "$1" in
    start|status|logs|test|stop|restart|clean) ACTION=$1 ;;
    -y|--yes)      ASSUME_YES=1; SETUP_ARGS+=(--yes) ;;
    --no-drafter)  SETUP_ARGS+=(--no-drafter) ;;
    --port)        PORT=${2:?--port needs a number}; shift ;;
    --port=*)      PORT=${1#*=} ;;
    --wait)        WAIT_SECS=${2:?--wait needs seconds}; shift ;;
    --wait=*)      WAIT_SECS=${1#*=} ;;
    --foreground)  FOREGROUND=1 ;;
    -h|--help)
      cat <<'USAGE'
docker-quickstart.sh -- native MXFP4 Qwen3.8-27B on AMD RDNA4, with Docker, in one command

  ./docker-quickstart.sh           set up everything and start the server (this is the one)
  ./docker-quickstart.sh status    is it up, and what is it doing
  ./docker-quickstart.sh logs      follow the server log (Ctrl-C stops watching, not the server)
  ./docker-quickstart.sh test      send a chat request and print the reply
  ./docker-quickstart.sh stop      stop the server
  ./docker-quickstart.sh restart   start it again -- nothing is downloaded twice
  ./docker-quickstart.sh clean     remove the container; checkpoints and caches stay

Options:
  -y, --yes         do not ask before the ~40 GiB download
  --no-drafter      skip the 2 GiB speculative drafter (serves with the in-model MTP head)
  --port 8080       listen port
  --wait 2400       seconds to wait for the first start before giving up watching
  --foreground      run the server in this terminal instead of in the background

Environment (the launcher's own variables -- see ./serve-mxfp4.sh --help for the rest):
  MODELS=~/models   where the checkpoints go            RUNTIME=docker  docker or podman
  IMAGE=...:0.9.3   container image                     NAME=...        container name

Disk: about 60 GiB for a full setup (19 source + 19 built checkpoint + 2 drafter + ~10 image).
The 19 GiB source download can be deleted afterwards; setup prints the command.
USAGE
      exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

case "$PORT"      in ''|*[!0-9]*) echo "--port/PORT must be a number, got: $PORT" >&2; exit 2 ;; esac
case "$WAIT_SECS" in ''|*[!0-9]*) echo "--wait/WAIT must be a number of seconds, got: $WAIT_SECS" >&2; exit 2 ;; esac

# ---------------------------------------------------------------- output
# Colour only on a terminal, and never when NO_COLOR is set: this script's output is the thing
# people paste into an issue, and escape codes in a paste help nobody.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; RST=$'\033[0m'
else
  B=""; DIM=""; GRN=""; YEL=""; RED=""; RST=""
fi
say()  { echo "$*"; }
step() { echo; echo "${B}$*${RST}"; }
ok()   { echo "  ${GRN}ok${RST}  $*"; }
warn() { echo "  ${YEL}note${RST}  $*"; }
hint() { echo "        ${DIM}$*${RST}"; }
die()  { echo; echo "  ${RED}stopped${RST}  $1" >&2; shift; for l in "$@"; do echo "          $l" >&2; done; echo >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# An array so the runtime is one word everywhere; RUNTIME=podman is the only other value that
# the scripts this one calls will also accept.
# shellcheck disable=SC2206
RT=(${RUNTIME})
RT_NAME=${RT[0]}

# ---------------------------------------------------------------- tiny HTTP client
# The host is not assumed to have curl: a minimal server install often does not, and the one
# place we are guaranteed a curl is inside the image itself (its own healthcheck uses one).
# Order: host curl, host python3, then curl inside the running container.
http_get() { # url -> body on stdout, non-zero if not 2xx
  local url=$1
  if have curl; then curl -fsS --max-time 10 "$url" 2>/dev/null
  elif have python3; then python3 - "$url" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=10).read().decode())
except Exception:
    sys.exit(1)
PY
  else "${RT[@]}" exec "$NAME" curl -fsS --max-time 10 "${url/localhost/127.0.0.1}" 2>/dev/null
  fi
}
http_post_json() { # url json -> body on stdout
  local url=$1 body=$2
  if have curl; then
    curl -fsS --max-time 180 -H 'Content-Type: application/json' -d "$body" "$url" 2>/dev/null
  elif have python3; then python3 - "$url" "$body" <<'PY' 2>/dev/null
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], data=sys.argv[2].encode(),
                             headers={"Content-Type": "application/json"})
try:
    sys.stdout.write(urllib.request.urlopen(req, timeout=180).read().decode())
except Exception:
    sys.exit(1)
PY
  else
    "${RT[@]}" exec "$NAME" curl -fsS --max-time 180 -H 'Content-Type: application/json' \
      -d "$body" "${url/localhost/127.0.0.1}" 2>/dev/null
  fi
}
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
health_ok()  { http_get "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

# ---------------------------------------------------------------- container state
container_exists()  { "${RT[@]}" inspect "$NAME" >/dev/null 2>&1; }
container_running() { [ "$("${RT[@]}" inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }

# What the server is busy with, in words, read off its own log. A first start spends most of its
# time in phases that print nothing for minutes, which is exactly when it looks hung.
log_phase() {
  local l
  l=$("${RT[@]}" logs --tail 400 "$NAME" 2>&1 | tail -400) || { echo "starting up"; return 0; }
  case "$l" in
    *"Application startup complete"*|*"Starting vLLM API server"*) echo "API server starting" ;;
    *"Capturing CUDA graph"*|*"Capturing cudagraphs"*|*"raph capturing finished"*) \
                                                                   echo "capturing CUDA graphs (nearly there)" ;;
    *"GPU KV cache size"*)                                         echo "KV cache allocated" ;;
    *"Compiling a graph"*|*"torch.compile takes"*|*"Dynamo bytecode transform"*) \
                                                                   echo "compiling kernels (first start only, this is the slow part)" ;;
    *"Loading safetensors"*|*"Loading weights"*)                   echo "loading weights" ;;
    *"libr4d"*|*"hipcc"*)                                          echo "building the R4D kernels" ;;
    *)                                                             echo "starting up" ;;
  esac
}

# One real request, printed. A server that answers /health but cannot generate is a failure
# mode worth catching here rather than in whatever the user tries next.
smoke_test() {
  local reply text
  reply=$(http_post_json "http://127.0.0.1:$PORT/v1/chat/completions" \
    '{"model":"Qwen3.8","max_tokens":256,"messages":[{"role":"user","content":"Say hello in one short sentence."}]}' || true)
  if [ -z "$reply" ]; then
    warn "the request did not come back"
    hint "if the engine only just came up, give it a moment and run: ./docker-quickstart.sh test"
    return 1
  fi
  text=""
  if have python3; then
    # The reply carries both content and reasoning; with thinking on, a short answer can land
    # entirely in the reasoning field, and printing nothing would look like a failure.
    text=$(printf '%s' "$reply" | python3 -c '
import json, sys
try:
    m = json.load(sys.stdin)["choices"][0]["message"]
    print((m.get("content") or m.get("reasoning") or "").strip()[:400])
except Exception:
    pass' 2>/dev/null)
  fi
  [ -n "$text" ] || text=$(printf '%s' "$reply" | head -c 400)
  ok "the model answered:"
  echo
  echo "    $text"
}

show_tail() { # print the end of the log, the thing you actually want after a failure
  echo
  echo "${DIM}--- last 40 lines of $RT_NAME logs $NAME ---${RST}"
  "${RT[@]}" logs --tail 40 "$NAME" 2>&1 || true
  echo "${DIM}--- end ---${RST}"
}

# ---------------------------------------------------------------- day-2 actions
case "$ACTION" in
  logs)
    container_exists || die "no container named $NAME" "start it with: ./docker-quickstart.sh"
    say "  following $RT_NAME logs -f $NAME  -- Ctrl-C stops watching, not the server"
    exec "${RT[@]}" logs -f "$NAME" ;;
  stop)
    if container_running; then
      "${RT[@]}" stop "$NAME" >/dev/null && ok "stopped $NAME"
    else
      ok "not running"
    fi
    exit 0 ;;
  clean)
    "${RT[@]}" rm -f "$NAME" >/dev/null 2>&1 && ok "removed the container $NAME" || ok "no container to remove"
    say "  checkpoints in $MODELS and the compile cache in ~/.radiance-cache-* were left alone"
    say "  start again any time with: ./docker-quickstart.sh"
    exit 0 ;;
  test)
    container_running || die "the server is not running" "start it with: ./docker-quickstart.sh"
    health_ok || die "the server is up but /health does not answer yet -- $(log_phase)" \
        "watch it with: ./docker-quickstart.sh logs"
    smoke_test
    exit 0 ;;
  status)
    if container_running; then
      if health_ok; then
        ok "serving on http://localhost:$PORT/v1  (container $NAME)"
        # data[].id only -- each entry also carries a permission object with an "id" of its own,
        # so a grep for "id" lists twice as many names as the server actually serves.
        m=$(http_get "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)
        if [ -n "$m" ] && have python3; then
          m=$(printf '%s' "$m" | python3 -c 'import json,sys
try: print(" ".join(d["id"] for d in json.load(sys.stdin)["data"]))
except Exception: pass' 2>/dev/null)
          [ -n "$m" ] && say "  model names: $m"
        fi
        say "  test it:  ./docker-quickstart.sh test"
      else
        warn "container $NAME is up but /health does not answer yet -- $(log_phase)"
        say "  watch it:  ./docker-quickstart.sh logs"
      fi
    elif container_exists; then
      warn "container $NAME exists but is not running"
      show_tail
      say "  start it again with: ./docker-quickstart.sh restart"
    else
      warn "no container named $NAME -- nothing is running"
      say "  start it with: ./docker-quickstart.sh"
    fi
    exit 0 ;;
esac

# ---------------------------------------------------------------- 1. host checks
say "${B}vllm-radiance -- MXFP4 Qwen3.8-27B on AMD RDNA4, via $RT_NAME${RST}"

step "1/5  host"

if ! have "$RT_NAME"; then
  if [ "$RT_NAME" != podman ] && have podman; then
    die "$RT_NAME is not installed, but podman is" \
        "podman needs no setup script -- run the two commands directly:" \
        "  ./setup-mxfp4.sh && ./serve-mxfp4.sh" \
        "or run this guided script on podman: RUNTIME=podman ./docker-quickstart.sh"
  fi
  die "$RT_NAME is not installed" \
      "install Docker Engine: https://docs.docker.com/engine/install/" \
      "  Debian/Ubuntu also ship it as the 'docker.io' package" \
      "  Fedora/RHEL: dnf install moby-engine   (or podman, which this repo also supports)"
fi

# `docker info` is the only check that the daemon is actually usable by THIS user. The two ways
# it fails have completely different fixes, and the raw error says so in a way people miss.
if ! info=$("${RT[@]}" info 2>&1); then
  case "$info" in
    *"permission denied"*|*"Got permission denied"*|*"connect: permission denied"*)
      if id -nG | tr ' ' '\n' | grep -qx docker; then
        die "your user is in the 'docker' group but this shell predates that" \
            "pick up the new group without logging out:" \
            "  newgrp docker      # then re-run ./docker-quickstart.sh"
      fi
      die "cannot talk to the Docker daemon: permission denied" \
          "add yourself to the docker group (one time):" \
          "  sudo usermod -aG docker $USER" \
          "  newgrp docker        # then re-run ./docker-quickstart.sh" \
          "this repo also runs on podman, which needs no daemon and no group: " \
          "  ./setup-mxfp4.sh && ./serve-mxfp4.sh" ;;
    *"Cannot connect"*|*"Is the docker daemon running"*|*"docker daemon is not running"*)
      die "the Docker daemon is not running" \
          "  sudo systemctl start docker" \
          "  sudo systemctl enable docker    # and at every boot" ;;
    *) die "$RT_NAME is installed but not usable" "$(echo "$info" | head -3)" ;;
  esac
fi
ok "$RT_NAME is installed and usable"

[ -e /dev/kfd ] || die "/dev/kfd is missing -- the amdgpu kernel driver is not loaded on this host" \
    "the image ships ROCm userspace, but the kernel driver has to be on the host" \
    "check: ls -l /dev/kfd /dev/dri   and   dmesg | grep amdgpu"
[ -d /dev/dri ] || die "/dev/dri is missing -- no GPU render nodes on this host"
ok "/dev/kfd and /dev/dri are present"

# The same scan the launcher uses, so this script and the serve cannot disagree about the
# hardware. It counts only cards big enough to hold a shard, which is what keeps an iGPU out.
# shellcheck source=gpu-detect.sh
. "$SCRIPT_DIR/gpu-detect.sh"
if [ "$RAD_GPU_COUNT" -lt 1 ]; then
  die "no AMD GPU with at least ${RAD_MIN_GPU_MIB} MiB of VRAM" \
      "found:${RAD_GPU_SKIPPED:- nothing on the amdgpu driver}" \
      "this image is compiled for gfx1201 (RDNA4) only -- an R9700 or another RDNA4 card" \
      "to admit a smaller card anyway: MIN_GPU_MIB=4096 ./docker-quickstart.sh"
fi
ok "$RAD_GPU_COUNT x $RAD_GPU_NAME ($RAD_GPU_MIB MiB each) -> tensor-parallel $RAD_TP"
[ -n "$RAD_GPU_SKIPPED" ] && hint "skipped as too small (an iGPU, usually):$RAD_GPU_SKIPPED"

# Docker runs containers as root, so everything the container writes into a bind mount lands
# root-owned on the host. Worth saying once, up front, rather than being discovered at `rm`.
if [ "$RT_NAME" != podman ]; then
  hint "Docker writes as root: files under $MODELS and ~/.radiance-cache-* will be root-owned."
  hint "To delete them later:  sudo rm -rf <path>   (podman rootless does not have this)"
fi

# Disk. Two filesystems matter and they are often not the same one: the checkpoints go to
# $MODELS, and the ~10 GiB image goes wherever the daemon keeps its data.
avail_gib() { # nearest existing parent of $1
  local p=$1; while [ ! -d "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done
  df -PBG "$p" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4+0}'
}
models_free=$(avail_gib "$MODELS")
if [ -n "$models_free" ] && [ "$models_free" -lt 45 ]; then
  warn "${models_free} GiB free where the checkpoints go ($MODELS); a full setup wants ~45 GiB there"
  hint "put them somewhere larger: MODELS=/data/models ./docker-quickstart.sh"
else
  ok "${models_free:-?} GiB free for checkpoints ($MODELS)"
fi
droot=$("${RT[@]}" info -f '{{.DockerRootDir}}' 2>/dev/null || true)
if [ -n "$droot" ]; then
  image_free=$(avail_gib "$droot")
  if [ -n "$image_free" ] && [ "$image_free" -lt 12 ]; then
    warn "${image_free} GiB free on the Docker data directory ($droot); the image needs ~10 GiB"
    hint "reclaim some with: $RT_NAME system prune -a"
  else
    ok "${image_free:-?} GiB free for the image ($droot)"
  fi
fi

# A busy port is almost always a server that is already holding the GPUs -- and this one needs
# every GPU it serves on, so it is worth catching here rather than 40 GiB later.
if [ "$ACTION" != restart ] && port_open "$PORT" && ! container_running; then
  die "port $PORT is already in use by something else" \
      "running containers: $("${RT[@]}" ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')" \
      "stop it with: $RT_NAME stop <name>" \
      "or serve on another port: ./docker-quickstart.sh --port 8081"
fi

if [ "$ACTION" = restart ] && container_running; then
  "${RT[@]}" stop "$NAME" >/dev/null 2>&1 || true
  ok "stopped the running container before restarting"
fi

# ---------------------------------------------------------------- 2. download and build
step "2/5  image, checkpoints and kernels"
say "  setup-mxfp4.sh does this part. It is idempotent: everything already done is skipped,"
say "  and it is safe to interrupt and re-run -- downloads resume."
if [ "$ASSUME_YES" = 0 ]; then
  hint "first run downloads ~40 GiB; it will ask before starting"
fi
echo
RUNTIME="$RUNTIME" MODELS="$MODELS" IMAGE="$IMAGE" \
  "$SCRIPT_DIR/setup-mxfp4.sh" ${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}

# ---------------------------------------------------------------- 3. start
step "3/5  start the server"
if [ "$FOREGROUND" = 1 ]; then
  say "  running in this terminal (Ctrl-C stops the server)"
  echo
  exec env RUNTIME="$RUNTIME" MODELS="$MODELS" IMAGE="$IMAGE" PORT="$PORT" NAME="$NAME" \
    "$SCRIPT_DIR/serve-mxfp4.sh"
fi
# The launcher's own [run] lines say which kernels, KV pin and cache directory this serve got,
# which is the first thing anyone needs when something looks wrong -- worth keeping. The only
# thing dropped is the bare container id `-d` prints last.
RUNTIME="$RUNTIME" MODELS="$MODELS" IMAGE="$IMAGE" PORT="$PORT" NAME="$NAME" DETACH=1 \
  "$SCRIPT_DIR/serve-mxfp4.sh" | sed -e '/^[0-9a-f]\{12,\}$/d' -e 's/^/  /'
ok "container $NAME started in the background"

# ---------------------------------------------------------------- 4. wait for it
step "4/5  wait for the engine"
say "  A first start compiles Triton and inductor kernels before the engine comes up. That is"
say "  several minutes of looking idle, and it is cached: later starts skip it. Nothing here"
say "  kills the server -- Ctrl-C only stops watching."
echo
started=$SECONDS
phase=""; beat=0; tick=0
while [ $((SECONDS - started)) -lt "$WAIT_SECS" ]; do
  if ! container_running; then
    show_tail
    die "the container exited while starting" \
        "the lines above say why; the common ones are in README.md#troubleshooting" \
        "re-run after a fix with: ./docker-quickstart.sh restart"
  fi
  if health_ok; then break; fi
  # /health every 5 s, but the log only every 30 s: reading it is a container round trip, and
  # the phases it reports change on the scale of minutes.
  if [ $((tick % 6)) = 0 ]; then
    p=$(log_phase)
    el=$(( (SECONDS - started) / 60 ))
    if [ "$p" != "$phase" ]; then
      phase=$p; beat=$SECONDS
      printf '  [%3d min] %s\n' "$el" "$p"
    elif [ $((SECONDS - beat)) -ge 120 ]; then
      beat=$SECONDS
      printf '  [%3d min] still %s\n' "$el" "$p"
    fi
  fi
  tick=$((tick + 1))
  sleep 5
done

if ! health_ok; then
  warn "gave up watching after $((WAIT_SECS / 60)) minutes -- the server is still running and may still come up"
  say "  watch it:  ./docker-quickstart.sh logs"
  say "  check it:  ./docker-quickstart.sh status"
  exit 1
fi
ok "engine up after $(( (SECONDS - started) / 60 ))m $(( (SECONDS - started) % 60 ))s"

# ---------------------------------------------------------------- 5. prove it works
step "5/5  a real request"
smoke_test || true

cat <<EOF

${B}=== serving on http://localhost:$PORT/v1 ===${RST}

  An OpenAI-compatible endpoint. The model name is ${B}Qwen3.8${RST}.

  curl http://localhost:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'

  ./docker-quickstart.sh status     is it up, and what is it doing
  ./docker-quickstart.sh logs       follow the log
  ./docker-quickstart.sh test       send another request
  ./docker-quickstart.sh stop       stop the server
  ./docker-quickstart.sh restart    start it again (fast: the compile cache is warm)

  Every knob:  ./serve-mxfp4.sh --help        What was detected:  ./gpu-detect.sh
EOF
