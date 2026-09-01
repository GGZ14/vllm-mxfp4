#!/bin/bash
# calibrate-kv.sh -- measure a --kv-cache-memory pin for THIS host and write it to the local
# profile table, so serve-mxfp4.sh stops leaving KV cache unclaimed on hardware we have not
# shipped a row for.
#
#   ./calibrate-kv.sh            measure at the default batch shape and save the result
#   ./calibrate-kv.sh --dry-run  show the plan and the shape it would measure, run nothing
#   ./calibrate-kv.sh --quick    pass 1 only: pin what vLLM profiles, skip the search
#
# It needs the GPUs to itself and takes roughly 4 minutes per pass, so budget 15-20 minutes.
# The result lands in ~/.cache/radiance-mxfp4/kv-profiles.local.tsv and is picked up by every
# later ./serve-mxfp4.sh automatically -- there is nothing to copy or edit afterwards.
#
# WHY THIS IS A SEARCH AND NOT A FORMULA
# vLLM sizes the cache as requested_memory - non_kv_cache_memory - cudagraph_estimate, where
# non_kv_cache_memory is measured during a profile run and therefore carries that run's
# TRANSIENT activation peak. Steady-state serving never needs that peak at the same time as a
# full cache, so the profiled figure is an underestimate of what the card will actually hold.
# How much of an underestimate depends on the activation peak at CHUNK, on the cudagraph
# capture set that MAXSEQS produces, and on how the allocator fragments on that specific card.
# None of that is in the log. The only honest way to find the margin is to raise the pin until
# startup stops surviving, and back off -- which is what pass 2 does.
#
# WHAT COUNTS AS SURVIVING
# Reaching "GPU KV cache size" is NOT enough: the cache is allocated before cudagraph capture,
# and capture is where an over-committed pin actually dies. A pass counts as passing only if
# the server answers /health AND completes one CHUNK-sized prefill plus a short decode, which
# is the first point at which every allocation the steady state needs has been made at once.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/gpu-detect.sh"

die() { echo "[calibrate-kv] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker"
  fi
fi
say() { echo "[calibrate-kv] $*"; }

DRY_RUN_ONLY=0; QUICK=0
for a in "$@"; do case "$a" in
  --dry-run) DRY_RUN_ONLY=1 ;;
  --quick)   QUICK=1 ;;
  -h|--help) sed -n '2,28p' "$0" | sed 's/^# \?//'; exit 0 ;;
  *) die "unknown argument: $a" "usage: ./calibrate-kv.sh [--dry-run] [--quick]" ;;
esac; done

# The shape the pin will be valid for. These MUST match what serve-mxfp4.sh will later run with,
# because the lookup key includes them -- a pin measured at MAXSEQS=8 is not valid at 16.
PORT=${PORT:-8080}
MAXSEQS=${MAXSEQS:-8}
CHUNK=${CHUNK:-8192}
MAXLEN=${MAXLEN:-262144}
SPEC_METHOD=${SPEC_METHOD:-dflash}
GPU_UTIL=${GPU_UTIL:-0.98}
NAME=${NAME:-vllmkvcal}

# Search shape. STEP is a fraction of the profiled figure; 2% is ~0.33 GiB per step on a 32 GiB
# card, which is fine grain against a margin that measured 0.93 GiB on the reference box.
STEP=${STEP:-0.02}
MAX_STEPS=${MAX_STEPS:-6}
# Back off one full step from the last size that passed. The failure this protects against is
# not startup -- that was just tested -- but a long-context prefill months later hitting a
# fragmentation pattern the calibration run never produced.
BACKOFF_STEPS=${BACKOFF_STEPS:-1}

LOCAL_TABLE=${KV_TABLE_LOCAL:-${XDG_CACHE_HOME:-$HOME/.cache}/radiance-mxfp4/kv-profiles.local.tsv}

say "hardware:  $RAD_GPU_COUNT x $RAD_GPU_NAME ($RAD_GPU_MIB MiB), tp=$RAD_TP, sig=$RAD_GPU_SIG"
say "shape:     maxseqs=$MAXSEQS chunk=$CHUNK maxlen=$MAXLEN spec=$SPEC_METHOD util=$GPU_UTIL"
say "table:     $LOCAL_TABLE"
existing=$(rad_kv_lookup "$RAD_GPU_SIG" "$MAXSEQS" "$CHUNK" "$MAXLEN" "$SPEC_METHOD")
if [ -n "$existing" ]; then
  say "note:      a pin already resolves for this key ($existing bytes). The measurement below"
  say "           is written to the local table, which is read last, so it shadows that row"
  say "           rather than editing it -- delete the local row to fall back again."
fi

if [ "$DRY_RUN_ONLY" = 1 ]; then
  say "pass 1: serve with KV_MEM=0 (profiling), read 'Available KV cache memory' from the log"
  if [ "$QUICK" = 0 ]; then
    say "pass 2: retry at +$(awk -v s="$STEP" 'BEGIN{printf "%.0f", s*100}')% steps, up to $MAX_STEPS, keeping the largest that serves"
    say "        then back off $BACKOFF_STEPS step(s) for margin"
  fi
  say "would write to: $LOCAL_TABLE"
  exit 0
fi

[ "$RAD_GPU_COUNT" -gt 0 ] || die "no usable GPU detected" "run ./gpu-detect.sh to see what was found"
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  die "port $PORT is in use -- calibration needs the GPUs to itself" \
      "stop the running server first ($RUNTIME ps; $RUNTIME stop <name>)"
fi

cleanup() { "$RUNTIME" stop -t 20 "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Start a server at a given pin ("" = let vLLM profile) and report what happened.
# Echoes "PASS <available_bytes>" / "FAIL <reason>" on stdout; everything else goes to stderr so
# the caller can capture the verdict without also capturing the progress chatter.
attempt() {
  local pin=$1 log; log=$(mktemp)
  cleanup
  NAME="$NAME" PORT="$PORT" MAXSEQS="$MAXSEQS" CHUNK="$CHUNK" MAXLEN="$MAXLEN" \
    SPEC_METHOD="$SPEC_METHOD" GPU_UTIL="$GPU_UTIL" KV_MEM="${pin:-0}" \
    VLLM_LOGGING_LEVEL=INFO \
    "$SCRIPT_DIR/serve-mxfp4.sh" >"$log" 2>&1 &
  local serve_pid=$!

  local i ready=0
  for i in $(seq 1 180); do
    if [ "$(curl -s -o /dev/null -w %{http_code} -m3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; then
      ready=1; break
    fi
    if ! kill -0 "$serve_pid" 2>/dev/null && ! "$RUNTIME" inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
      break
    fi
    sleep 5
  done

  if [ "$ready" != 1 ]; then
    local why="did not become healthy"
    grep -qiE 'out of memory|HIP out of memory|hipErrorOutOfMemory' "$log" && why="out of memory"
    echo "FAIL $why"; echo "--- last 15 log lines ---" >&2; tail -15 "$log" >&2
    rm -f "$log"; return 0
  fi

  # Health alone does not prove the steady state fits. Force one CHUNK-sized prefill and a short
  # decode: this is the first moment the full cache, the cudagraph pool and a real activation
  # buffer are all live together.
  local prompt; prompt=$(python3 -c "print('the quick brown fox jumps over the lazy dog. ' * $((CHUNK / 10)))")
  local code
  code=$(curl -s -o /dev/null -w %{http_code} -m 300 -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'Qwen3.8','prompt':sys.argv[1],'max_tokens':32,'temperature':0}))" "$prompt")" 2>/dev/null || echo 000)

  if [ "$code" != 200 ]; then
    echo "FAIL prefill probe returned HTTP $code"
    echo "--- last 15 log lines ---" >&2; tail -15 "$log" >&2
    rm -f "$log"; return 0
  fi

  # What the engine actually gave the cache, which for a profiled run is the number pass 2
  # starts from.
  local avail
  avail=$("$RUNTIME" logs "$NAME" 2>&1 | sed -n 's/.*Available KV cache memory: \([0-9.]*\) GiB.*/\1/p' | tail -1)
  local toks
  toks=$("$RUNTIME" logs "$NAME" 2>&1 | sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' | tail -1)
  echo "PASS ${avail:-0} ${toks:-?}"
  rm -f "$log"
}

# ---------------------------------------------------------------- pass 1: profile
say "pass 1/2: profiling run (vLLM sizes the cache itself)"
res=$(attempt "")
set -- $res
[ "$1" = PASS ] || die "the profiling run itself failed: ${*:2}" \
    "this is not a calibration problem -- the configuration does not serve on this host at all" \
    "try: MAXLEN=32768 CHUNK=4096 ./serve-mxfp4.sh   and read the log"
base_gib=$2; base_toks=$3
base=$(awk -v g="$base_gib" 'BEGIN{printf "%d", g*1073741824}')
say "pass 1: profiled ${base_gib} GiB/GPU, ${base_toks} KV tokens"

best=$base; best_toks=$base_toks
if [ "$QUICK" = 1 ]; then
  say "--quick: keeping the profiled figure, skipping the search"
else
  # ---------------------------------------------------------------- pass 2: push
  say "pass 2/2: raising the pin until it stops serving"
  step=0
  while [ "$step" -lt "$MAX_STEPS" ]; do
    step=$((step + 1))
    try=$(awk -v b="$base" -v s="$STEP" -v n="$step" 'BEGIN{printf "%d", b*(1+s*n)}')
    pct=$(awk -v s="$STEP" -v n="$step" 'BEGIN{printf "%.0f", s*n*100}')
    say "  +${pct}%: $try bytes ($(awk -v b="$try" 'BEGIN{printf "%.2f", b/1073741824}') GiB/GPU)"
    r=$(attempt "$try"); set -- $r
    if [ "$1" = PASS ]; then
      say "  +${pct}%: served, $3 KV tokens"
      best=$try; best_toks=$3
    else
      say "  +${pct}%: ${*:2} -- stopping the search here"
      break
    fi
  done

  if [ "$best" != "$base" ] && [ "$BACKOFF_STEPS" -gt 0 ]; then
    backed=$(awk -v b="$best" -v s="$STEP" -v n="$BACKOFF_STEPS" 'BEGIN{printf "%d", b/(1+s*n)}')
    if [ "$backed" -gt "$base" ]; then
      say "backing off $BACKOFF_STEPS step(s) from the largest that served, for margin"
      best=$backed
    fi
  fi
fi

cleanup

gain=$(awk -v b="$best" -v a="$base" 'BEGIN{printf "%.1f", (b/a-1)*100}')
mkdir -p "$(dirname "$LOCAL_TABLE")"
if [ ! -s "$LOCAL_TABLE" ]; then
  cat > "$LOCAL_TABLE" <<'HDR'
# KV cache pins measured on THIS host by calibrate-kv.sh.
# Read after the repo's kv-profiles.tsv and wins over it -- your hardware outranks our table.
# Delete a row to go back to profiling for that key. Re-run calibrate-kv.sh after changing
# MAXSEQS or CHUNK: a pin is only valid for the batch shape it was measured at.
# sig	maxseqs	chunk	maxlen	spec	bytes	note
HDR
fi
# Drop any previous row for this key before appending, so the table does not grow a history that
# rad_kv_lookup would then resolve by "last one wins" rather than by "most recent measurement".
tmp=$(mktemp)
awk -F'\t' -v s="$RAD_GPU_SIG" -v q="$MAXSEQS" -v c="$CHUNK" -v l="$MAXLEN" -v m="$SPEC_METHOD" \
  '$1 ~ /^#/ || !($1==s && $2==q && $3==c && $4==l && $5==m)' "$LOCAL_TABLE" > "$tmp"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RAD_GPU_SIG" "$MAXSEQS" "$CHUNK" "$MAXLEN" "$SPEC_METHOD" \
  "$best" "measured $(date +%Y-%m-%d) by calibrate-kv.sh; ${best_toks} KV tokens, +${gain}% over profiled" >> "$tmp"
mv "$tmp" "$LOCAL_TABLE"

say "done. pin = $best bytes ($(awk -v b="$best" 'BEGIN{printf "%.2f", b/1073741824}') GiB/GPU), +${gain}% over profiling"
say "written to $LOCAL_TABLE -- ./serve-mxfp4.sh will use it automatically"
