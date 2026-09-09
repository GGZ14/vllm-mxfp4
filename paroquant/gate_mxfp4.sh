#!/bin/bash
# Accuracy gate for a re-quantized ParoQuant checkpoint, before any kernel work is done.
#
# convert --mode pseudo writes fp16 weights that have already been through
# rotate -> quantize -> inverse-rotate, so they carry the EXACT error of the scheme while loading
# on the ordinary path. Serving those and scoring GSM8K measures the quality of the format
# without needing the MXFP4+rotation serving kernel to exist yet. If the number is in band, the
# kernel work is justified; if it is not, we stop having spent zero kernel effort.
#
# Everything except the weights is held fixed against the shipped PARO serve: same patch prelude,
# same pinned libr4d, same chat template (the 97-98 band is template-bound -- the repo template
# scores 95-96 and would make this comparison lie).
set -euo pipefail

MODELS=${MODELS:-$HOME/models}
PSEUDO=${PSEUDO:-Qwen3.8-27B-PARO-MXFP4-pseudo}
# REAL=1: the directory is a real paroquant_mxfp4 checkpoint, served through the W4A8 kernel path
# (the deployed accuracy, not the W4A16 upper bound the pseudo model gives). The v1 loader takes
# with the rotation streams at the launcher defaults (STREAMS=0 forces them off to gate the plain
# prologue instead), and CHECKALL gates every partition.
REAL=${REAL:-0}
if [ "$REAL" = 1 ]; then
  [ "${STREAMS:-1}" = 0 ] && export RADIANCE_PQ_ROT_STREAM=0 RADIANCE_PQ_ROT_STREAM2=0 RADIANCE_PQ_ROT_STREAM3=0
  export RADIANCE_PQM_CHECKALL=${RADIANCE_PQM_CHECKALL:-"7168:5120,5120:3072,17408:5120,5120:8704,8192:5120"}
fi
N=${N:-500}
TAG=${TAG:-mxfp4-paro-pseudo}
MAXLEN=${MAXLEN:-4096}          # GSM8K is short; the fp16 pseudo model is 55.6 GiB, so leave
GPU_UTIL=${GPU_UTIL:-0.95}      # as little as possible to KV
PORT=${PORT:-8080}
LOG=${LOG:-$HOME/gate_${TAG}.log}
REPO=$(cd "$(dirname "$0")/.." && pwd)

[ -d "$MODELS/$PSEUDO" ] || { echo "pseudo model missing at $MODELS/$PSEUDO" >&2; exit 1; }

echo "=== serving $PSEUDO (fp16 pseudo-quantized) ==="
MODE=eval MODEL_DIR="$PSEUDO" MAXLEN_EVAL="$MAXLEN" GPU_UTIL="$GPU_UTIL" \
  SERVED_NAMES=gate NAME=vllmgate PORT="$PORT" CHECKALL="${RADIANCE_PQM_CHECKALL:-}" \
  "$REPO/paroquant/run_paroquant.sh" > "$LOG" 2>&1 &

echo "waiting for the server (log: $LOG)"
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || {
  echo "server did not come up; last lines:" >&2; tail -30 "$LOG" >&2; exit 1; }
echo "server up"

echo "=== GSM8K ${N}q (greedy) ==="
python3 "$HOME/pibench-local/gsm8k.py" --model gate --tag "$TAG" --n "$N" \
  --url "http://localhost:$PORT" --concurrency 8
rc=$?

echo "=== stopping ==="
podman stop -t 30 vllmgate >/dev/null 2>&1 || true
exit $rc
