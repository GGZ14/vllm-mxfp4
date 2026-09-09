#!/bin/bash
# Kernel-path speed A/B on ONE card, eager: int4 PARO vs MXFP4-PARO, same launcher, same mode.
# Prefill at long M is launch-negligible in eager mode, so the prefill numbers isolate the GEMM
# path (16-VALU loop vs 0-VALU + A-tiled). Eager decode is launch-dominated and NOT representative
# of prod (compiled graphs + drafter); it is reported only to show v1's two-launch prologue cost.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
GPU=${GPU:-1}; PORT=${PORT:-8081}; MAXLEN=${MAXLEN:-32768}
SIZES=${SIZES:-"2000 8000 16000 30000"}   # target prompt TOKENS
OUT=${OUT:-$HOME/ab_prefill_tp1.txt}
: > "$OUT"

# Preflight: refuse to start into an occupied card. A prod unit auto-started on reboot once and
# held both GPUs; the run then OOM'd 20 minutes in with a misleading error. Fail here instead.
vram_used_mib() { podman run --rm --privileged --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --entrypoint rocm-smi stilldeadcode/vllm-radiance:0.9.3 --showmeminfo vram 2>/dev/null | awk -v g="GPU[$1]" 'index($0, g) && /Used/ {print int($NF/1048576)}'; }
used=$(vram_used_mib "$GPU"); if [ "${used:-0}" -gt 4000 ]; then echo "GPU $GPU has ${used} MiB in use -- not benching into an occupied card" >&2; exit 1; fi

bench() {  # tag
  python3 - "$1" "$PORT" $SIZES <<'PY'
import json, random, string, sys, time, urllib.request
tag, port, sizes = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
URL = f"http://localhost:{port}/v1/completions"
TPW = [1.3]                                            # tokens per random "word"; calibrated below
def words(nw):
    r = random.Random(time.time_ns())
    return " ".join("".join(r.choices(string.ascii_lowercase, k=r.randint(3, 9))) for _ in range(nw))
def prompt(n):
    return words(max(8, int(n / TPW[0])))
def call(p, max_tokens):
    body = json.dumps({"model": "ab", "prompt": p, "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t0, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
_, pt0, _ = call(words(400), 1); TPW[0] = pt0 / 400.0     # random words are ~3.4 tok each here
print(f"{tag:12s} calibrated {TPW[0]:.2f} tok/word", flush=True)
call(prompt(3000), 1)                                  # warm up at real size
for n in sizes:
    best = None
    for _ in range(3):
        dt, pt, _ = call(prompt(n), 1)
        best = min(best, dt) if best else dt
    print(f"{tag:12s} prefill {pt:6d} tok  {pt/best:8.0f} tok/s  ({best*1000:.0f} ms best of 3)", flush=True)
dt, pt, ct = call(prompt(8000), 256)
dt1, _, _ = call(prompt(8000), 1)
print(f"{tag:12s} decode  8k ctx  {ct} tok in {dt-dt1:.2f}s  -> {ct/(dt-dt1):6.1f} tok/s  (EAGER, single stream; indicative only)", flush=True)
PY
}

serve_and_bench() {  # tag model_dir extra-env...
  local tag=$1 dir=$2; shift 2
  echo "=== $tag: $dir ===" | tee -a "$OUT"
  ( env "$@" MODE=eval MODEL_DIR="$dir" TP=1 GPUS="$GPU" MEM_LIMIT=14g GPU_UTIL=0.90 MAXLEN_EVAL="$MAXLEN" \
      NAME=vllmab PORT="$PORT" SERVED_NAMES=ab CHECKALL= RADIANCE_PQM_CHECKALL= \
      "$REPO/paroquant/run_paroquant.sh" > "$HOME/ab_${tag}.log" 2>&1 & )
  for _ in $(seq 1 90); do curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  if ! curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then echo "$tag: server did not come up" | tee -a "$OUT"; tail -5 "$HOME/ab_${tag}.log"; podman stop -t 20 vllmab >/dev/null 2>&1; return 1; fi
  bench "$tag" | tee -a "$OUT"
  echo "$tag  tiled-path evidence: int4 PTOK/ATILED=$(grep -ac 'path=tiled' "$HOME/ab_${tag}.log")  mxfp4 A-tiled calls: $(grep -a 'A-tiled GEMM path' "$HOME/ab_${tag}.log" | tail -1 | grep -oE 'call #[0-9]+' || echo none)" | tee -a "$OUT"
  podman stop -t 20 vllmab >/dev/null 2>&1; sleep 5
}

serve_and_bench int4-paro  Qwen3.8-27B-PARO
serve_and_bench mxfp4-paro Qwen3.8-27B-PARO-MXFP4 RADIANCE_PQ_ROT_STREAM=0 RADIANCE_PQ_ROT_STREAM2=0 RADIANCE_PQ_ROT_STREAM3=0
echo; echo "=== summary written to $OUT ($(date +%H:%M)) ==="
