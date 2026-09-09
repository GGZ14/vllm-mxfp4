#!/bin/bash
# Serve prod (MXFP4-PARO unit config) with an alternative drafter dir and measure: bench_decode_ctx
# (acc/draft, ms/step), BetterBench decode single pass, GSM8K 500q. Usage: gate_drafter.sh <DRAFTER_DIR_NAME> <TAG>
set -u
export XDG_RUNTIME_DIR=/run/user/$(id -u)
DR=$1; TAG=$2; OUT=${OUT:-$HOME/drafter_ft/gate}; mkdir -p "$OUT"
systemctl --user stop qwen_vllm_paro_mxfp4; sleep 8
env MODE=prod SPEC=7 SERVED_NAMES="Qwen3.8 Qwen3.6 Qwen3.8-PARO Qwen3.8-PARO-MXFP4" MODEL_DIR=Qwen3.8-27B-PARO-MXFP4-ft NAME=vllmparomx GPU_UTIL=${GPU_UTIL:-0.95} \
    TP=${TP:-2} GPUS=${GPUS:-0,1} CACHE=${CACHE:-$HOME/.radiance-cache-paro-mxfp4-093-rs-rs2-sl} DRAFTER="$DR" MAXLEN=${MAXLEN:-262144} CHUNK=${CHUNK:-8192} \
    nohup ~/deadcode-vllm/paroquant/run_paroquant.sh > "$OUT/$TAG.serve.log" 2>&1 &
for i in $(seq 1 300); do sleep 5; curl -s -m 3 localhost:8080/v1/models >/dev/null 2>&1 && break; done
curl -s -m 3 localhost:8080/v1/models >/dev/null 2>&1 || { echo "$TAG did not come up"; grep -iE "error|traceback" "$OUT/$TAG.serve.log" | grep -vE "cpuinfo|usage" | tail -5; podman stop -t 30 vllmparomx >/dev/null 2>&1; exit 1; }
echo "=== $TAG up $(date +%H:%M) drafter=$DR"
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{"model":"Qwen3.8-PARO-MXFP4","messages":[{"role":"user","content":"Count from 1 to 30."}],"max_tokens":120,"temperature":0}' | python3 -c "import sys,json; r=json.load(sys.stdin); print('warm:', r['choices'][0]['message']['content'][:50].replace(chr(10),' '))"
(cd ~/mxfp4_work && BENCH_MODEL=Qwen3.8-PARO-MXFP4 BENCH_CTX=0,8000,32000 python3 bench_decode_ctx.py) > "$OUT/$TAG.decode_ctx.log" 2>&1; grep "ms/step" "$OUT/$TAG.decode_ctx.log"
(cd ~/betterbench && python3 -m betterbench.cli run --endpoint http://localhost:8080/v1 --model Qwen3.8-PARO-MXFP4 --passes 1 --warmup 1 --decode --note drafter="$TAG" --out "results/mxfp4paro-drafter-$TAG-decode.json") > "$OUT/$TAG.bb.log" 2>&1; grep -E "Combined|^\| (code|reasoning|prose|json|file_edit|summarization|math|chat) " "$OUT/$TAG.bb.log"
[ "${SKIP_GSM8K:-0}" = 1 ] || python3 ~/pibench-local/gsm8k.py --model Qwen3.8-PARO-MXFP4 --tag "drafter-$TAG" --n 500 2>&1 | tail -1 | tee "$OUT/$TAG.gsm8k.log"
echo "=== $TAG done $(date +%H:%M)"
