#!/bin/bash
# SPEC re-sweep on the Paro serve: boot a manual serve per SPEC, run bench_decode_ctx (0/8k/32k)
# and a BetterBench decode single pass, stop it. Results land in $OUT/spec<N>.*.
set -u
export XDG_RUNTIME_DIR=/run/user/$(id -u)
OUT=${OUT:-$HOME/mxfp4_work/paro/spec_sweep}
mkdir -p "$OUT"
systemctl --user stop qwen_vllm_paro
sleep 5
for SPEC in "$@"; do
  echo "=== SPEC=$SPEC boot $(date)"
  MODE=prod SPEC=$SPEC SERVED_NAMES="Qwen3.8 Qwen3.6 Qwen3.8-PARO" nohup ~/mxfp4_work/paro/run_paroquant.sh > "$OUT/spec$SPEC.serve.log" 2>&1 &
  for i in $(seq 1 240); do sleep 5; curl -s -m 3 localhost:8080/v1/models >/dev/null 2>&1 && break; done
  curl -s -m 3 localhost:8080/v1/models >/dev/null 2>&1 || { echo "SPEC=$SPEC did not come up"; podman stop -t 30 vllmparo; continue; }
  echo "=== SPEC=$SPEC up $(date)"
  (cd ~/mxfp4_work && BENCH_MODEL=Qwen3.8-PARO BENCH_CTX=0,8000,32000 python3 bench_decode_ctx.py) > "$OUT/spec$SPEC.decode_ctx.log" 2>&1
  (cd ~/betterbench && python3 -m betterbench.cli run --endpoint http://localhost:8080/v1 --model Qwen3.8 --passes 1 --warmup 1 --decode --note build=sep03-rotstream --note spec=$SPEC --out results/paro-spec$SPEC-decode.json) > "$OUT/spec$SPEC.bb.log" 2>&1
  echo "=== SPEC=$SPEC done $(date)"
  podman stop -t 30 vllmparo >/dev/null 2>&1
  sleep 5
done
echo SWEEP_DONE
