#!/bin/bash
# Iterative drafter loop. Prod serves at TP=1 on GPU 0 with capture on (unit drop-in loop.conf);
# GPU 1 is the training lane. Per round: sample prompts from the pool -> generate on prod (captured)
# -> train one epoch from the best drafter so far on the last two rounds' data -> export FP8 ->
# gate at TP=1 (decode bench + BetterBench decode) -> promote if it beats the best -> prune old
# captures. State in $L: best.txt (bf16 dir|fp8 name), best_bb.txt, pool_used.txt, round logs.
# Usage: loop.sh <rounds> [prompts_per_round]
set -u
export XDG_RUNTIME_DIR=/run/user/$(id -u)
ROUNDS=${1:-6}; PER=${2:-1500}
D=$HOME/drafter_ft; L=$D/loop; S=$HOME/deadcode-vllm/paroquant/drafter; mkdir -p "$L" "$L/capture_live" "$L/gate"
LOG=$L/loop.log; say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
UNIT=qwen_vllm_paro_mxfp4; DROP=~/.config/systemd/user/$UNIT.service.d/loop.conf
SERVE_ENV="MODE=prod SPEC=7 SERVED_NAMES=Qwen3.8_Qwen3.6_Qwen3.8-PARO_Qwen3.8-PARO-MXFP4 MODEL_DIR=Qwen3.8-27B-PARO-MXFP4-ft NAME=vllmparomx"
TP1_CACHE=$HOME/.radiance-cache-paro-mxfp4-093-rs-rs2-sl-tp1
GE=""
[ -f "$L/best.txt" ] || { echo "need $L/best.txt = <bf16 dir>|<fp8 model dir name>"; exit 1; }
BEST_BF16=$(cut -d'|' -f1 "$L/best.txt"); BEST_FP8=$(cut -d'|' -f2 "$L/best.txt")
touch "$L/pool_used.txt"

LOOP_TP=${LOOP_TP:-2}   # 1: TP=1 serve on GPU 0 + training lane on GPU 1 (does not fit this model: 20 GiB weights + ~6.5 GiB non-torch on a 32 GB card); 2: prod at TP=2, stopped during training
prod_up() {   # capture into capture_live, prefix cache off, given drafter; TP per LOOP_TP
  mkdir -p "$(dirname "$DROP")"
  if [ "$LOOP_TP" = 1 ]; then
    printf '[Service]\nEnvironment=TP=1\nEnvironment=GPUS=0\nEnvironment=GPU_UTIL=0.97\nEnvironment=MAXLEN=16384\nEnvironment=CHUNK=4096\nEnvironment=CACHE=%s\nEnvironment=CAPTURE_DIR=%s\nEnvironment=PREFIX_CACHE=0\nEnvironment=DRAFTER=%s\n' "$TP1_CACHE" "$L/capture_live" "$1" > "$DROP"
  else
    printf '[Service]\nEnvironment=CAPTURE_DIR=%s\nEnvironment=PREFIX_CACHE=0\nEnvironment=DRAFTER=%s\n' "$L/capture_live" "$1" > "$DROP"
  fi
  systemctl --user daemon-reload; systemctl --user restart $UNIT
  for i in $(seq 1 240); do sleep 10; curl -sf localhost:8080/v1/models >/dev/null 2>&1 && return 0; done
  say "prod (TP=1, $1) did not come up"; return 1
}
gate_tp1() {  # $1 = fp8 dir name, $2 = tag ; leaves the unit stopped; TP per LOOP_TP
  if [ "$LOOP_TP" = 1 ]; then GE="TP=1 GPUS=0 GPU_UTIL=0.97 MAXLEN=16384 CHUNK=4096 CACHE=$TP1_CACHE"; else GE="TP=2 GPUS=0,1 GPU_UTIL=0.95 CACHE=$HOME/.radiance-cache-paro-mxfp4-093-rs-rs2-sl-sk RADIANCE_SKINNY_GEMM=all"; fi
  env $GE OUT=$L/gate SKIP_GSM8K=${SKIP_GSM8K:-1} "$S/gate_drafter.sh" "$1" "$2" 2>&1 | tee -a "$LOG"
  podman stop -t 30 vllmparomx >/dev/null 2>&1; sleep 8
  grep -oE "median ≈ \*\*[0-9.]+" "$L/gate/$2.bb.log" | grep -oE "[0-9.]+$"
}

# baseline gate for the current best at TP=1 (once)
if [ ! -f "$L/best_bb.txt" ]; then
  say "baseline gate (LOOP_TP=$LOOP_TP) for $BEST_FP8"; bb=$(gate_tp1 "$BEST_FP8" best_tp1 | tail -1)
  python3 -c "float('${bb:-x}')" 2>/dev/null || { say "baseline gate failed ($bb) -- restoring prod at TP=2 and stopping"; rm -f "$DROP"; systemctl --user daemon-reload; systemctl --user start $UNIT; exit 1; }
  echo "$bb" > "$L/best_bb.txt"; say "best BB combined @TP1 = $bb"
fi
prod_up "$BEST_FP8" || exit 1

for r in $(seq 1 "$ROUNDS"); do
  # resume: if the newest capture round was never trained to completion (e.g. a reboot mid-training),
  # take it as this round and skip generation
  LAST=$(ls "$L" | grep -E '^capture_r[0-9]+$' | sed 's/capture_r//' | sort -n | tail -1); RESUME=0
  if [ -n "$LAST" ] && [ -d "$L/capture_r$LAST" ] && ! grep -qs "\[eval after\]" "$L/train_r$LAST.log"; then N=$LAST; RESUME=1; else N=$((${LAST:-0} + 1)); fi
  say "===== round $N (loop iteration $r, resume=$RESUME) best=$BEST_FP8 BB=$(cat $L/best_bb.txt)"
  if [ "$RESUME" = 0 ]; then
  # 1. sample prompts
  python3 - "$D/pool.jsonl" "$L/pool_used.txt" "$L/prompts_r$N.jsonl" "$PER" <<'PY'
import json, random, sys
pool, used_f, out, per = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
used = set(l.strip() for l in open(used_f) if l.strip())
rows = [json.loads(l) for l in open(pool)]; rows = [r for r in rows if str(r["id"]) not in used]
random.Random(len(used)).shuffle(rows); pick = rows[:per]
with open(out, "w") as f:
    for r in pick: f.write(json.dumps(r) + "\n")
with open(used_f, "a") as f:
    for r in pick: f.write(str(r["id"]) + "\n")
from collections import Counter; print("sampled", len(pick), "remaining", len(rows) - len(pick), dict(Counter(r["source"] for r in pick).most_common(8)))
PY
  # 2. generate (captured)
  python3 "$S/generate.py" "$L/prompts_r$N.jsonl" "$L/responses_r$N.jsonl" 8 768 2>&1 | tail -2 | tee -a "$LOG"
  sleep 40; mkdir -p "$L/capture_r$N"; find "$L/capture_live" -maxdepth 1 -name '*.pt' -size +1200k -exec mv {} "$L/capture_r$N/" \; ; find "$L/capture_live" -maxdepth 1 -name '*.pt' -delete
  say "round $N: $(ls $L/capture_r$N | wc -l) captures, $(du -sh $L/capture_r$N | cut -f1), disk free $(df -h /home | awk 'NR==2{print $4}')"
  fi   # RESUME
  # 3. train from best on the last two rounds (round 1 also sees the big round-2 set). TP=2 mode: prod
  #    is stopped for the training window (both cards busy otherwise); TP=1 mode: GPU 1 lane.
  if [ "$LOOP_TP" = 2 ]; then systemctl --user stop $UNIT; sleep 8; say "round $N: prod stopped for training"; fi
  DATA="$L/capture_r$N"; [ -d "$L/capture_r$((N-1))" ] && DATA="$DATA,$L/capture_r$((N-1))"; [ $N = 1 ] && DATA="$DATA,$D/capture2"
  podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=$([ "$LOOP_TP" = 1 ] && echo 1 || echo 0) -v $D:/data:z -v $HOME/models:/models:z -v $S:/scripts:z --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc \
    "cd /data && python3 /scripts/train_drafter.py --capture $(echo $DATA | sed "s|$D|/data|g") --val-dir /data/val_fixed --drafter $(echo $BEST_BF16 | sed "s|$D|/data|") --target /models/Qwen3.8-27B-embed-head --out /data/loop/ft_r$N --epochs 1 --lr 5e-5 --seqs 6 --anchors 64 --eval-every 200 --save-every 100000" > "$L/train_r$N.log" 2>&1
  grep -E "\[eval (before|after)\]" "$L/train_r$N.log" | tee -a "$LOG"
  # the big round-2 set is baked into the round-1 model: free its 117 GB once it has been trained on
  [ $N = 1 ] && [ -f "$L/ft_r1/model.safetensors" ] && rm -rf "$D/capture2" && say "capture2 removed after round 1 training"
  before=$(grep "\[eval before\]" "$L/train_r$N.log" | grep -oE "accepted/block [0-9.]+" | grep -oE "[0-9.]+$"); after=$(grep "\[eval after\]" "$L/train_r$N.log" | grep -oE "accepted/block [0-9.]+" | grep -oE "[0-9.]+$")
  [ -f "$L/ft_r$N/model.safetensors" ] || { say "round $N: training failed"; continue; }
  # 4. export
  CAND=Qwen3.8-27B-DFlash2-FP8-paro-r$N
  podman run --rm -v $D:/data:z -v $HOME/models:/models:z -v $S:/scripts:z --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc "python3 /scripts/export_fp8.py /data/loop/ft_r$N /models/Qwen3.8-27B-DFlash2-FP8 /models/$CAND" | tee -a "$LOG"
  # 5. gate at TP=1 (prod is stopped by the gate script; restarted below)
  systemctl --user stop $UNIT; sleep 8
  bb=$(gate_tp1 "$CAND" "r$N"); best_bb=$(cat "$L/best_bb.txt")
  win=$(python3 -c "b,c,a,f=float('${best_bb:-0}'),float('${bb:-0}'),float('${after:-0}'),float('${before:-0}'); print('yes' if c>=b*1.015 and a>=f else 'no')")
  say "round $N: proxy $before -> $after | BB combined @TP1 cand=$bb best=$best_bb | promote=$win"
  if [ "$win" = yes ]; then
    if [ "${SKIP_GSM8K:-1}" = 1 ]; then   # quality check only on promotion candidates
      env $GE OUT=$L/gate SKIP_GSM8K=0 "$S/gate_drafter.sh" "$CAND" "r${N}_gsm" 2>&1 | grep -E "accuracy" | tee -a "$LOG"; podman stop -t 30 vllmparomx >/dev/null 2>&1; sleep 8
      g=$(grep -oE "accuracy [0-9.]+" "$L/gate/r${N}_gsm.gsm8k.log" | grep -oE "[0-9.]+"); python3 -c "import sys; sys.exit(0 if float('${g:-0}')>=97.0 else 1)" || { say "round $N: GSM8K $g < 97.0, NOT promoted"; win=no; }
    fi
  fi
  if [ "$win" = yes ]; then
    BEST_BF16="$L/ft_r$N"; BEST_FP8="$CAND"; echo "$BEST_BF16|$BEST_FP8" > "$L/best.txt"; echo "$bb" > "$L/best_bb.txt"
    for f in ~/.config/systemd/user/$UNIT.service ~/deadcode-vllm/paroquant/$UNIT.service; do grep -q "^Environment=DRAFTER=" "$f" && sed -i "s|^Environment=DRAFTER=.*|Environment=DRAFTER=$CAND|" "$f" || sed -i "/^Environment=NAME=vllmparomx$/a Environment=DRAFTER=$CAND" "$f"; done
    say "round $N: PROMOTED $CAND"
  else
    rm -rf "$L/ft_r$N"   # keep the FP8 dir for inspection, drop the bf16
  fi
  # 6. prune captures older than two rounds, restart prod for the next round
  [ -d "$L/capture_r$((N-2))" ] && rm -rf "$L/capture_r$((N-2))"
  prod_up "$BEST_FP8" || exit 1
done
say "LOOP DONE best=$BEST_FP8 BB=$(cat $L/best_bb.txt)"
