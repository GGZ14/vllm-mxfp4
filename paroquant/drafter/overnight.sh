#!/bin/bash
# Overnight drafter fine-tune pipeline. Waits for generation, takes prod down, trains on GPU 0
# (memory fallbacks), exports FP8, gates NEW vs OLD drafter on the prod config, restores prod with
# the winner (new only if BetterBench combined >= old + 2% and GSM8K >= 97.0).
set -u
export XDG_RUNTIME_DIR=/run/user/$(id -u)
D=$HOME/drafter_ft; S=$HOME/deadcode-vllm/paroquant/drafter; LOG=$D/overnight.log
NEW=Qwen3.8-27B-DFlash2-FP8-paro; OLD=Qwen3.8-27B-DFlash2-FP8
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
# 1. wait for generation (client exits when done)
while pgrep -f '^python3 /home/brian/deadcode-vllm/paroquant/drafter/generate.py' >/dev/null; do sleep 60; done
say "generation finished: $(wc -l < $D/responses.jsonl) responses, $(ls $D/capture | wc -l) capture files, $(du -sh $D/capture | cut -f1)"
sleep 30   # let the last requests flush
# 2. prod down, capture drop-in removed (prod comes back later WITHOUT capture / with prefix cache)
rm -f ~/.config/systemd/user/qwen_vllm_paro_mxfp4.service.d/capture.conf; systemctl --user daemon-reload
systemctl --user stop qwen_vllm_paro_mxfp4; sleep 10; say "prod stopped for training"
# 3. train (GPU 0), fallbacks on failure
run_train() {
  podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -v $D:/data:z -v $HOME/models:/models:z -v $S:/scripts:z --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc \
    "cd /data && python3 /scripts/train_drafter.py --capture /data/capture --drafter /models/Qwen3.8-27B-DFlash2-bf16 --target /models/Qwen3.8-27B-bf16 --out /data/ft_bf16 --epochs 2 --lr 5e-5 --val 96 --eval-every 150 --save-every 300 $*" >> $D/train.log 2>&1
}
say "training: seqs 4 x anchors 64"; run_train --seqs 4 --anchors 64 && ok=1 || ok=0
if [ $ok = 0 ]; then say "retry: seqs 2 x anchors 48"; run_train --seqs 2 --anchors 48 && ok=1; fi
if [ $ok = 0 ]; then say "retry: frozen MLPs"; run_train --seqs 4 --anchors 64 --freeze-mlp && ok=1; fi
if [ $ok = 0 ]; then say "TRAINING FAILED -- restoring prod"; systemctl --user start qwen_vllm_paro_mxfp4; exit 1; fi
grep -E "\[eval" $D/train.log | tail -20 | tee -a "$LOG"
# 4. export FP8 in tcclaviger's layout
podman run --rm -v $D:/data:z -v $HOME/models:/models:z -v $S:/scripts:z --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc \
  "python3 /scripts/export_fp8.py /data/ft_bf16 /models/$OLD /models/$NEW" 2>&1 | tee -a "$LOG"
[ -f $HOME/models/$NEW/model.safetensors ] || { say "EXPORT FAILED -- restoring prod"; systemctl --user start qwen_vllm_paro_mxfp4; exit 1; }
# 5. gate new, then old, same config
$S/gate_drafter.sh $NEW new 2>&1 | tee -a "$LOG"; podman stop -t 30 vllmparomx >/dev/null 2>&1; sleep 8
$S/gate_drafter.sh $OLD old 2>&1 | tee -a "$LOG"; podman stop -t 30 vllmparomx >/dev/null 2>&1; sleep 8
cn=$(grep -oE "median ≈ \*\*[0-9.]+" $D/gate/new.bb.log | grep -oE "[0-9.]+$"); co=$(grep -oE "median ≈ \*\*[0-9.]+" $D/gate/old.bb.log | grep -oE "[0-9.]+$")
gn=$(grep -oE "accuracy [0-9.]+" $D/gate/new.gsm8k.log | grep -oE "[0-9.]+"); go=$(grep -oE "accuracy [0-9.]+" $D/gate/old.gsm8k.log | grep -oE "[0-9.]+")
say "RESULT combined t/s new=$cn old=$co | GSM8K new=$gn old=$go"
win=$(python3 -c "import sys; cn,co,gn=float('${cn:-0}'),float('${co:-0}'),float('${gn:-0}'); print('new' if cn>=co*1.02 and gn>=97.0 else 'old')")
if [ "$win" = new ]; then
  for f in ~/.config/systemd/user/qwen_vllm_paro_mxfp4.service ~/deadcode-vllm/paroquant/qwen_vllm_paro_mxfp4.service; do
    grep -q "^Environment=DRAFTER=" "$f" || sed -i "/^Environment=NAME=vllmparomx$/a Environment=DRAFTER=$NEW" "$f"; done
  systemctl --user daemon-reload; say "NEW drafter wins -> prod unit now DRAFTER=$NEW"
else
  say "OLD drafter kept (new did not clear the bar)"
fi
systemctl --user start qwen_vllm_paro_mxfp4; say "prod restarted with $win drafter"
