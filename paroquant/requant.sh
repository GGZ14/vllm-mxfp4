#!/bin/bash
# Re-quantize Qwen3.8-27B with ParoQuant, keeping the learned rotations but changing the weight
# grid so the serving GEMM's inner loop costs ZERO extra VALU.
#
# FORMAT=mxfp4 (default) writes OCP microscaling MXFP4: e2m1 elements, one e8m0 (power-of-two)
# shared scale per 32 along K -- plus ParoQuant's rotation tensors per projection. MXFP4 removes
# BOTH loop FMAs at once: the e8m0 scale is pow2 so it folds at weight staging, and e2m1 is a
# signed float with no zero point, so the row-sum correction disappears. That is the 0-VALU loop
# (225 TF/s) rather than the 8-VALU one, and it is the same weight traffic (4 + 8/32 = 4.25
# bits/weight, identical to int4 g128 with fp16 scale+zscale).
#
# FORMAT=int keeps the upstream uniform-affine int4 grid; POW2=1 then constrains its group scale
# to powers of two, which removes only the scale FMA (8 of 16 VALU).
#
# Why: the prefill GEMM ablation (paroquant/RESULTS.md, 2026-09-03) showed cost is linear in VALU
# per tile-group -- 16 ops 1.00x, 8 ops 0.90x, 0 ops 0.80x (= MXFP4's 225 TF/s loop). The two
# FMAs are the fp16 group scale and the asymmetric zero point; MXFP4 deletes both.
#
# NOTE ON SERVING: unlike the FORMAT=int/POW2 path, an MXFP4 output is NOT loadable by today's
# radiance_paroquant.py, which expects AWQ int4 buffers. It needs an MXFP4 sibling loader that
# pairs the existing MXFP4 GEMM with the rotation prologue. The optimize stage is the long pole
# and does not depend on that, so it runs first.
#
# Upstream is z-lab/paroquant at 9ee635a plus paroquant_radiance.patch (this repo), which adds:
#   - an MXFP4 (e2m1 + e8m0/32) fake-quantizer with a learnable per-block exponent bias, and the
#     pow2 projection for the int path -- each applied identically in the optimizer and in
#     convert, since a mismatch makes the exported codes disagree with the exported scales,
#   - a ROCm/HIP path for the rotation extension (upstream's load() only hipifies the files it
#     is handed, so rotation.cuh stays CUDA and the build fails),
#   - a capture path that does not move the whole model to GPU (55.6 GiB does not fit 32 GiB;
#     every block is a Catcher during capture and is never executed).
#
# MEMORY: the optimizer keeps the model CPU-resident in fp16 (55.6 GiB) against 60 GiB of RAM.
# Swap is REQUIRED. After the capture phase only one layer (~0.5 GiB) is touched per iteration,
# so the model pages out and stays out; the hot set is the activation shard.
#   sudo fallocate -l 48G /swapfile && sudo chmod 600 /swapfile
#   sudo mkswap /swapfile && sudo swapon /swapfile
set -euo pipefail

SRC=${SRC:-$HOME/paroquant-src}
MODELS=${MODELS:-$HOME/models}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
BASE=${BASE:-Qwen3.8-27B-bf16}
OUT=${OUT:-Qwen3.8-27B-PARO-MXFP4}
RESULTS=${RESULTS:-$HOME/paroquant-out}
CACHE=${CACHE:-$HOME/.cache/paroquant}
# The calibration sets (wikitext2, one C4 shard, the RedPajama *sample*, pileval -- ~8 GiB
# total, not the full corpora) are fetched on first run, so this cannot be offline. Set
# HF_OFFLINE=1 once they are cached.
FORMAT=${FORMAT:-mxfp4}          # mxfp4 | int
POW2=${POW2:-1}                  # only meaningful when FORMAT=int
SCALE_RULE=${SCALE_RULE:-ocp}    # mxfp4 shared exponent: ocp (AMD-compatible) | noclip

# Calibration size. 512 fits the swap budget above; the upstream 27B recipe uses 2048 and the
# 70B one 1024. Raise it (and the swapfile) if the accuracy gate lands just below band -- that
# is the documented retry.
TRAIN_SIZE=${TRAIN_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-4}      # logits are batch x 2048 x 248064 x 2 B = 1 GiB per sample
CACHE_SHARDS=${CACHE_SHARDS:-16} # only one shard is resident on GPU at a time
SEQLEN=${SEQLEN:-2048}
EPOCHS=${EPOCHS:-"2 2"}
# Stage-1 (channel_scales + rotation angles) learning rate. Upstream's 27B recipe uses 0.05, but
# measured on this stack the rotation stage stops contributing past ~layer 10 and lands at
# EXACTLY 0.0% on several layers -- it diverges on the first epoch and best_sd keeps the starting
# point. Each layer is optimized independently against its own captured inputs/outputs, so there
# is no cross-layer consistency requirement and this may be tuned freely.
ROT_LR=${ROT_LR:-0.05}
WEIGHT_LR=${WEIGHT_LR:-1e-5}
QUANT_LR=${QUANT_LR:-1e-6}
STAGE=${STAGE:-all}              # all | optimize | pseudo | convert
PSEUDO_OUT=${PSEUDO_OUT:-Qwen3.8-27B-PARO-MXFP4-pseudo}

[ -d "$MODELS/$BASE" ] || { echo "base model missing at $MODELS/$BASE" >&2; exit 1; }
[ -d "$SRC/paroquant" ] || { echo "paroquant source missing at $SRC" >&2; exit 1; }
if [ "$(free -g | awk '/^Swap:/{print $2}')" -lt 32 ]; then
  echo "WARNING: less than 32 GiB of swap; the 27B optimize run will be OOM-killed." >&2
  echo "  sudo fallocate -l 48G /swapfile && sudo chmod 600 /swapfile" >&2
  echo "  sudo mkswap /swapfile && sudo swapon /swapfile" >&2
  [ "${FORCE:-0}" = 1 ] || exit 1
fi
mkdir -p "$RESULTS" "$CACHE"

run() {
  # --privileged and --ipc=host are load-bearing: without them a side HIP container faults on
  # the first GPU allocation ("Memory critical error ... Reason: Memory in use") even for a
  # plain matmul, with the GPUs idle. Recorded trap; do not trim these.
  podman run --rm --network=host --name paro_requant \
    --privileged --ipc=host \
    --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 -e PYTORCH_ROCM_ARCH=gfx1201 \
    -e PYTHONPATH=/src:/src/.pydeps \
    -e PARO_QUANT_FORMAT="$FORMAT" \
    -e PARO_MXFP4_SCALE_RULE="$SCALE_RULE" \
    -e PARO_POW2_SCALES="$POW2" \
    -e HF_HUB_OFFLINE="${HF_OFFLINE:-0}" -e HF_HOME=/root/.cache/huggingface \
    -v "$SRC":/src:z -v "$MODELS":/models -v "$RESULTS":/out:z \
    -v "$CACHE":/root/.cache/paroquant:z \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /src --entrypoint python3 "$IMAGE" "$@"
}

if [ "$STAGE" = all ] || [ "$STAGE" = optimize ]; then
  echo "=== optimize (format=$FORMAT rule=$SCALE_RULE train_size=$TRAIN_SIZE epochs=$EPOCHS rot_lr=$ROT_LR) ==="
  # shellcheck disable=SC2086
  run -m paroquant.cli.optimize \
    --model "/models/$BASE" \
    --params "channel_scales:$ROT_LR,angles:$ROT_LR" "weight:$WEIGHT_LR,quantizer:$QUANT_LR" \
    --epochs $EPOCHS \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --skipped-modules "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
    --datasets wikitext2 c4 redpajama \
    --val-dataset pileval \
    --train-size "$TRAIN_SIZE" \
    --validation-size 64 \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps 1 \
    --seqlen "$SEQLEN" \
    --cache-shards "$CACHE_SHARDS" \
    --output-dir /out \
    --resume \
    --seed 0
fi

# The accuracy gate. --mode pseudo writes fp16 weights that have been through
# rotate -> MXFP4 quantize -> inverse rotate, so they carry the EXACT error of the scheme while
# loading on the stock path. That measures quality without needing the MXFP4+rotation serving
# kernel, so accuracy can be gated before any kernel work is done.
if [ "$STAGE" = pseudo ]; then
  echo "=== pseudo-quantize -> $MODELS/$PSEUDO_OUT (fp16, ~55 GiB) ==="
  run -m paroquant.cli.convert \
    --model "/models/$BASE" \
    --result-dir "/out/$BASE" \
    --output-path "/models/$PSEUDO_OUT" \
    --mode pseudo
  echo "gate it:  serve $MODELS/$PSEUDO_OUT on the stock path and run GSM8K"
fi

# NOTE (real mode): RotateQuantizedLinear registers AWQ buffer names and convert loads them with
# strict=False, so the MXFP4 names ("weight"/"weight_scale") would be silently DROPPED and the
# checkpoint written as zeros. Fix that before trusting STAGE=convert output for MXFP4.
if [ "$STAGE" = all ] || [ "$STAGE" = convert ]; then
  echo "=== convert -> $MODELS/$OUT ==="
  run -m paroquant.cli.convert \
    --model "/models/$BASE" \
    --result-dir "/out/$BASE" \
    --output-path "/models/$OUT" \
    --mode real
  echo
  echo "serve it with the existing stack (the format is unchanged):"
  echo "  MODE=eval MODEL=/models/$OUT ./paroquant/run_paroquant.sh"
fi
