#!/bin/bash
# HISTORICAL REFERENCE -- the 0.5.8 launch, kept only because it is what the baseline numbers in
# README.md were measured with. It has absolute paths from the box it ran on and points at an
# image that predates this fork's MXFP4 work. For the current build use ./serve-mxfp4.sh.
# EVALUATION (not production): native MXFP4 body on gfx1201, with the MTP drafter in FP8.
#
# Checkpoint built by this repo's ./fp8_mtp.py from amd/Qwen3.8-27B-Quark-AWQ-MXFP4, served on
# stilldeadcode/vllm-radiance:0.5.8 with patch_quark_mxfp4.py applied at container start.
#
# The drafter is NOT MXFP4, and that is a settled result, not an oversight. AMD ships the MTP
# drafter BF16; at n=8 it runs 8 draft passes per verify step and is 34% of decode weight traffic,
# so quantizing it is worth real bandwidth. MXFP4 on it failed twice: data-free RTN (~11.6% rel
# error) cost acceptance 2.5 -> 2.21, and AWQ calibration did not rescue it (0-5% error
# improvement; the alpha search chose 0.1-0.2, and 0.0 for mtp.fc, because MXFP4's per-32 E8M0
# block exponent already does most of what per-channel scaling would). The error is intrinsic to
# 4 bits, and for a drafter accuracy IS throughput. FP8 e4m3 per-channel is ~2-3% rel error and
# removes ~17% of decode weight traffic instead of 25% -- the smaller win that actually holds:
# measured acceptance 2.60-2.80 vs the MXFP4 drafter's 2.21-2.61.
# The MXFP4-drafter builds (-mtpq / -mtpawq) exist only for that comparison and are not shipped;
# do not point this script at their checkpoints.
#
# Derived from run_radiance_prod_38.sh (2026-08-20 state). Serve flags and RADIANCE_* values are
# inherited from prod EXCEPT where listed below, so the numbers stay comparable to
# ~/bench_fp8_rebaseline_20260819.log -- but note the tuning deltas at the end of this list: this
# is no longer a pure quantization-only A/B against prod.
# Deliberate differences, and only these:
#
#   - container vllm38 -> vllmminm; cache .radiance-cache-058 -> .radiance-cache-w4a8-058.
#     Cache dirs validate on model + torch/Triton version and MUST NOT be shared.
#   - model -> the MXFP4 snapshot; --served-model-name Qwen3.8 Qwen3.6 Qwen3.8-MXFP4, NOT the prod ids, so nothing
#     routes here by accident.
#   - --quantization fp8 DROPPED: config.json declares quantization_config.quant_method=quark
#     (mxfp4 weights AND activations, group 32, e8m0 scales) and the runtime routes it itself.
#   - RADIANCE_MXFP4=1: patch_quark_mxfp4.py stops forcing vLLM's emulated MXFP4 path on gfx12x and
#     lets aiter's gemm_afp4wfp4 (Triton tl.dot_scaled -> bf16 WMMA) serve the mxfp4 x mxfp4 linears.
#     Measured bit-identical to emulation -- the activation quantization is the same either way --
#     so this is purely a speed change, not a quality one.
#   - the patch and the gfx1201 tiles are applied by an entrypoint prelude rather than baked into the
#     image: the image builds torch/triton/vllm/aiter from source, so a rebuild costs hours and buys
#     nothing for an eval. ~/deadcode-vllm's Dockerfile carries the durable form of the same two
#     changes (COPY mxfp4-configs/ + patch_quark_mxfp4 in the patch list).
#   - RADIANCE_MXFP4_MAX_M=1e9 disables the large-M fallback to emulation. The fallback is a
#     throughput optimisation on paper (emulation overtakes the fp4 kernel past M~256) but it
#     CANNOT RUN on this stack: vLLM's quant_dequant_mxfp4 dispatches to
#     quark/torch/kernel/mx/hip.py, whose TileLang backend dies with "HIP runtime library
#     (libamdhip64.so) not found" inside the vLLM worker. Worse, the branch gets specialised into
#     the torch.compile graph during the M=8192 profile run, so it kills startup rather than one
#     request. Corollary: the *stock* emulated MXFP4 path cannot serve this checkpoint here either,
#     which makes the native kernel the only way to run this model on this box, not merely a faster
#     one.
#   - RADIANCE_MXFP4_W4A8=1 with RADIANCE_MXFP4_W4A8_MIN_M=16 routes large-M (prefill) MXFP4
#     linears to the hand-written W4A8 fp8-WMMA kernel, leaving decode on aiter's W4A4 Triton path.
#     Triton lowers tl.dot_scaled by upconverting e2m1 to bf16 and using the 16-bit WMMA; fp8 WMMA
#     measures 325.2 TFLOP/s vs f16's 160.2 here, and the hand-written kernel runs 1.6-1.9x the
#     tuned aiter path at prefill shapes. This is a NUMERICS change, not only a speed one: the
#     checkpoint declares W4A4 and this runs W4A8 (fp8 activations are strictly more precise than
#     the fp4 the model was calibrated against, but output is no longer bit-identical to
#     emulation). MIN_M=16 is well below radiance_mxfp4.py's documented default of 256, i.e. the
#     kernel is deliberately used far under its measured fp8-wins-here crossover -- if a prefill
#     regression ever shows up, put MIN_M back to 256 first.
#   - RADIANCE_AR_MAX_KB=98304 (prod: 32768). The cap excludes prefill, so prod's default sends
#     every 80 MiB prefill all-reduce to NCCL; raising it keeps them on the fast reduce path and
#     measured +3-13% prefill for free.
#   - RADIANCE_DRAFT_TAU=0.20 (radiance_draft.py default: 0.35) lowers the confidence-product stop
#     threshold, so drafts run longer before the drafter gives up.
#   - --kv-cache-memory=19105177314 (17.79 GiB) sizes the KV cache explicitly instead of letting
#     0.97 minus the measured weights/activations decide it. vLLM 0.26's CUDA-graph memory profiler
#     is conservative -- it warns that 0.97 here behaves like 0.904 -- which left 1.2 GiB/GPU
#     unused (29.39 of 31.86 GiB in use, 92.3%). The value is vLLM's own "fit into requested
#     memory" recommendation for TP0, taken from the 2026-08-23 startup log; TP0 is the binding
#     rank because it carries more non-torch memory (1.44 vs 1.25 GiB) and one value applies to
#     every rank, so the smaller of the two must be used or TP0 OOMs. Effect: KV 16.59 -> 17.79 GiB,
#     856,151 -> ~918k tokens (+7.3%). Do NOT raise this to the "fully utilize" figure without
#     rechecking both ranks: that one spends the 3% headroom the 0.97 request is reserving.
#   - RADIANCE_PRESHUFFLE and RADIANCE_FUSE_RMS_QUANT keep prod's values but are INERT here: both
#     hook the fp8 linear path, which this checkpoint never takes. Left set so the env diff vs prod
#     stays minimal.
#
#   - --gpu-memory-utilization 0.97 (prod: 0.92). More of the card is given to KV here because
#     the mxfp4 weights are roughly half the size of prod's fp8 ones.
#
# Port 8080 is prod's port and this needs both GPUs at 0.97, so production must be stopped first:
#   systemctl --user stop qwen_vllm_38          restore with: vllm-switch 38
#
# What to check in the log:
#   "[radiance] native MXFP4 enabled on gfx12x"   -> the native path is live
#   the stock "current platform does not support native MXFP4/MXFP6" notice still prints; it is
#   emitted from a separate supports_mx() check and does NOT mean the layers were emulated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$HOME/models/Qwen3.8-27B-MXFP4-mtpfp8"
[ -f "$SNAP/config.json" ] || { echo "no checkpoint at $SNAP" >&2; exit 1; }
# HF_HUB_OFFLINE=1 inside the container, and the cache mounts at /root/.cache/huggingface, so vllm
# must be handed the CONTAINER path (a host path fails HF repo-id validation, not "not found").
CSNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8

mkdir -p "$HOME"/.radiance-cache-w4a8-058/{vllm,inductor,triton,aiter}

exec podman run --replace --name vllmminm --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES=0,1 -e HIP_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_ATTN_TUNE=1 -e RADIANCE_FAST_REDUCE=1 \
  -e RADIANCE_AR_MAX_KB=98304 -e RADIANCE_FUSE_RMS_QUANT=1 -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_MAX_M=1000000000 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=16 -e RADIANCE_DRAFT_TAU=0.20 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/models":/models \
  -v "$HOME/.radiance-cache-w4a8-058":/cache \
  -v "$SCRIPT_DIR":/patches:z \
  --entrypoint bash \
  stilldeadcode/vllm-radiance:0.5.8 -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches && python3 patch_quark_mxfp4.py
    # Non-fatal: fixes content=null on thinking-off requests; not required to serve. See prod script.
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    cp radiance_mxfp4.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    exec /opt/radiance_entrypoint.sh "$@"' _ \
    "$CSNAP" --served-model-name Qwen3.8 Qwen3.6 Qwen3.8-MXFP4 --host 0.0.0.0 --port 8080 \
    --kv-cache-dtype fp8 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.97 --kv-cache-memory 19105177314 \
    --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' \
    --no-async-scheduling \
    --enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja
# NOTE: this script is FROZEN at the 0.5.8 baseline, template included -- the numbers it reproduces
# were measured with that exact file, so it is not switched to the repo template the way
# serve-mxfp4.sh is. It is host-local: supply your own at that path, or serve with ./serve-mxfp4.sh.
