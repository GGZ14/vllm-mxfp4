#!/bin/bash
# Serve z-lab/Qwen3.8-27B-PARO through the radiance ParoQuant W4A8 stack (gfx1201, TP2).
#
# Mirrors run_mxfp4_074_kvgroup.sh's env and patch list so numbers compare like-for-like against
# the MXFP4 production server. Deliberate differences, and only these:
#   - container vllmparo; cache ~/.radiance-cache-paro-093 (compile caches validate on model +
#     torch/Triton version and MUST NOT be shared across checkpoints).
#   - model -> the PARO snapshot; --served-model-name Qwen3.8-PARO, NOT the prod ids.
#   - the paroquant quant method + kernels are built into site-packages at container start
#     (config.json declares quant_method=paroquant; a sitecustomize import registers it in the
#     engine and every TP worker).
#   - no --kv-cache-memory pin yet: the PARO weight footprint differs from MXFP4's, so the tuned
#     18563072000 could OOM. First boots run util 0.92; pin after measuring.
#   - MXFP4-only env knobs are left at prod values but are INERT here (no quark layers load).
#
# MODE=eval  (default): --enforce-eager, CHECKALL numerics gate on the four model shapes,
#                       no speculative decoding, 32K ctx. For correctness gating only.
# MODE=prod           : full config -- DFlash2 FP8 drafter (SPEC tokens configurable), 262K ctx,
#                       compiled graphs.
#
# Port 8080 is prod's port and both need both GPUs: stop production first
#   systemctl --user stop qwen_vllm_38        restore with: vllm-switch 38
set -euo pipefail

MODE=${MODE:-eval}
PORT=${PORT:-8080}
NAME=${NAME:-vllmparo}
# Space-separated served ids. Eval default answers ONLY to Qwen3.8-PARO so nothing pinned to the
# prod ids routes here by accident; the qwen_vllm_paro systemd unit (vllm-switch paro) overrides
# with the prod ids so clients like the Pi (model id Qwen3.6) work unchanged.
SERVED_NAMES=${SERVED_NAMES:-Qwen3.8-PARO}
SPEC=${SPEC:-5}      # dflash re-sweep 2026-08: 5 beats 7 by 8-13% aggregate on this stack
GPU_UTIL=${GPU_UTIL:-0.92}

MODEL=/models/Qwen3.8-27B-PARO
[ -d "$HOME/models/Qwen3.8-27B-PARO" ] || { echo "model missing" >&2; exit 1; }

if [ "$MODE" = eval ]; then
  EXTRA_ARGS=(--enforce-eager --max-model-len 32768 --max-num-seqs 8
              --max-num-batched-tokens 8192)
  # Per-rank quantized shapes: qkv, o, gate_up, down, in_proj(+merge), out_proj
  CHECKALL=${CHECKALL:-"7168:5120,5120:3072,17408:5120,5120:8704,8192:5120,5120:3072"}
  SPEC_ARGS=()
else
  EXTRA_ARGS=(--max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
              --enable-prefix-caching
              --compilation-config
              '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true},"compile_sizes":[1,2,4,8],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false,"combo_kernels":true,"benchmark_combo_kernel":true,"triton.cooperative_reductions":true}}')
  CHECKALL=${CHECKALL:-}
  SPEC_ARGS=(--speculative-config
    "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-FP8\",\"num_speculative_tokens\":${SPEC},\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":true,\"draft_sample_method\":\"greedy\"}")
fi

exec podman run --replace --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES=0,1 -e HIP_VISIBLE_DEVICES=0,1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_AR=1 -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB=86016 \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_FUSE_RMS_QUANT=1 \
  -e R4D_ATTN_FP8=3 \
  -e RADIANCE_GDN_FUSED_UPDATE=0 \
  -e RADIANCE_DYNAMIC_WIDTH=1 -e RADIANCE_DYNW_ALPHA=0.35 -e RADIANCE_DYNW_MARGIN=2 \
  -e RADIANCE_DYNW_MIN=2 -e RADIANCE_DYNW_MIN_BATCH=3 \
  -e RADIANCE_AR_QNB=96 -e RADIANCE_AR_QNT=1024 -e RADIANCE_AR_OVERLAP=0 \
  -e RADIANCE_DFLASH_SELECTOR_TOPK= \
  -e RADIANCE_PAROQUANT=1 \
  -e RADIANCE_PQ_CHECKALL="$CHECKALL" \
  -e RADIANCE_PQ_CHECK_MAX_M=${PQ_CHECK_MAX_M:-128} \
  -e RADIANCE_PQ_DECODE_MAX_M=${PQ_DECODE_MAX_M:-64} \
  -e RADIANCE_FAST_DRAFT=1 -e RADIANCE_DRAFT_TAU=0.20 -e RADIANCE_DRAFT_RERANK=80 \
  -e RADIANCE_VERIFY_HEAD=1 -e RADIANCE_VERIFY_HEAD_MAX_M=32 \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS=1 -e RADIANCE_SKINNY_GEMM=1 \
  -e RADIANCE_GDN_PATHS=both \
  -e RADIANCE_KV_GROUP_OPT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -v /home/brian/.cache/huggingface:/root/.cache/huggingface \
  -v /home/brian/models:/models \
  -v /home/brian/.radiance-cache-paro-093:/cache \
  -v /home/brian/deadcode-vllm:/patches:z \
  -v /home/brian/mxfp4_work/paro:/paro:z \
  -v /home/brian/.cache/radiance-libr4d/b9e42ab-rx4:/r4d:z \
  -e R4D_SO=/home/brian/.cache/radiance-libr4d/b9e42ab-rx4 \
  --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    python3 patch_quark_mxfp4.py
    python3 patch_ar_maxbytes.py
    python3 patch_topk_triton_rows.py
    python3 patch_dflash_calib.py
    python3 patch_dflash_mxfp4_kv.py
    python3 patch_rmsquant_fusion.py
    python3 patch_verify_head.py
    python3 patch_kv_group_size.py
    python3 patch_topk_composite.py
    python3 patch_gdn_shared_build.py
    python3 patch_dflash_selector_topk.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    cp radiance_mxfp4.py radiance_gdn.py radiance_rmsquant.py radiance_drafthead.py \
       radiance_verifyhead.py radiance_aroverlap.py radiance_topk.py radiance_arnq.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
      cp /r4d/r4d.so "$SP"/r4d.so
      echo "[radiance] using patched r4d.so from $R4D_SO"
    fi
    # ---- paroquant: build the kernel module and register the quant method in every process ----
    cd /paro
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_paroquant.hip -o "$SP"/radiance_paroquant_kernel.so
    cp radiance_paroquant.py "$SP"/
    # NB: appended to the STDLIB sitecustomize, not written to site-packages -- Ubuntu ships
    # /usr/lib/python3.12/sitecustomize.py and it shadows any site-packages one, so a file
    # dropped there is silently never imported. Each podman run starts from the pristine image,
    # so the append does not accumulate.
    printf "%s\n" \
      "try:" \
      "    import radiance_paroquant  # registers the paroquant quantization config" \
      "except Exception as e:" \
      "    import sys" \
      "    sys.stderr.write(\"[radiance.paroquant] registration failed: %r\\n\" % (e,))" \
      >> /usr/lib/python3.12/sitecustomize.py
    # Leave the bind mounts before exec: a stale .so in the working dir precedes site-packages
    # on sys.path (see run_mxfp4_074 for the 17-hours-stale-kernel incident).
    cd /
    exec /opt/radiance_entrypoint.sh "$@"' \
  _ \
  "$MODEL" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --attention-backend R4D \
  --no-async-scheduling \
  --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
  --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja \
  "${SPEC_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
