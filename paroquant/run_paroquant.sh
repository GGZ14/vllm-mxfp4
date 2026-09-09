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
# MXFP4 checkpoint (quant_method paroquant_mxfp4, from paroquant/build_hybrid.py): same launcher,
#   MODEL_DIR=Qwen3.8-27B-PARO-MXFP4-ft. The rotation streams (defaults on) apply: the loader's
#   per-token producers (pqm_add_rms_rot / pqm_ew_rot) replace the int4 per-group ones through the
#   same install_stream. MODE=eval then gates RADIANCE_PQM_CHECKALL per partition.
#   The RADIANCE_MXFP4_* kernel knobs (fragment-order weights, NT decode loads, decode band) are
#   passed at MXFP4 prod's values; they are inert for the int4 checkpoint (no quark layers load)
#   and load-bearing for MXFP4-PARO's speed.
#
# MODE=eval  (default): --enforce-eager, CHECKALL numerics gate on the four model shapes,
#                       no speculative decoding, 32K ctx. For correctness gating only.
# MODE=prod           : full config -- DFlash2 FP8 drafter (SPEC tokens configurable), 262K ctx,
#                       compiled graphs.
#
# Port 8080 is prod's port and both need both GPUs: stop production first
#   systemctl --user stop qwen_vllm_38        restore with: vllm-switch 38
#
# 2026-09-03 default sampling temperature 1.0 -> 0.7 (--override-generation-config), fleet-wide
#   across every vllm-switch target. DEFAULT only -- a client-supplied temperature still wins.
#   Rollback: sed -i 's/"temperature":0.7/"temperature":1.0/' run_paroquant.sh && systemctl --user restart qwen_vllm_paro
#
set -euo pipefail

# Paths are derived, not hardcoded: this script is the one the systemd unit runs, so it has to
# work from a clone anywhere. PATCHES defaults to the repo root (this file lives in paroquant/).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PATCHES_DIR="$(realpath -m "${PATCHES:-$SCRIPT_DIR/..}")"
MODELS="$(realpath -m "${MODELS:-$HOME/models}")"
HF_CACHE="$(realpath -m "${HF_CACHE:-$HOME/.cache/huggingface}")"

MODE=${MODE:-eval}
PORT=${PORT:-8080}
NAME=${NAME:-vllmparo}
# Space-separated served ids. Eval default answers ONLY to Qwen3.8-PARO so nothing pinned to the
# prod ids routes here by accident; the qwen_vllm_paro systemd unit (vllm-switch paro) overrides
# with the prod ids so clients like the Pi (model id Qwen3.6) work unchanged.
SERVED_NAMES=${SERVED_NAMES:-Qwen3.8-PARO}
SPEC=${SPEC:-5}      # dflash re-sweep 2026-08: 5 beats 7 by 8-13% aggregate on this stack
DRAFTER=${DRAFTER:-Qwen3.8-27B-DFlash2-FP8}   # DFlash2 drafter dir under $MODELS
PREFIX_CACHE=${PREFIX_CACHE:-1}                # 0 for a drafter-capture serve (cached prefixes yield no hidden states)
CAPTURE_DIR=${CAPTURE_DIR:-}                   # host dir: record drafter training data (radiance_dflash_capture.py)
PROFILE=${PROFILE:-0}                          # 1: arm the torch profiler (traces in $CACHE/prof; POST /start_profile, /stop_profile)
# Async scheduling overlaps the engine's scheduling round trip with GPU execution. vLLM refuses it
# together with disable_padded_drafter_batch, so the two are one switch (as in serve-mxfp4.sh). The
# 2026-09-09 drafter profile on this stack showed a 3.1 ms GPU-idle bubble per step between the
# drafter's last kernel and the next step's input prep in sync mode -- the case async exists for.
ASYNC=${ASYNC:-0}
if [ "$ASYNC" = 1 ]; then ASYNC_FLAG="--async-scheduling"; UNPAD=false; else ASYNC_FLAG="--no-async-scheduling"; UNPAD=true; fi
CHUNK=${CHUNK:-8192} # prod prefill chunk (--max-num-batched-tokens); sweep knob
MAXLEN=${MAXLEN:-262144}; MAXSEQS=${MAXSEQS:-8}   # prod context / concurrency; a TP=1 serve on one 32 GB card needs MAXLEN <= 65536
GPU_UTIL=${GPU_UTIL:-0.92}
# GDN decode step as ONE launch (conv -> grid barrier -> recurrent), the libr4d rx5 build that
# also zeroes the cudagraph pad rows. The AutoRound int4 serve (same bf16-input linear contract)
# has run it since 08-30; the merge hook it needs is installed with the merge itself left OFF
# (in_proj_a/b are fp16 here, there is nothing to merge). Compile cache keyed on the flag.
GDN_FUSED=${RADIANCE_GDN_FUSED_UPDATE:-1}
# Pinned to the build the shipped PARO numbers were measured on. serve-mxfp4.sh now derives
# b9e42ab-rx6 from r4d_radiance_extras.patch (rx6 adds the 3-rank all-reduce for TP=3); rx5 is not
# reproducible from the current patch, so a fresh box has to use rx6 and re-gate GSM8K for this
# stack. Override with R4D_KEY=.
R4D_KEY=${R4D_KEY:-b9e42ab-rx5}
R4D_CACHE=${R4D_CACHE:-$HOME/.cache/radiance-libr4d}
# The image's own libr4d predates the gated-delta-net overflow fix and NaNs this model, and the
# in-container copy is guarded by [ -f /r4d/r4d.so ] -- a missing build there is a silent fallback
# to the NaN kernel, not an error. Check it on the host, where it can still be a message.
[ -f "$R4D_CACHE/$R4D_KEY/r4d.so" ] || {
  echo "libr4d $R4D_KEY not built at $R4D_CACHE/$R4D_KEY/r4d.so" >&2
  echo "  serving without it falls back to the image's libr4d, which NaNs this model." >&2
  echo "  Build the current one:  ./setup-paroquant.sh   (produces b9e42ab-rx6)" >&2
  echo "  then either R4D_KEY=b9e42ab-rx6 $0 ... or re-gate and change the default here." >&2
  exit 1
}
# Rotation stream: fused add+rmsnorm+rotate+quant producers for the norm-fed linears (decode
# band); patches the decoder-layer forward, so the compile cache is keyed (-rs).
ROT_STREAM=${RADIANCE_PQ_ROT_STREAM:-1}
# Stream 2: silu-mul -> down, GDN gated norm -> out_proj, attention gate -> o_proj producers
# fused with rotate+quant (default off until the serve gate lands; also patches the graph).
ROT_STREAM2=${RADIANCE_PQ_ROT_STREAM2:-1}
# Stream 3: the two-rank all-reduce fused into the norm+rotate producers (default off until gated).
ROT_STREAM3=${RADIANCE_PQ_ROT_STREAM3:-0}
CACHE_SUF=""; [ "$GDN_FUSED" = 1 ] && CACHE_SUF="-fu"; [ "$ROT_STREAM" = 1 ] && CACHE_SUF="$CACHE_SUF-rs"
[ "$ROT_STREAM2" = 1 ] && CACHE_SUF="${CACHE_SUF}-rs2"
[ "$ROT_STREAM3" = 1 ] && CACHE_SUF="${CACHE_SUF}-rs3"
CACHE=${CACHE:-$HOME/.radiance-cache-paro-093$CACHE_SUF}
mkdir -p "$CACHE"

# MODEL_DIR names a directory under $MODELS. Overridable so the same launcher (same patches,
# same patched libr4d, same template) can serve a pseudo-quantized checkpoint for an accuracy
# gate -- keeping every variable but the weights fixed.
MODEL_DIR=${MODEL_DIR:-Qwen3.8-27B-PARO}
MODEL=/models/$MODEL_DIR
[ -d "$MODELS/$MODEL_DIR" ] || { echo "model missing at $MODELS/$MODEL_DIR; run setup-paroquant.sh" >&2; exit 1; }
MAXLEN_EVAL=${MAXLEN_EVAL:-32768}
# TP and the card set are overridable so a single-card CHECKALL boot can run beside another job.
TP=${TP:-2}
GPUS=${GPUS:-0,1}
# ROCR_VISIBLE_DEVICES selects the physical cards; HIP_VISIBLE_DEVICES then indexes INTO that
# filtered list. Passing the same list to both works for "0,1" only by coincidence and breaks a
# single-card run (GPUS=1 -> HIP asks for index 1 of a one-element list -> "No CUDA GPUs").
HIP_IDX=$(seq -s, 0 $(( $(tr -cd , <<<"$GPUS" | wc -c) )))
# Optional cgroup memory cap for the container (e.g. MEM_LIMIT=14g). Lets a gate boot run beside
# a job that owns most of the host's RAM: the server OOMs itself instead of starving the job.
MEM_LIMIT=${MEM_LIMIT:-}
MEM_ARGS=(); [ -n "$MEM_LIMIT" ] && MEM_ARGS=(--memory "$MEM_LIMIT")
CAPTURE_MOUNT=(); if [ -n "$CAPTURE_DIR" ]; then mkdir -p "$CAPTURE_DIR"; CAPTURE_MOUNT=(-v "$CAPTURE_DIR:/capture:z"); fi

if [ "$MODE" = eval ]; then
  EXTRA_ARGS=(--enforce-eager --max-model-len "$MAXLEN_EVAL" --max-num-seqs 8
              --max-num-batched-tokens 8192)
  # Per-rank quantized shapes: qkv, o, gate_up, down, in_proj(+merge), out_proj
  CHECKALL=${CHECKALL:-"7168:5120,5120:3072,17408:5120,5120:8704,8192:5120,5120:3072"}
  SPEC_ARGS=()
else
  PROF_ARGS=(); [ "$PROFILE" = 1 ] && PROF_ARGS=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/prof --profiler-config.torch_profiler_with_stack=false); [ "$PROFILE" = 1 ] && mkdir -p "$CACHE/prof"
  EXTRA_ARGS=("${PROF_ARGS[@]}" --max-model-len "${MAXLEN:-262144}" --max-num-seqs "${MAXSEQS:-8}" --max-num-batched-tokens "${CHUNK:-8192}"
              $([ "$PREFIX_CACHE" = 1 ] && echo --enable-prefix-caching || echo --no-enable-prefix-caching)
              --compilation-config
              '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true},"compile_sizes":[1,2,4,8],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false,"combo_kernels":true,"benchmark_combo_kernel":true,"triton.cooperative_reductions":true}}')
  CHECKALL=${CHECKALL:-}
  SPEC_ARGS=(--speculative-config
    "{\"method\":\"dflash\",\"model\":\"/models/${DRAFTER}\",\"num_speculative_tokens\":${SPEC},\"attention_backend\":\"TRITON_ATTN\",\"disable_padded_drafter_batch\":${UNPAD},\"draft_sample_method\":\"greedy\"}")
fi

exec podman run --replace --name "$NAME" --privileged --ipc=host --network=host "${MEM_ARGS[@]}" \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES="$GPUS" -e HIP_VISIBLE_DEVICES="$HIP_IDX" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D=1 -e RADIANCE_USE_R4D_AR=1 -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB=86016 \
  -e RADIANCE_STEP_TRACE="${RADIANCE_STEP_TRACE:-0}" \
  -e RADIANCE_DFLASH_CAPTURE_DIR="${CAPTURE_DIR:+/capture}" "${CAPTURE_MOUNT[@]}" \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_FUSE_RMS_QUANT=1 \
  -e R4D_ATTN_FP8=3 \
  -e RADIANCE_GDN_FUSED_UPDATE="$GDN_FUSED" -e RADIANCE_GDN_MERGE_INPROJ=0 \
  -e RADIANCE_GDN_FUSED_MAX_ITEMS="${RADIANCE_GDN_FUSED_MAX_ITEMS:-32}" \
  -e RADIANCE_DYNAMIC_WIDTH=1 -e RADIANCE_DYNW_ALPHA=0.35 -e RADIANCE_DYNW_MARGIN=2 \
  -e RADIANCE_DYNW_MIN=2 -e RADIANCE_DYNW_MIN_BATCH=3 \
  -e RADIANCE_AR_QNB=96 -e RADIANCE_AR_QNT=1024 -e RADIANCE_AR_OVERLAP=0 \
  -e RADIANCE_DFLASH_SELECTOR_TOPK= \
  -e RADIANCE_PAROQUANT=1 \
  -e RADIANCE_MXFP4_W4A8="${RADIANCE_MXFP4_W4A8:-1}" -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-1}" \
  -e RADIANCE_MXFP4_DECODE_NT="${RADIANCE_MXFP4_DECODE_NT:-1}" -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_A_TILED_MIN_M="${RADIANCE_MXFP4_A_TILED_MIN_M:-513}" \
  -e RADIANCE_MXFP4_A_TILED_MIN_M="${RADIANCE_MXFP4_A_TILED_MIN_M:-513}" \
  -e RADIANCE_PQ_CHECKALL="$CHECKALL" \
  -e RADIANCE_PQM_CHECKALL="${RADIANCE_PQM_CHECKALL:-$CHECKALL}" \
  -e RADIANCE_PQM_CHECK_MAX_M="${RADIANCE_PQM_CHECK_MAX_M:-128}" \
  -e RADIANCE_PQ_CHECK_MAX_M=${PQ_CHECK_MAX_M:-128} \
  -e RADIANCE_PQ_DECODE_MAX_M=${PQ_DECODE_MAX_M:-64} \
  -e RADIANCE_PQ_WPERM="${RADIANCE_PQ_WPERM:-1}" -e RADIANCE_PQ_DECODE_NT="${RADIANCE_PQ_DECODE_NT:-1}" \
  -e RADIANCE_PQ_ATILED="${RADIANCE_PQ_ATILED:-1}" -e RADIANCE_PQ_AT_LBK="${RADIANCE_PQ_AT_LBK:-128}" \
  -e RADIANCE_PQ_AT_HOIST="${RADIANCE_PQ_AT_HOIST:-1}" -e RADIANCE_PQ_PTOK="${RADIANCE_PQ_PTOK:-1}" \
  -e RADIANCE_PQ_ROT_STREAM="$ROT_STREAM" -e RADIANCE_PQ_ROT_STREAM2="$ROT_STREAM2" \
  -e RADIANCE_PQ_ROT_STREAM3="$ROT_STREAM3" -e RADIANCE_PQ_AR_CHECK="${RADIANCE_PQ_AR_CHECK:-0}" \
  -e RADIANCE_PQ_AR_FALLBACK="${RADIANCE_PQ_AR_FALLBACK:-0}" \
  -e RADIANCE_PQ_ROT_V2="${RADIANCE_PQ_ROT_V2:-1}" \
  -e RADIANCE_FAST_DRAFT=1 -e RADIANCE_DRAFT_TAU=0.20 -e RADIANCE_DRAFT_RERANK=80 \
  -e RADIANCE_VERIFY_HEAD=1 -e RADIANCE_VERIFY_HEAD_MAX_M=32 \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS=1 -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_GDN_PATHS=both \
  -e RADIANCE_KV_GROUP_OPT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$MODELS":/models \
  -v "$CACHE":/cache \
  -v "$PATCHES_DIR":/patches:z \
  -v "$SCRIPT_DIR":/paro:z \
  -v "$R4D_CACHE/$R4D_KEY":/r4d:z \
  -e R4D_SO="$R4D_CACHE/$R4D_KEY" \
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
    python3 patch_async_dynwidth.py     # dynamic verify width under async (AsyncScheduler bypasses update_draft_token_ids)
    python3 patch_step_trace.py         # RADIANCE_STEP_TRACE=N per-step CPU/GPU trace
    python3 patch_skinny_gemm.py        # small bf16 projections -> radiance_gemm (RADIANCE_SKINNY_GEMM=all adds in_proj_ba)
    python3 patch_dflash_selector_topk.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    python3 patch_gdn_merge_inproj.py
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    cp radiance_mxfp4.py radiance_gemm.py radiance_gdn.py radiance_gdnmerge.py radiance_rmsquant.py \
       radiance_drafthead.py radiance_verifyhead.py radiance_aroverlap.py radiance_topk.py \
       radiance_arnq.py "$SP"/
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
    cp radiance_paroquant.py radiance_paroquant_mxfp4.py "$SP"/
    cp /patches/radiance_dflash_capture.py "$SP"/ 2>/dev/null || cp ../radiance_dflash_capture.py "$SP"/
    # NB: appended to the STDLIB sitecustomize, not written to site-packages -- Ubuntu ships
    # /usr/lib/python3.12/sitecustomize.py and it shadows any site-packages one, so a file
    # dropped there is silently never imported. Each podman run starts from the pristine image,
    # so the append does not accumulate.
    printf "%s\n" \
      "try:" \
      "    import radiance_paroquant  # registers the paroquant quantization config" \
      "    import radiance_paroquant_mxfp4  # and the MXFP4-weights variant (paroquant_mxfp4)" \
      "    import radiance_dflash_capture  # drafter training-data capture (inert unless RADIANCE_DFLASH_CAPTURE_DIR)" \
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
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --attention-backend R4D \
  $ASYNC_FLAG \
  --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
  --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja \
  "${SPEC_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
