#!/bin/bash
# EVALUATION (not production): native MXFP4 body on gfx1201 with the MTP drafter in FP8,
# on radiance 0.7.4 (libr4d).
#
# The 0.5.8 form of this run is run_mxfp4_minm.sh, kept as-is because it is the only way to
# reproduce the baseline these numbers are measured against:
#   prefill 3602 / 3409 / 2283 / 1730 / 1390 tok/s at 7.8k/26k/104k/182k/259k
#   real decode 55.4 / 61.7 tok/s | weights 9.24 GiB/GPU | KV ~856k tok | WikiText-2 PPL 8.3335
#
# Checkpoint built by ~/mxfp4_work/fp8_mtp.py from amd/Qwen3.8-27B-Quark-AWQ-MXFP4. The drafter is
# FP8, not MXFP4, and that is a settled result: MXFP4 RTN cost acceptance 2.5 -> 2.21 and AWQ did
# not rescue it (MXFP4's per-32 E8M0 block exponent already does most of what per-channel scaling
# would), while FP8 e4m3 per-channel holds acceptance at 2.60-2.80. Do not point this at
# ~/models/Qwen3.8-27B-MXFP4-mtpq or -mtpawq; those exist only for that comparison.
#
# WHAT IS DIFFERENT FROM THE 0.5.8 RUN
#   - image 0.5.8 -> 0.7.4; cache .radiance-cache-w4a8-058 -> -074. Cache dirs validate on model +
#     torch/Triton version and MUST NOT be shared across configurations.
#   - patch_quark_mxfp4.py is a different patch. vLLM 0.27 replaced QuarkOCP_MX's inline dispatch
#     with a kernel plugin ABC, so instead of seven string hunks against one file it now registers
#     RadianceMxfp4W4A8LinearKernel into _POSSIBLE_MXFP4_KERNELS and relaxes two aiter gates.
#   - RADIANCE_MXFP4_MAX_M is GONE. It used to hand large batches back to emulation; that path
#     could never run here anyway (quark's TileLang backend dies with "libamdhip64.so not found"
#     inside the worker, and the branch got specialised into the compile graph during the M=8192
#     profile run, killing startup), and with W4A8 on, large M belongs to the fp8-WMMA kernel.
#   - RADIANCE_ATTN_TUNE is gone upstream (the AITER tune is unconditional now).
#     RADIANCE_FAST_REDUCE -> RADIANCE_USE_R4D_AR, RADIANCE_AR_QUANT -> RADIANCE_USE_R4D_AR_QUANT,
#     both under the master RADIANCE_USE_R4D.
#   - RADIANCE_AR_MAX_KB is restored by patch_ar_maxbytes.py. Upstream hardcoded it at 48 MB,
#     sized for its 4096-token chunk; at CHUNK=8192 the message is 80 MiB and every prefill
#     reduction would silently fall back to RCCL. See that patch's docstring for the measurement.
#   - --kv-cache-memory is NOT set. The 0.5.8 value (19105177314) was read off a 0.5.8 startup log
#     and 0.7.4 has a different non-torch footprint (r4d scratch, a differently sized AR buffer).
#     Re-derive it from vLLM's own "fit into requested memory" line in THIS build's log, taking the
#     SMALLER of the two ranks -- TP0 carries more non-torch memory, and one value applies to both,
#     so using TP1's number OOMs TP0.
#   - the chat template is deliberately unchanged for the parity run. Upstream re-derived
#     qwen3.8-enhanced.jinja against the released official template (it was rendering booleans as
#     True/False via `| string`); adopting that is a separate change, and doing it here would
#     confound the benchmark.
#
# STAGING. Land the enhancements one at a time -- a combined A/B cannot attribute a regression.
#   (default)        R4D GDN + WHT6 all-reduce (RADIANCE_USE_R4D=1), AITER attention   <- parity gate
#   R4D_ATTN=1       + the R4D paged attention backend -- ON BY DEFAULT, measured +37.8% prefill
#                      at 260k, +30.9% at 182k, +20.9% at 104k against the 0.5.8 baseline
#   FAST_DRAFT       int2 MTP draft head with an exact bf16 rerank. ON BY DEFAULT, worth
#                      +6.5% decode short / +0.1% medium against the 0.5.8 baseline, and the
#                      difference between 58.5 and 67.1 tok/s on this build.
#
#                      It HUNG a worker at chunk 8192 before the libr4d GDN overflows were fixed:
#                      the draft head was being fed NaN like everything else downstream of the
#                      gated-delta-net. Fixing the kernel at source fixed the hang too, so it now
#                      runs at the full chunk size. Historical note: it arms correctly
#                      (this checkpoint's lm_head is bf16 and excluded from quantization, so the
#                      exact-rerank guarantee holds), but a long-prompt sweep at chunk 8192 hung a
#                      worker and killed the engine with an RPC TimeoutError in sample_tokens.
#                      Upstream ships it opt-in at chunk 4096. Retry there before trusting it.
#                      Original note: it quantizes the drafter's
#                      lm_head and reranks against an untouched bf16 copy, but this checkpoint's
#                      drafter is FP8 and vLLM may be sharing the target's head, in which case
#                      that bf16 copy -- and the exactness guarantee -- is not what it assumes.
#
# NUMERICS REFERENCE (~/pibench-local/results/ppl/, WikiText-2, --chunks 300 --chars 3000,
# 208,539 tokens). Reproduce with GPU_UTIL=0.75 and `python3 ~/pibench-local/ppl.py --model
# Qwen3.8-MXFP4 --tag <tag>`:
#     8.3317  MXFP4 W4A8, exact bf16 all-reduce
#     8.3335  MXFP4 W4A8, fp8 all-reduce   <- what 0.5.8 shipped
#     8.3386  MXFP4 W4A4 (no W4A8), fp8 all-reduce
# The 6-bit rotated payload replaces the fp8 one and is claimed slightly more accurate, so a
# healthy 0.7.4 lands at or just under 8.3335. A jump well past it means the rewritten weight-prep
# path is wrong, not that the all-reduce changed -- the whole AR spread is only 0.02%.
#
# Port 8080 is prod's and this needs both GPUs, so stop production first:
#   systemctl --user stop qwen_vllm_38          restore with: vllm-switch 38
#
# WHAT TO CHECK IN THE LOG
#   "Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"  -> our kernel won the selection
#   "[radiance] native MXFP4 enabled on gfx12x"           -> the aiter fp4 gate was relaxed
#   the R4D selections table (RADIANCE_R4D_REPORT=1)      -> which kernels bound, and why not
#   the stock "current platform does not support native MXFP4/MXFP6" notice still prints and is a
#   false alarm; it comes from a separate supports_mx() call.

set -euo pipefail

IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.7.4}
NAME=${NAME:-vllmmxfp4074}
PORT=${PORT:-8080}
CHUNK=${CHUNK:-8192}
R4D_ATTN=${R4D_ATTN:-1}
FAST_DRAFT=${FAST_DRAFT:-1}
CACHE=${CACHE:-$HOME/.radiance-cache-w4a8-074}
# prompt_logprobs allocates a ~1-1.7 GiB prompt x vocab logits transient that vLLM does not reserve
# for, and KV is sized to eat everything else -- 0.97 and even 0.92 OOM the engine on ppl.py. Use
# GPU_UTIL=0.75 for perplexity work, 0.98 for throughput.
# 0.98 is the ceiling on this box, not a guess: the card has 32624 MiB, and vLLM measures free
# memory AFTER its own HIP context and torch init exist, so it sees 31980 MiB. 0.99 asks for
# 31.54 GiB and fails at startup. 0.98 gives 857,399 KV tokens against 840,019 at 0.97 and
# survives a full 260k-prefill sweep with no OOM.
GPU_UTIL=${GPU_UTIL:-0.98}
# MTP speculative depth. Measured on this build, 4 beats 8 at decode -- 59.8/60.2 tok/s against
# 53.1/58.6, because acceptance falls (42.1% -> 33.7%) faster than the deeper drafts pay for
# themselves. Prefill is unaffected within run-to-run noise. The 0.5.8 baseline also ran 4, so
# this keeps the comparison honest as well as fast.
SPEC=${SPEC:-4}
# Context length. Only lower it for diagnostics -- the FLA GDN fallback allocates against this,
# not against the chunk size, and OOMs at 262144.
MAXLEN=${MAXLEN:-262144}
# Patched libr4d (three GDN exponent-overflow guards). Stock 0.7.4 NaNs the gated-delta-net
# output on this model; see README. Set R4D_SO= to fall back to the image's stock r4d.so.
# Point this at a libr4d checkout carrying libr4d-gdn-overflow-guards.patch, built with
# ./build.sh (its r4d.so is copied over the image's at container start). STRONGLY RECOMMENDED:
# stock 0.7.4 NaNs the gated-delta-net output on this model -- WikiText-2 PPL 653586 vs 8.3706.
# Leave empty to use the image's stock r4d.so, in which case set RADIANCE_MXFP4_SANITIZE=1.
R4D_SO=${R4D_SO:-}
# Batch size above which the W4A8 fp8-WMMA kernel takes over from aiter's W4A4 Triton path.
# DEFAULT 0 = never fall back; our kernel serves every M.
#
# This is a correctness requirement, not a tuning choice. aiter's W4A4 path returns a WRONG result
# for N=5120 K=3072 (o_proj): captured from a live serve and replayed against an fp32 reference,
# aiter lands at rel=1.066 with ~1/35th of the correct magnitude, while our kernel is at rel=0.0017.
# That shape is the one with no tuned table in mxfp4-configs/, so it takes aiter's generic bands.
# With MIN_M=16 it went unnoticed in prefill (M=17, our kernel) and poisoned decode (M=9, aiter),
# which is exactly the fluent-looking garbage this build shipped with for an afternoon.
#
# Note the comparison is `x.shape[0] > MIN_M`, so MIN_M=1 still routes M=1 to aiter. Use 0.
# Set it absurdly high to route everything to aiter -- only useful for bisecting.
MIN_M=${MIN_M:-16}

# All 304 linear layers run on the W4A8 kernel. RADIANCE_MXFP4_KERNEL_NK / _PERBLOCK_NK remain as
# shape-level bisect tools (N:K pairs) but are unset by default.
#
# They existed because the 64 layers at N=5120 K=3072 (gdn out_proj, attention o_proj) produced a
# broken model, which turned out NOT to be a kernel bug: those layers legitimately receive NaN in
# their activations -- one whole gated-delta-net head -- and per-token fp8 quantization turns a
# single NaN into a NaN row scale, poisoning the row. aiter tolerated the same input only because
# mxfp4 quantization squashes NaN to a finite code. RADIANCE_MXFP4_SANITIZE (default 1) fixes it.
# Extra vllm serve args, for bisecting (e.g. EXTRA="--enforce-eager").
EXTRA=${EXTRA:-}

SNAP="${SNAP:-$HOME/models/Qwen3.8-27B-MXFP4-mtpfp8}"
[ -f "$SNAP/config.json" ] || { echo "no checkpoint at $SNAP" >&2; exit 1; }
# HF_HUB_OFFLINE=1 inside the container and the cache mounts at /root/.cache/huggingface, so vllm
# must be handed the CONTAINER path -- a host path fails HF repo-id validation, not "not found".
CSNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8

# The AR size gate compares the raw bf16 byte count: CHUNK x hidden(5120) x 2. Derive it rather
# than hardcoding it, so changing CHUNK cannot silently drop prefill back onto RCCL.
AR_MAX_KB=$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))

if [ "$R4D_ATTN" = "1" ]; then ATTN=R4D; else ATTN=ROCM_AITER_UNIFIED_ATTN; fi

mkdir -p "$CACHE"/{vllm,inductor,triton,aiter}

echo "[run] image=$IMAGE attn=$ATTN chunk=$CHUNK ar_max_kb=$AR_MAX_KB fast_draft=$FAST_DRAFT min_m=$MIN_M fuse_rms=${RADIANCE_FUSE_RMS_QUANT:-1} preshuf=${RADIANCE_PRESHUFFLE:-1} util=$GPU_UTIL cache=$CACHE"

exec podman run --replace --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e ROCR_VISIBLE_DEVICES=0,1 -e HIP_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D="${RADIANCE_USE_R4D:-1}" -e RADIANCE_USE_R4D_AR="${RADIANCE_USE_R4D_AR:-1}" -e RADIANCE_USE_R4D_AR_QUANT="${RADIANCE_USE_R4D_AR_QUANT:-1}" \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB="$AR_MAX_KB" \
  -e RADIANCE_PRESHUFFLE="${RADIANCE_PRESHUFFLE:-1}" -e RADIANCE_FUSE_RMS_QUANT="${RADIANCE_FUSE_RMS_QUANT:-1}" \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M="$MIN_M" \
  -e RADIANCE_FAST_DRAFT="$FAST_DRAFT" -e RADIANCE_DRAFT_TAU=0.20 \
  -e RADIANCE_MXFP4_DEBUG="${RADIANCE_MXFP4_DEBUG:-0}" \
  -e RADIANCE_MXFP4_PUREQUANT="${RADIANCE_MXFP4_PUREQUANT:-0}" \
  -e RADIANCE_MXFP4_SYNC="${RADIANCE_MXFP4_SYNC:-0}" \
  -e RADIANCE_MXFP4_CLONE="${RADIANCE_MXFP4_CLONE:-0}" -e RADIANCE_MXFP4_CHECKX="${RADIANCE_MXFP4_CHECKX:-0}" \
  -e RADIANCE_MXFP4_PADOUT="${RADIANCE_MXFP4_PADOUT:-0}" \
  -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_SHADOW="${RADIANCE_MXFP4_SHADOW:-}" \
  -e RADIANCE_MXFP4_SANITIZE="${RADIANCE_MXFP4_SANITIZE:-0}" \
  -e RADIANCE_GDN_PATHS="${RADIANCE_GDN_PATHS:-both}" \
  -e RADIANCE_GDN_NANTRACE="${RADIANCE_GDN_NANTRACE:-0}" \
  -e RADIANCE_MXFP4_KERNEL_N="${RADIANCE_MXFP4_KERNEL_N:-}" \
  -e RADIANCE_MXFP4_KERNEL_NK="${RADIANCE_MXFP4_KERNEL_NK:-}" \
  -e RADIANCE_MXFP4_CHECKALL="${RADIANCE_MXFP4_CHECKALL:-}" \
  -e RADIANCE_MXFP4_PERBLOCK_NK="${RADIANCE_MXFP4_PERBLOCK_NK:-}" \
  -e RADIANCE_MXFP4_REFLINEAR="${RADIANCE_MXFP4_REFLINEAR:-0}" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1 \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "${MODELS:-$HOME/models}":/models \
  -v "$CACHE":/cache \
  -v "${PATCHES:-$(cd "$(dirname "$0")" \&\& pwd)}":/patches:z \
  ${R4D_SO:+-v "$R4D_SO":/r4d:z} \
  ${R4D_SO:+-e R4D_SO="$R4D_SO"} \
  --entrypoint bash \
  "$IMAGE" -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    python3 patch_quark_mxfp4.py
    python3 patch_ar_maxbytes.py
    # Non-fatal: fixes content=null on thinking-off requests; not required to serve.
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    cp radiance_mxfp4.py radiance_gdn.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    # Optional patched libr4d. R4D_SO points at an r4d.so built from a local libr4d checkout;
    # the Dockerfile supports the same substitution through R4D_REPO / R4D_VERSION.
    if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
      cp /r4d/r4d.so "$SP"/r4d.so
      echo "[radiance] using patched r4d.so from $R4D_SO"
    fi
    # Leave /patches before exec. It is a bind mount of the repo, and a stale
    # radiance_mxfp4_fp8.so left there by a `make` shadows the one just compiled into
    # site-packages, because the working directory precedes it on sys.path. That is not a
    # hypothetical: an Aug-20 build sat there and silently served a kernel 17 hours older than
    # its own source, producing fluent-looking garbage with no error anywhere in the log.
    cd /
    exec /opt/radiance_entrypoint.sh "$@"' _ \
    "$CSNAP" --served-model-name Qwen3.8 Qwen3.6 Qwen3.8-MXFP4 --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8 --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAXLEN" --max-num-seqs 8 --max-num-batched-tokens "$CHUNK" \
    --attention-backend "$ATTN" \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$ATTN\",\"disable_padded_drafter_batch\":true}" \
    --no-async-scheduling $EXTRA \
    --enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja
