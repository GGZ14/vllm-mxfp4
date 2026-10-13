#!/bin/bash
# Serve a Qwen3.5/3.6-35B-A3B-class MXFP4 MoE checkpoint on ONE gfx1201 card: plain `vllm serve` on the
# Radiance image plus the gfx1201 MoE fixes. serve-mxfp4.sh is built around Qwen3.8-27B (R4D attention
# needs GQA 6, the DFlash2 drafter, measured KV pins); an A3B MoE (GQA 8, 256 experts, its own MTP
# head) runs here instead. Details and measurements: MOE-GFX1201.md.
#
#   SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 ./serve-moe-mxfp4.sh [extra vllm serve args]
#
# Knobs (all ${VAR:-default}):
#   SNAP              checkpoint directory (required). Measured: amd/Qwen3.5-35B-A3B-MXFP4 (Quark W4A4,
#                     experts only). A Hugging Face cache snapshot path works (its hub dir is mounted).
#   MOE_FIXES=1       1: experts weight-only on the AITER a16w4 lane (patch_quark_moe_w4a16.py,
#                     patch_gfx12_aiter_a16w4.py) + split-KV verify attention (patch_attn_3d_multiq.py).
#                     0: stock vLLM, i.e. the EMULATION MoE backend on gfx1201 (~5x slower decode).
#   MOE_PREFILL_ATTN=r4d  prefill attention with MOE_FIXES=1 (patch_moe_prefill_attn.py):
#                     r4d: R4D backend for the model and the MTP drafter; libr4d's prefill kernel built at
#                          GQA 8 (moe-r4d/) serves runs >= 17 tokens, stock Triton split-KV the rest
#                          (decode, verify). ~6x lower TTFT at 16k. Needs a libr4d checkout (below).
#                     off: TRITON_ATTN for everything, as before.
#   RADIANCE_GDN_SCAN_FIX=1  1: exact GDN chunk scan (moe-gdn2/, patch_gdn_scan_fix.py) in place of libr4d
#                     v0.5.0's, which is wrong for any (sequence, head) whose in-chunk decay span exceeds
#                     160 (libr4d #4; about 7% of GDN heads on real prompts). Same prefill speed. Built at
#                     container start from a patch against the libr4d checkout (R4D_SRC), so it needs that
#                     checkout with or without MOE_PREFILL_ATTN=r4d, and applies with or without MOE_FIXES.
#                     0: libr4d's scan as is.
#   RADIANCE_MOE_W4A8=1  1: expert GEMM calls of >= RADIANCE_MOE_W4A8_MIN_TOKENS tokens (default 1025: prefill
#                     chunks) run on an fp8-WMMA W4A8 grouped kernel (moe-w4a8/, patch_moe_w4a8.py); decode,
#                     MTP verify and CUDA-graph calls stay on a16w4. ~+25% prefill. Rides on the a16w4 lane,
#                     so it needs MOE_FIXES=1 and is ignored with MOE_FIXES=0. 0: a16w4 only.
#   RADIANCE_MOE_DRAFT_HEAD=int2  off | fp8 | int4 | int2: the MTP drafter's draft passes score against a compressed copy
#                     of lm_head (patch_moe_drafthead.py, moe-drafthead/); the verify pass keeps the bf16 head, so
#                     accepted tokens are unchanged. int2 = the image's radiance_drafthead (int2 g128 + exact top-32
#                     rerank), 0.13 GiB: single-stream decode +19% on identical outputs, KV -0.20 GiB. Applies only with
#                     SPEC > 0 (there is no draft head otherwise). off = the stock bf16 head. Do not also set
#                     RADIANCE_FAST_DRAFT=1: the image's own hook would re-arm the head last.
#   RADIANCE_MOE_DENSE_FP8=1  1: the target's bf16 dense linears (GDN in_proj_qkvz/out_proj, attention qkv/o, shared
#                     expert gate_up/down; 2.62 -> 1.31 GiB) become fp8 e4m3 with a scale per output channel, the bf16
#                     freed before KV sizing (+1.06 GiB KV). Decode weight-only (HIP skinny / Triton), prefill chunks
#                     > 80 tokens W8A8. in_proj_ba, router, shared_expert_gate, lm_head and the drafter stay bf16
#                     (patch_moe_densefp8.py, moe-densefp8/). Needs MOE_FIXES=1. 0 = bf16 dense layers.
#   RADIANCE_MOE_GATE_FIX=1  1: sigmoid(shared_expert_gate(x)) * out as one fused kernel instead of hipBLASLt GEMV +
#                     sigmoid + mul (same bf16 roundings, only the dot's summation order differs). Needs MOE_FIXES=1.
#                     0 = the three stock ops.
#   RADIANCE_MOE_PAD_ROUTE=1  1: vLLM pads every CUDA-graph replay to a multiple of 1 + SPEC rows, and the padded rows' stale
#                     hidden states route to 8 experts of their own each, so the expert GEMMs read the weights of up to 8 extra
#                     experts per padded row and layer (target verify and every draft pass). patch_moe_padroute.py
#                     (moe-padroute/) copies row 0's router logits into the padded rows right after the router GEMV, so they
#                     reuse row 0's experts; real rows' logits are never written, outputs flip only at near-ties, like a
#                     restart. Needs MOE_FIXES=1, ignored with MOE_FIXES=0. No compile-cache change. 0 = stock routing.
#   RADIANCE_MOE_DRAFT_GRAPH=1  1: each MTP drafter loop pass (1 token per request) replays ONE captured graph per batch size
#                     instead of two graph pieces plus an eager attention segment re-entered from Python. seq_lens is staged
#                     into a fixed buffer, and every replay compares the live attention metadata's addresses with the
#                     captured ones and falls back to the piecewise path on any difference (a capture error turns the knob
#                     off for the process). 0 = piecewise.
#   RADIANCE_MOE_DRAFT_OVERLAP=1  1: the image's dynamic-draft gate drains the GPU with a blocking D2H after every draft pass;
#                     the copy goes to pinned memory without blocking and the gate decision is resolved after the next
#                     pass's input preparation, right before its forward (the same passes run). Use both: drafter host
#                     stalls 1.56 -> 0.48 ms per step, +3.6% single-stream together, +2.1% / +0.8% alone. Nothing is left
#                     to overlap under RADIANCE_MOE_ASYNC=1, which turns the dynamic draft off. Both knobs need MOE_FIXES=1
#                     and SPEC > 0 (patch_moe_draftloop.py, moe-draftloop/, applied after pad-route; no compile-cache
#                     change). 0 = the stock loop.
#   RADIANCE_MOE_ASYNC=0  (off by default: run in production, but this launcher with it on is dry-run checked only;
#                     turn on together with RADIANCE_MOE_DRAFT_WARM=16) 1: vLLM async scheduling. Step N+1's scheduling and input preparation run while step N's sampler and
#                     MTP drafter are on the GPU, which hides ~2.2 ms per step of GPU-idle host glue, and every verify batch
#                     replays the FULL graph. vLLM refuses --async-scheduling together with disable_padded_drafter_batch,
#                     so this also turns the padded drafter batch on, and patch_moe_async.py turns the image's dynamic draft
#                     (RADIANCE_DYNAMIC_DRAFT) off: its ragged drafts and per-pass host gate do not fit the GPU-resident
#                     draft path, so MTP drafts a fixed SPEC tokens every step. Fixed-prompt decode +12.1%, single stream
#                     +8%, 8 / 12 / 16 streams +8 / +3 / +2%. Applies only with SPEC > 0. The first start on a compile cache
#                     that never held the padded drafter graph compiles it and sizes KV ~0.85 GiB smaller; restart once.
#                     0 = sync scheduling, the unpadded drafter and the image's dynamic draft (the behavior before this knob).
#   RADIANCE_MOE_DRAFT_WARM=0  N, 0..64 (16 with async): once the server answers, a background client in the container sends k concurrent
#                     24-token requests for k = 1..N (warm_draftloop.py), so the drafter-loop graph of every batch size is
#                     captured before users arrive. N=16: 6.7 s after ready, ~0.55 GiB allocated earlier (the memory lazy
#                     capture takes after the first bursts; KV unchanged), first-burst TTFT at 8 / 12 / 16 streams
#                     0.79 / 0.84 / 0.88 -> 0.42 / 0.68 / 0.67 s. Needs RADIANCE_MOE_DRAFT_GRAPH=1 and SPEC > 0, and is
#                     ignored without them. 0 = capture lazily, inside the first burst at each new concurrency.
#   RADIANCE_ADAPTIVE_CHUNK=0  0 | N, an integer above 4096 (4320 recommended): adaptive prefill chunk budget
#                     (patch_adaptive_chunk.py, moe-adaptivechunk/). Align-mode attention blocks are 2,160 tokens and vLLM
#                     floors every non-final prefill chunk to whole blocks, so the 4096 budget prefills a long prompt in
#                     2,160-token steps. N starts vLLM with --max-num-batched-tokens N (it replaces CHUNK, which must stay
#                     4096; own compile cache, suffix -acN), and the patch gives a step the full N only when one request is
#                     alone in the scheduler with more than 4096 tokens left to prefill; every other step keeps the 4096
#                     budget (today's chunks, room for arrivals and decodes). 4320 = 2 blocks: idle prefill +5.2% / +5.1%
#                     at 16.7k / 33.9k tokens, flat at 4k. Off by default: it was validated only together with
#                     RADIANCE_MOE_ASYNC=1 (in production since 2026-10-02, not through this launcher). Turn on with
#                     RADIANCE_MOE_ASYNC=1 RADIANCE_MOE_DRAFT_WARM=16 RADIANCE_ADAPTIVE_CHUNK=4320. 0 = the plain CHUNK
#                     budget, no patch, no variable.
#   RADIANCE_ADAPTIVE_CHUNK_SYNC=1  0 | 1: under async scheduling, do not queue a big solo prefill step ahead of the one in
#                     flight, so an arriving request waits for at most one big step before its own (without it the
#                     arrival sweep averages 0.80 s against 0.53 s). The hook sits in async scheduling's batch queue, so it
#                     does nothing without RADIANCE_MOE_ASYNC=1; it is passed on only when RADIANCE_ADAPTIVE_CHUNK is on.
#   R4D_SRC=~/.radiance-libr4d-<R4D_VERSION>  libr4d checkout for r4d and the GDN scan fix; cloned from
#                     R4D_REPO at R4D_VERSION (v0.5.0, the tag the image's r4d.so is built from) when missing
#   RADIANCE_HW_QUEUES=1  GPU_MAX_HW_QUEUES for the container (see serve-mxfp4.sh); 0 = HIP default
#   SPEC=4            MTP draft tokens; 0 disables speculation. In align mode each request pins 2 + SPEC GDN
#                     state blocks, so SPEC sets how many requests fit: SPEC=4 fits 15 short ones at
#                     MAXSEQS=16 and GPU_UTIL=0.97, SPEC=8 fits 7. Sections 1-3 of MOE-GFX1201.md were
#                     measured at SPEC=8, MAXSEQS=8, GPU_UTIL=0.95.
#   MAXLEN=65536  MAXSEQS=16  CHUNK=4096  GPU_UTIL=0.97  PORT=8080  NAME=radiance-moe  GPUS=0
#   IMAGE=stilldeadcode/vllm-radiance:0.9.3   RUNTIME=podman|docker (auto)
#   CACHE=~/.radiance-cache-moe-<model>-f<fixes>[-pa<mode>][-gdn2][-w4a8][-dfp8][-gate][-ac<N>]   compile cache; never share one across knobs
#                     (dense fp8 swaps the linears' apply, which vLLM's compile-cache key does not see: a bf16 cache would
#                     replay the bf16 graph on the fp8 weights and die at the profile run. The gate knob is a runtime flag
#                     in a source file that the patch edits either way, so it gets a suffix too rather than trusting the key;
#                     the compile range follows --max-num-batched-tokens, which RADIANCE_ADAPTIVE_CHUNK changes)
#   SERVED_NAMES=<basename of the model>   DRY_RUN=1 print the command   DETACH=1 run in background
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
die() { echo "[serve-moe] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

SNAP=${SNAP:-}
[ -n "$SNAP" ] || die "set SNAP to the checkpoint directory" "e.g. SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 $0"
[ -f "$SNAP/config.json" ] || die "no config.json in $SNAP"
SNAP=$(cd "$SNAP" && pwd)
KIND=$(python3 - "$SNAP/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c)
moe = any((t.get(k) or 0) > 0 for k in ("num_experts", "n_routed_experts", "num_local_experts"))
print("moe" if moe or str(t.get("model_type", "")).endswith("_moe") else "dense")
PY
)
[ "$KIND" = moe ] || echo "[serve-moe] WARNING: $SNAP does not look like an MoE checkpoint; serve-mxfp4.sh is the dense launcher" >&2

MOE_FIXES=${MOE_FIXES:-1}; SPEC=${SPEC:-4}
MAXLEN=${MAXLEN:-65536}; MAXSEQS=${MAXSEQS:-16}; CHUNK=${CHUNK:-4096}; GPU_UTIL=${GPU_UTIL:-0.97}
PORT=${PORT:-8080}; NAME=${NAME:-radiance-moe}; GPUS=${GPUS:-0}
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
PA=${MOE_PREFILL_ATTN:-r4d}; HWQ=${RADIANCE_HW_QUEUES:-1}
GDNFIX=${RADIANCE_GDN_SCAN_FIX:-1}; W4A8=${RADIANCE_MOE_W4A8:-1}; W4A8_MIN=${RADIANCE_MOE_W4A8_MIN_TOKENS:-}
case "$PA" in off|r4d) ;; *) die "MOE_PREFILL_ATTN must be r4d or off (got $PA)" ;; esac
case "$GDNFIX" in 0|1) ;; *) die "RADIANCE_GDN_SCAN_FIX must be 0 or 1 (got $GDNFIX)" ;; esac
case "$W4A8" in 0|1) ;; *) die "RADIANCE_MOE_W4A8 must be 0 or 1 (got $W4A8)" ;; esac
DH=${RADIANCE_MOE_DRAFT_HEAD:-int2}; DFP8=${RADIANCE_MOE_DENSE_FP8:-1}; GF=${RADIANCE_MOE_GATE_FIX:-1}
PR=${RADIANCE_MOE_PAD_ROUTE:-1}; DG=${RADIANCE_MOE_DRAFT_GRAPH:-1}; DO=${RADIANCE_MOE_DRAFT_OVERLAP:-1}
AS=${RADIANCE_MOE_ASYNC:-0}; DW=${RADIANCE_MOE_DRAFT_WARM:-0}
ACH=${RADIANCE_ADAPTIVE_CHUNK:-0}; ACS=${RADIANCE_ADAPTIVE_CHUNK_SYNC:-1}
case "$DH" in off|fp8|int4|int2) ;; *) die "RADIANCE_MOE_DRAFT_HEAD must be off, fp8, int4 or int2 (got $DH)" ;; esac
case "$DFP8" in 0|1) ;; *) die "RADIANCE_MOE_DENSE_FP8 must be 0 or 1 (got $DFP8)" ;; esac
case "$GF" in 0|1) ;; *) die "RADIANCE_MOE_GATE_FIX must be 0 or 1 (got $GF)" ;; esac
case "$PR" in 0|1) ;; *) die "RADIANCE_MOE_PAD_ROUTE must be 0 or 1 (got $PR)" ;; esac
case "$DG" in 0|1) ;; *) die "RADIANCE_MOE_DRAFT_GRAPH must be 0 or 1 (got $DG)" ;; esac
case "$DO" in 0|1) ;; *) die "RADIANCE_MOE_DRAFT_OVERLAP must be 0 or 1 (got $DO)" ;; esac
case "$AS" in 0|1) ;; *) die "RADIANCE_MOE_ASYNC must be 0 or 1 (got $AS)" ;; esac
case "$DW" in ''|*[!0-9]*) die "RADIANCE_MOE_DRAFT_WARM must be an integer from 0 to 64 (got $DW)" ;; esac
[ "$DW" -le 64 ] || die "RADIANCE_MOE_DRAFT_WARM must be an integer from 0 to 64 (got $DW)"
case "$ACH" in 0) ;; ''|*[!0-9]*) die "RADIANCE_ADAPTIVE_CHUNK must be 0 or a token budget above 4096 (got $ACH)" ;;
  *) [ "$ACH" -gt 4096 ] || die "RADIANCE_ADAPTIVE_CHUNK must be 0 or a token budget above 4096 (got $ACH)" ;; esac
case "$ACS" in 0|1) ;; *) die "RADIANCE_ADAPTIVE_CHUNK_SYNC must be 0 or 1 (got $ACS)" ;; esac
[ "$ACH" = 0 ] || [ "$CHUNK" = 4096 ] || die "RADIANCE_ADAPTIVE_CHUNK sets --max-num-batched-tokens itself; leave CHUNK at 4096 (got $CHUNK)"
case "$W4A8_MIN" in ''|*[!0-9]*) [ -z "$W4A8_MIN" ] || die "RADIANCE_MOE_W4A8_MIN_TOKENS must be an integer (got $W4A8_MIN)" ;; esac
case "$SPEC" in ''|*[!0-9]*) die "SPEC must be a non-negative integer (got $SPEC)" ;; esac
case "$MAXSEQS" in ''|*[!0-9]*|0) die "MAXSEQS must be a positive integer (got $MAXSEQS)" ;; esac
[ "$MOE_FIXES" = 1 ] || PA=off   # measured only on top of the MoE fixes
[ "$MOE_FIXES" = 1 ] || W4A8=0   # W4A8 rides on the a16w4 lane that MOE_FIXES=1 enables
[ "$MOE_FIXES" = 1 ] || { DFP8=0; GF=0; }   # fp8 dense layers and the fused gate were measured only on top of the MoE fixes
[ "$MOE_FIXES" = 1 ] || { PR=0; DG=0; DO=0; }   # pad-route and the drafter-loop knobs were measured only on top of the MoE fixes
[ "$SPEC" != 0 ] || { DH=off; DG=0; DO=0; AS=0; }   # the draft head, the drafter loop and async spec decode only exist with speculation
[ "$DG" = 1 ] || DW=0            # the warm-up captures the drafter-loop graphs, so it needs them
[ "$ACH" = 0 ] || CHUNK=$ACH    # the full budget; the patch caps every step but a solo long prefill at 4096
R4D_REPO=${R4D_REPO:-https://codeberg.org/StillDeadcode/libr4d.git}; R4D_VERSION=${R4D_VERSION:-v0.5.0}
R4D_SRC=${R4D_SRC:-$HOME/.radiance-libr4d-$R4D_VERSION}

# A Hugging Face cache snapshot holds symlinks into ../../blobs: mount the whole hub dir, at the same
# path, so they resolve. Anything else mounts just the checkpoint.
case "$SNAP" in */models--*/snapshots/*) MOUNT=${SNAP%%/models--*} ;; *) MOUNT=$SNAP ;; esac
MODEL_ID=$(basename "$SNAP")
case "$SNAP" in */snapshots/*) MODEL_ID=$(basename "$(dirname "$(dirname "$SNAP")")" | sed 's/^models--//; s#--#/#') ;; esac
SERVED_NAMES=${SERVED_NAMES:-$(basename "$MODEL_ID")}

CACHE_SUF="-f$MOE_FIXES"; [ "$PA" != off ] && CACHE_SUF="$CACHE_SUF-pa$PA"
[ "$GDNFIX" = 1 ] && CACHE_SUF="$CACHE_SUF-gdn2"
[ "$W4A8" = 1 ] && CACHE_SUF="$CACHE_SUF-w4a8"
[ "$DFP8" = 1 ] && CACHE_SUF="$CACHE_SUF-dfp8"
[ "$GF" = 1 ] && CACHE_SUF="$CACHE_SUF-gate"
[ "$ACH" != 0 ] && CACHE_SUF="$CACHE_SUF-ac$ACH"
CACHE=${CACHE:-$HOME/.radiance-cache-moe-$(echo "$MODEL_ID" | tr '/' '_')$CACHE_SUF}
mkdir -p "$CACHE"/vllm "$CACHE"/inductor "$CACHE"/triton "$CACHE"/aiter

RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker, then re-run"; fi
fi
RT_FLAGS=(); GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  RT_FLAGS+=(--replace); GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    [ -n "$gid" ] && GROUP_FLAGS+=(--group-add "$gid")
  done
  [ -z "${DRY_RUN:-}" ] && "$RUNTIME" rm -f "$NAME" >/dev/null 2>&1
fi
[ "${DETACH:-0}" = 1 ] && RT_FLAGS+=(-d)

BACKEND=TRITON_ATTN; R4D_MNT=()
[ "$PA" = r4d ] && BACKEND=R4D
# The libr4d checkout is read by both the GQA-8 prefill build (r4d) and the exact GDN scan build.
if [ "$PA" = r4d ] || [ "$GDNFIX" = 1 ]; then
  if [ ! -f "$R4D_SRC/r4d_attn_prefill_h256_gqa6.hip" ] || [ ! -f "$R4D_SRC/r4d_gdn_wmma.h" ]; then
    echo "[serve-moe] cloning libr4d $R4D_VERSION into $R4D_SRC (once)"
    git clone -q --depth 1 -b "$R4D_VERSION" "$R4D_REPO" "$R4D_SRC" \
      || die "could not clone $R4D_REPO at $R4D_VERSION" "point R4D_SRC at a libr4d $R4D_VERSION checkout, or set MOE_PREFILL_ATTN=off RADIANCE_GDN_SCAN_FIX=0"
  fi
  R4D_MNT=(-v "$(cd "$R4D_SRC" && pwd)":/r4dsrc:ro)
fi
# vLLM refuses --async-scheduling together with disable_padded_drafter_batch, so the two are one switch.
UNPAD=true; ASYNC_FLAG=--no-async-scheduling
[ "$AS" = 1 ] && { UNPAD=false; ASYNC_FLAG=--async-scheduling; }
SPEC_ARGS=()
if [ "$SPEC" != 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$BACKEND\",\"disable_padded_drafter_batch\":$UNPAD}")
fi
PRE=""; pre_add() { PRE="${PRE:+$PRE && }$1"; }
if [ "$MOE_FIXES" = 1 ]; then
  PRE="python3 patch_quark_moe_w4a16.py && python3 patch_gfx12_aiter_a16w4.py && python3 patch_attn_3d_multiq.py"
  FIX_ENV=(-e RADIANCE_MOE_W4A16=1 -e RADIANCE_MOE_BACKEND=aiter -e RADIANCE_ATTN_3D_MAX_Q=16)
  if [ "$PA" = r4d ]; then
    # build the GQA-8 prefill module with the image's hipcc (a few seconds), then install it
    PRE="R4D_SRC=/r4dsrc OUT=/cache/r4d_moe.so bash moe-r4d/build.sh && $PRE && python3 patch_moe_prefill_attn.py"
    FIX_ENV+=(-e RADIANCE_MOE_PREFILL_ATTN=r4d -e RADIANCE_R4D_MOE_SO=/cache/r4d_moe.so)
  fi
else
  FIX_ENV=(-e RADIANCE_MOE_W4A16=0)
fi
# The new kernels are built straight into the image's site-packages, which is also where the patches
# below edit the installed vLLM / radiance sources (never a copy that sits next to the patch scripts).
if [ "$GDNFIX" = 1 ] || [ "$W4A8" = 1 ] || [ "$DFP8" = 1 ]; then
  pre_add "SP=\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
fi
if [ "$GDNFIX" = 1 ]; then
  # libr4d #4: libr4d v0.5.0's GDN chunk scan without the midpoint decay split. moe-gdn2/build.sh patches a
  # temporary copy of R4D_SRC's scan (no libr4d source in this repo), builds it, and the patch binds it.
  pre_add "R4D_SRC=/r4dsrc OUT=\$SP/radiance_gdn2.so bash moe-gdn2/build.sh && python3 patch_gdn_scan_fix.py"
  FIX_ENV+=(-e RADIANCE_GDN_SCAN_FIX=1)
fi
if [ "$W4A8" = 1 ]; then
  # fp8-WMMA W4A8 expert GEMMs for prefill-sized calls; anchors on what patch_gfx12_aiter_a16w4.py inserts
  pre_add "OUT=\$SP bash moe-w4a8/build.sh && python3 patch_moe_w4a8.py"
  FIX_ENV+=(-e RADIANCE_MOE_W4A8=1)
  [ -n "$W4A8_MIN" ] && FIX_ENV+=(-e RADIANCE_MOE_W4A8_MIN_TOKENS="$W4A8_MIN")
fi
if [ "$DH" != off ]; then
  # draft-only compressed lm_head: the drafter's LogitsProcessor gets the copy at Qwen3_5MTP.load_weights, before KV
  # sizing; the target's (verify) keeps bf16. int2 reuses the image's radiance_drafthead (checked, MOE-GFX1201.md).
  pre_add "python3 patch_moe_drafthead.py"
  FIX_ENV+=(-e RADIANCE_MOE_DRAFT_HEAD="$DH")
fi
if [ "$DFP8" = 1 ] || [ "$GF" = 1 ]; then
  # fp8 dense layers (needs the HIP skinny kernel, built here) and/or the fused expert gate (Triton only). One hook at
  # the end of process_weights_after_loading, one flag in qwen2_moe.py.
  [ "$DFP8" = 1 ] && pre_add "OUT=\$SP bash moe-densefp8/build.sh"
  pre_add "python3 patch_moe_densefp8.py"
  FIX_ENV+=(-e RADIANCE_MOE_DENSE_FP8="$DFP8" -e RADIANCE_MOE_GATE_FIX="$GF")
fi
if [ "$PR" = 1 ]; then
  # padded CUDA-graph rows take row 0's router logits: one hook after the router GEMV in vLLM's MoE runner, and the real-row
  # count before every target forward and drafter pass. No cache suffix: the compiled graphs are the same.
  pre_add "python3 patch_moe_padroute.py"
  FIX_ENV+=(-e RADIANCE_MOE_PAD_ROUTE=1)
fi
if [ "$DG" = 1 ] || [ "$DO" = 1 ]; then
  # less host time in the MTP drafter loop: hooks each loop pass's forward in llm_base_proposer.py (graph replay, deferred
  # gate decision) and the gate in the image's radiance_draft.py. After pad-route, which edits the same proposer.
  pre_add "python3 patch_moe_draftloop.py"
  FIX_ENV+=(-e RADIANCE_MOE_DRAFT_GRAPH="$DG" -e RADIANCE_MOE_DRAFT_OVERLAP="$DO")
fi
if [ "$AS" = 1 ]; then
  # async scheduling needs the padded drafter batch (the two flags above) and the image's dynamic draft off: its ragged
  # drafts do not fit vLLM's GPU-resident [B, K] draft tensor. After the drafter-loop patch, which edits the same image file.
  pre_add "python3 patch_moe_async.py"
  FIX_ENV+=(-e RADIANCE_MOE_ASYNC=1)
fi
if [ "$ACH" != 0 ]; then
  # the full --max-num-batched-tokens only for a long prefill alone in the scheduler, 4096 otherwise: one hook in vLLM's
  # Scheduler.schedule and, for SYNC, one in EngineCore.step_with_batch_queue. After the async patch; the warm-up stays last.
  pre_add "python3 patch_adaptive_chunk.py"
  FIX_ENV+=(-e RADIANCE_ADAPTIVE_CHUNK="$ACH" -e RADIANCE_ADAPTIVE_CHUNK_SYNC="$ACS")
fi
if [ "$DW" != 0 ]; then
  # background client in the container: waits for /v1/models, then k concurrent short requests for k = 1..DW, so every
  # drafter-loop graph is captured before the first users arrive. Never fails the serve.
  pre_add "{ python3 warm_draftloop.py $PORT $DW ${SERVED_NAMES%% *} >&2 & }"
fi

HWQ_ENV=(); [ "$HWQ" != 0 ] && HWQ_ENV=(-e GPU_MAX_HW_QUEUES="$HWQ")
echo "[serve-moe] $MODEL_ID ($KIND) fixes=$MOE_FIXES prefill_attn=$PA backend=$BACKEND gdn_scan_fix=$GDNFIX moe_w4a8=$W4A8 draft_head=$DH dense_fp8=$DFP8 gate_fix=$GF pad_route=$PR draft_graph=$DG draft_overlap=$DO async=$AS draft_warm=$DW adaptive_chunk=$ACH achunk_sync=$ACS hw_queues=$HWQ spec=$SPEC max_seqs=$MAXSEQS gpu_util=$GPU_UTIL runtime=$RUNTIME cache=$CACHE"
exec ${DRY_RUN:+echo} "$RUNTIME" run --rm "${RT_FLAGS[@]}" --name "$NAME" --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
  -v "$MOUNT":"$MOUNT":ro -v "$CACHE":/cache -v "$SCRIPT_DIR":/patches:ro ${R4D_MNT[@]+"${R4D_MNT[@]}"} \
  ${HWQ_ENV[@]+"${HWQ_ENV[@]}"} -e HIP_VISIBLE_DEVICES="$GPUS" -e ROCR_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_ROCM_USE_AITER_MHA=0 \
  -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_ROCM_USE_AITER_LINEAR=0 \
  -e VLLM_ROCM_USE_AITER_FP8BMM=0 -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  "${FIX_ENV[@]}" -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e RADIANCE_PRE="$PRE" --entrypoint bash "$IMAGE" \
  -c 'cd /patches && eval "$RADIANCE_PRE" && cd / && exec /opt/radiance_entrypoint.sh "$@"' _ \
  "$SNAP" --served-model-name $SERVED_NAMES --host 0.0.0.0 --port "$PORT" \
  --max-num-seqs "$MAXSEQS" --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-batched-tokens "$CHUNK" --kv-cache-dtype fp8 --attention-backend "$BACKEND" \
  --enable-prefix-caching --mamba-cache-mode align ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72]}' \
  "$ASYNC_FLAG" --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --trust-remote-code "$@"
