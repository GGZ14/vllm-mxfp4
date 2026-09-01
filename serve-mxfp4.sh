#!/bin/bash
# Serve Qwen3.8-27B in native MXFP4 (4-bit) on 2x AMD Radeon AI PRO R9700 (gfx1201), with an FP8
# speculative drafter, on the vllm-radiance image.
#
#   ./setup-mxfp4.sh     one-time: checks the host, pulls the image, builds the checkpoints
#   ./serve-mxfp4.sh     start the server on http://localhost:8080/v1
#   ./serve-mxfp4.sh -h  every knob, its default and what it does
#
# It needs two checkpoints under $MODELS, both produced by setup-mxfp4.sh:
#   Qwen3.8-27B-MXFP4-mtpfp8   AMD's amd/Qwen3.8-27B-Quark-AWQ-MXFP4 with the MTP head requantized
#                              to fp8 by ./fp8_mtp.py. NOT optional: AMD's release leaves mtp.* out
#                              of both `exclude` and `layer_quant_config`, so vLLM applies the mxfp4
#                              scheme to a bf16 head and asserts on a half-width parameter at load.
#                              The drafter is fp8 and not mxfp4 on purpose -- 4-bit costs more
#                              acceptance than it saves in bandwidth, and AWQ does not rescue it.
#   Qwen3.8-27B-DFlash2-FP8    the block-diffusion drafter used by SPEC_METHOD=dflash (the default).
#                              SPEC_METHOD=mtp uses the head inside the target and needs no drafter.
#
# Everything below is `${VAR:-default}`, so any of it can be overridden from the environment
# without editing this file. The defaults are the measured production configuration; each one
# carries the measurement that chose it in the comment above it.
#
# WHAT TO CHECK IN THE LOG
#   "Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"  -> our kernel won the selection
#   "[radiance] native MXFP4 enabled on gfx12x"           -> the aiter fp4 gate was relaxed
#   the R4D selections table (RADIANCE_R4D_REPORT=1)      -> which kernels bound, and why not
#   the stock "current platform does not support native MXFP4/MXFP6" notice still prints and is a
#   false alarm; it comes from a separate supports_mx() call.
#
# Port 8080 is production's and this needs both GPUs, so stop production first:
#   systemctl --user stop qwen_vllm_38          restore with: vllm-switch 38
#
# The measurements behind the defaults, the numerics reference, the 0.5.8 baseline and the history
# of this file are in MXFP4-NOTES.md; the user-facing documentation is in README.md.

set -euo pipefail

# ---------------------------------------------------------------- usage / arguments
usage() {
  cat <<'USAGE'
serve-mxfp4.sh -- native MXFP4 Qwen3.8-27B on 2x R9700 (gfx1201)

  ./setup-mxfp4.sh          one-time setup (host check, image, checkpoints)
  ./serve-mxfp4.sh          serve on http://localhost:8080/v1
  ./serve-mxfp4.sh [ARGS]   any extra arguments are passed through to `vllm serve`

Everything is an environment variable; these are the ones worth knowing.

  MODELS=~/models           directory holding the checkpoints (bind-mounted at /models)
  PORT=8080                 listen port
  IMAGE=...:0.9.3           container image (CACHE is keyed to it -- move both together)
  RUNTIME=podman|docker     container runtime (auto-detected)
  CHAT_TEMPLATE=./qwen3.8-enhanced.jinja
                            chat template; must be readable on the host

  SPEC_METHOD=dflash        speculative drafter: dflash (fastest, needs the DFlash2 checkpoint)
                            or mtp (uses the head inside the target, no extra download)
  SPEC=7 dflash / 4 mtp     speculative depth
  MAXSEQS=8                 max concurrent sequences
  MAXLEN=262144             max context length
  CHUNK=8192                prefill chunk (--max-num-batched-tokens)
  GPU_UTIL=0.98             VRAM fraction; use 0.75 for perplexity work (prompt_logprobs)
  KV_MEM=<bytes>            explicit KV cache size; 0 re-enables vLLM's own profiling

  R4D_ATTN=1                R4D paged attention backend (0 = AITER unified attention)
  FAST_DRAFT=1              int2 draft head with an exact rerank
  MIN_M=0                   M above which the W4A8 kernel takes over from aiter (0 = always)
  AUTO_R4D=1                build the pinned libr4d on first run (cached); 0 uses the image's
  R4D_SO=<dir>              use your own libr4d checkout instead of building one
  EXTRA="--enforce-eager"   extra `vllm serve` flags (same as passing them as arguments)
  DRY_RUN=1                 print the container command instead of running it
  PREPARE_ONLY=1            do the one-time work (image, libr4d) and stop before serving

Full knob reference: README.md. Design notes and measurements: MXFP4-NOTES.md.
USAGE
}

PASSTHRU=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

die() { echo "[serve-mxfp4] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

# ---------------------------------------------------------------- container runtime
# podman and docker differ in three places this script touches: `--replace` is podman-only,
# `--group-add keep-groups` is podman-only (docker wants numeric render/video GIDs), and docker
# needs the stale container removed by hand. Everything else is identical.
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker, then re-run"
  fi
fi
command -v "$RUNTIME" >/dev/null 2>&1 || die "RUNTIME=$RUNTIME is not on PATH"

RT_FLAGS=()
GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  RT_FLAGS+=(--replace)
  GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    if [ -n "$gid" ]; then GROUP_FLAGS+=(--group-add "$gid"); fi
  done
fi

# ---------------------------------------------------------------- host preflight
# Every check here fails with the command that fixes it. They are cheap, and each one stands for a
# failure that otherwise surfaces minutes later as a Python traceback from inside a TP worker.
preflight() {
  [ -e /dev/kfd ] || die "/dev/kfd is missing -- the amdgpu kernel driver is not loaded" \
      "this image ships ROCm userspace, but the kernel driver has to be on the host" \
      "check: ls -l /dev/kfd /dev/dri  and  dmesg | grep amdgpu"
  [ -d /dev/dri ] || die "/dev/dri is missing -- no GPU render nodes on this host"

  local amd=0 d
  for d in /sys/class/drm/renderD*; do
    if [ "$(cat "$d/device/vendor" 2>/dev/null)" = "0x1002" ]; then amd=$((amd+1)); fi
  done
  if [ "$amd" -lt 2 ]; then
    echo "[serve-mxfp4] WARNING: found $amd AMD GPU(s); this configuration serves with" >&2
    echo "  --tensor-parallel-size 2 and will fail at startup with fewer than two." >&2
  fi

  [ -d "$MODELS" ] || die "MODELS=$MODELS does not exist" \
      "point MODELS at the directory holding your checkpoints, or run ./setup-mxfp4.sh"

  if ! "$RUNTIME" image exists "$IMAGE" >/dev/null 2>&1 &&
     ! "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[serve-mxfp4] pulling $IMAGE (a few GiB, once)"
    "$RUNTIME" pull "$IMAGE" || die "could not pull $IMAGE" "pull it by hand, or set IMAGE=<a local tag>"
  fi

  # A listening port is almost always the previous server or production still holding both GPUs.
  # PREPARE_ONLY is doing the one-time work, not serving, so a busy port is irrelevant there.
  [ "${PREPARE_ONLY:-0}" = 1 ] && return 0
  # The probe opens fd 3 in a SUBSHELL, so there is nothing to close here -- and closing it with
  # a bare `exec 3>&- 2>/dev/null` would apply that redirection to the shell itself and silence
  # every error message after it.
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    die "port $PORT is already in use" \
        "another server is running -- this one needs both GPUs to itself:" \
        "  $RUNTIME ps                            # stop the container you find here" \
        "  systemctl --user stop qwen_vllm_38     # or its unit, on the dev box (vllm-switch 38 restores it)" \
        "or serve on a different port: PORT=8081 ./serve-mxfp4.sh"
  fi

  [ -r "$CHAT_TEMPLATE" ] || die "chat template not readable: $CHAT_TEMPLATE" \
      "set CHAT_TEMPLATE=<path to a .jinja on the host>, or leave it unset to use the" \
      "one shipped in this repo (qwen3.8-enhanced.jinja)"
}

# Image and cache MUST move together: cache dirs validate on model + torch/Triton version and must
# not be shared across configurations. Both defaulted to 0.7.4 / -074 long after production moved to
# 0.9.3 / -093, so anyone taking the defaults got a DIFFERENT server than the one being measured.
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
NAME=${NAME:-vllmmxfp4074}
PORT=${PORT:-8080}
CHUNK=${CHUNK:-8192}
R4D_ATTN=${R4D_ATTN:-1}
# GDN in_proj merge (radiance_gdnmerge.py): in_proj_qkvz + in_proj_ba as ONE GEMM, removing 96
# GEMM launches and 48 activation quants per forward. Measured 2026-08-29: single-stream decode
# 26.25 -> 25.50 ms/step (-2.9%), prefill unchanged, all 48 layers merge; stacks with WPERM=1
# for 24.55 ms/step (-6.5%) at a 3% prefill cost. Output drift is split-K reassociation only
# (merged N crosses a dks boundary), same class as the decode kernel's own M-dependent split;
# gated with GSM8K 500q paired.  Resolved EARLY because the CACHE default is keyed on it:
# the merge changes the traced graph, and reusing a cache dir compiled without it replays a
# graph that still calls the two ORIGINAL projections -- whose weights the merge freed --
# and the engine dies at startup on an N=0 GEMM.
GDN_MERGE=${RADIANCE_GDN_MERGE_INPROJ:-1}
# AR/GEMM overlap (radiance_aroverlap.py) changes the traced graph too -- same cache rule.
AR_OVERLAP=${RADIANCE_AR_OVERLAP:-0}
# Norm+quant fusion (2026-08-30). Three pieces that only work TOGETHER: hoist the per-linear fp8
# activation quant into the traced graph (RADIANCE_MXFP4_HOIST_QUANT), swap the aiter pattern's
# replacement op for one that works on RDNA4 (RADIANCE_RMS_QUANT_FUSION + patch_rmsquant_fusion),
# and enable the vLLM passes themselves (pass_config.fuse_norm_quant/fuse_act_quant -- the piece
# the Aug-28 experiment missed: its serve config shows 'fuse_norm_quant': False, so that
# "neutral" result was a null test). Changes the traced graph => own cache suffix.
NQF=${RADIANCE_NORMQUANT_FUSION:-0}
# FP8 residual stream (radiance_arnq): fuse each RowParallel linear's post-AR epilogue
# (residual add + Gemma rmsnorm + per-token fp8 quant) into one HIP kernel and hand the next
# linear a pre-quantized (q, scale). Kernel is bit-identical to the traced path; the contract
# change is why it gets its own cache key and its own gate run. Requires NQF=1 and GDN_MERGE=1.
# TRAP: if the arnq installer SKIPS at startup (guard failure), the stock graph lands in the
# -fp8s cache dir, and because that trace never touched radiance_arnq.py the cache key cannot
# tell the difference afterwards -- a later fixed launch silently replays the stock graph
# (measured 2026-08-30: epilogue kernels 0/step, bench byte-identical). After fixing whatever
# made the installer skip, rm the -fp8s cache dir.
FP8S=${RADIANCE_FP8_STREAM:-0}
# Built with if-appends, NOT $([ ... ] && echo ...): a command substitution that "fails" (the
# test arm) makes the ASSIGNMENT fail, and under set -e that exits the script silently before a
# single line of output. It bit exactly when a flag was 0.
CACHE_SUF=""
if [ "$GDN_MERGE" = 1 ]; then CACHE_SUF="$CACHE_SUF-gdnm"; fi
if [ "$AR_OVERLAP" = 1 ]; then CACHE_SUF="$CACHE_SUF-arov"; fi
# -nqft, not -nqf: -nqf was the pass-only null experiment. TRACED_QUANT flips the traced graph
# via env alone (no hashed file changes), so it MUST key the cache dir.
if [ "$NQF" = 1 ]; then CACHE_SUF="$CACHE_SUF-nqft"; fi
if [ "$FP8S" = 1 ]; then CACHE_SUF="$CACHE_SUF-fp8s"; fi
CACHE=${CACHE:-$HOME/.radiance-cache-w4a8-093$CACHE_SUF}
# prompt_logprobs allocates a ~1-1.7 GiB prompt x vocab logits transient that vLLM does not reserve
# for, and KV is sized to eat everything else -- 0.97 and even 0.92 OOM the engine on ppl.py. Use
# GPU_UTIL=0.75 for perplexity work, 0.98 for throughput.
# 0.98 is the ceiling on this box, not a guess: the card has 32624 MiB, and vLLM measures free
# memory AFTER its own HIP context and torch init exist, so it sees 31980 MiB. 0.99 asks for
# 31.54 GiB and fails at startup. 0.98 gives 857,399 KV tokens against 840,019 at 0.97 and
# survives a full 260k-prefill sweep with no OOM.
GPU_UTIL=${GPU_UTIL:-0.98}
# Explicit KV cache size, which OVERRIDES GPU_UTIL and skips vLLM's memory profiling. Defaulted
# only for the throughput GPU_UTIL, because the ppl.py prompt_logprobs transient above is exactly
# what this eats: with KV pinned, GPU_UTIL=0.75 would no longer buy the headroom it exists to buy.
# KV_MEM=0 forces profiling back on. See the --kv-cache-memory note in the header for re-deriving.
KV_MEM=${KV_MEM:-}
# The KV pin was derived at max_num_seqs=8's capture sizes and activation peak; any other
# MAXSEQS re-profiles instead (re-derive a pin per the header procedure if 16 becomes standing).
if [ -z "$KV_MEM" ] && [ "$GPU_UTIL" = "0.98" ] && [ "${MAXSEQS:-8}" = "8" ]; then KV_MEM=18563072000; fi
if [ "$KV_MEM" = "0" ]; then KV_MEM=""; fi
# Which drafter to speculate with.
#   mtp    -- the multi-token-prediction head inside the target checkpoint. One draft forward per
#             speculative position, so RADIANCE_DYNAMIC_DRAFT can stop the loop early.
#   dflash -- a separate block-diffusion drafter (DFlash2) that emits the whole block in ONE graphed
#             pass. Depth is fixed when its CUDA graph is captured, so DYNAMIC_DRAFT is inert and
#             num_speculative_tokens becomes a real tuning knob again.
# dflash is the default because it is what production serves and what the README's numbers were
# measured on; a default that does not match the shipped configuration silently invalidates any
# A/B run taken against it. It costs one extra 2 GiB download (setup-mxfp4.sh fetches it, and the
# check further down prints the command if it is missing). SPEC_METHOD=mtp needs no drafter at all
# and is the fallback if you do not want the second checkpoint.
SPEC_METHOD=${SPEC_METHOD:-dflash}
# MODELS is bind-mounted at /models below, so SNAP and DRAFTER must live somewhere under it.
# Resolved HERE rather than next to SNAP further down: DRAFTER's default dereferences it, and under
# `set -u` that made an un-exported MODELS an "unbound variable" abort rather than a default.
MODELS="$(realpath -m "${MODELS:-$HOME/models}")"
# Drafter checkpoint for SPEC_METHOD=dflash. Must live under MODELS -- only MODELS is mounted.
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-FP8}
# The drafter's own attention backend. It has to support FULL cuda graphs or vLLM logs "running the
# draft eagerly" and the single-pass draft loses its graph -- which is the entire point of dflash.
# TRITON_ATTN does; R4D is the target's backend and is what mtp uses for the drafter too.
DRAFT_ATTN=${DRAFT_ATTN:-TRITON_ATTN}
# Speculative depth.
#   mtp: measured on this build, 4 beats 8 at decode -- 59.8/60.2 tok/s against 53.1/58.6, because
#   acceptance falls (42.1% -> 33.7%) faster than the deeper drafts pay for themselves. The 0.5.8
#   baseline also ran 4, so this keeps the comparison honest as well as fast.
#   dflash: the drafter's block_size is 8; 7 is the shipped default and the depth is
#   CONTENT-DEPENDENT, so mind the corpus before re-tuning it. The 2026-08-29 sweep on
#   bench_decode_conc said 5 (+8-13% aggregate at every level) -- but that corpus asks for
#   deliberately non-repetitive prose, which is exactly the low-acceptance content where shallow
#   drafts win. On BetterBench's weighted mix (code 0.30), same build, back to back: SPEC=7
#   combined decode 184.3 t/s vs SPEC=5's 159.4 (+15.6% for 7) -- code/json/file_edit run
#   tok/update 4.7-6.0 at depth 7 and the cap at 5 truncates precisely that tail. 5 remains the
#   better setting for prose-heavy or batch-throughput serving (conc-8 562 vs 544 aggregate);
#   8 falls off DEC_MAX_TM at conc 8 (M=72>64, -25%). Tune acceptance-coupled knobs on the
#   weighted mix, not on a single content class.
#
#   RADIANCE_DYNAMIC_WIDTH (patch_dynwidth.py, default ON) mostly dissolves this trade: the
#   scheduler caps each request's VERIFY width from a per-request acceptance EMA (the DFlash2
#   draft pass is one fixed-cost graphed block either way), so prose sequences verify ~4 wide
#   while code keeps the full depth. Measured at base SPEC=7: weighted single-stream unchanged
#   (184.7 vs 184.3) with code tok/update intact, and conc-8 recovers static SPEC=5's batch
#   efficiency (steps 52-57 -> 46-47 ms, aggregate 391-413 -> 444-461 t/s). Lossless by
#   construction -- verification preserves the distribution at any proposal length.
if [ "$SPEC_METHOD" = dflash ]; then SPEC=${SPEC:-7}; else SPEC=${SPEC:-4}; fi
# The tuned drafter stack. The right default is NOT the same for both methods:
#   mtp    -- 1. The 2-bit draft head with an exact rerank is a straight win here (+6.5% decode).
#   dflash -- 1 as of 2026-08-27, WITH RERANK=64 (below). It used to be 0: FAST_DRAFT=1 crashed
#             this drafter at load with an IndexError in vLLM's rocm_unquantized_gemm_impl. That
#             was radiance_w4 freeing `layer.weight` to torch.empty(0) and DFlash2's fused
#             context-KV precompute then slicing it -- `k = weight.shape[1]` on a 1-D tensor. It no
#             longer fires because the pinned libr4d (b9e42ab) ships no w4a16 gemm_nt kernel, so
#             radiance_w4 disables itself and only the int2 head arms. IF LIBR4D IS EVER REBUILT
#             WITH r4d_gemm_w4a16_nt_m64, that crash path comes back and needs a guard in
#             patch_dflash_mxfp4_kv.py for a converted (0-element) weight.
#             Measured, ctx 0, 3 reps, interleaved A/B/A/B, dup-8gram 0.0% throughout:
#               bf16 head        30.13 ms/step | acc/draft 1.904 |  96.4 tok/s
#               int2 R=32        28.43         | acc/draft 1.804 |  98.6   (-5.3% acceptance)
#               int2 R=64        28.66         | acc/draft 1.904 | 101.3   (+5.1%)
if [ "$SPEC_METHOD" = dflash ]; then FAST_DRAFT=${FAST_DRAFT:-1}; else FAST_DRAFT=${FAST_DRAFT:-1}; fi
# Rerank width. RADIANCE_DRAFT_RERANK caps the candidate pool a TOP-K caller can draw from, because
# _radiance_topk_only blanks everything the rerank did not touch. mtp asks the head for an argmax
# and 32 is ample; DFlash2 asks for selector_top_k=16 and 32 costs 5.3% of acceptance. 64 restores
# it EXACTLY to the bf16 head's 1.904 for +0.23 ms, and 128/256 measure identical -- so the pool
# saturates at 4x K, and this is a ceiling to raise with selector_top_k, not a free parameter.
# 80 rather than 64 under dflash: VERIFY_HEAD needs 4x the SAMPLER's top_k (20 here) as well as 4x
# the drafter's selector_top_k (16). At 64 the verify gate rejects every sampled request and the
# feature silently does nothing. The drafter is indifferent -- 64/128/256 measured identical.
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_DRAFT_RERANK=${RADIANCE_DRAFT_RERANK:-80}; fi
# int2 TARGET verify head. ON under dflash as of 2026-08-27: the profile shows the bf16 lm_head is
# one 2.02 ms GEMM per step (5.9% of wall) and this reuses the drafter's int2 packing at zero extra
# VRAM. BetterBench single pass, combined decode 170.0 -> 174.9 t/s (+2.9%) with all eight
# categories +2.7 to +3.4%, conc 1/2/4 +2.8/+2.5/+1.6%, conc 8 neutral, prefill unchanged.
# Output-equivalent on everything measured: GSM8K 500q greedy identical (486/500 both), 8/8 greedy
# completions byte-identical, and 24/24 SEEDED SAMPLED completions byte-identical at the serve's own
# temperature 0.7 / top_p 0.95 / top_k 20.
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_VERIFY_HEAD=${RADIANCE_VERIFY_HEAD:-1}; fi
# Context length. Only lower it for diagnostics -- the FLA GDN fallback allocates against this,
# not against the chunk size, and OOMs at 262144.
MAXLEN=${MAXLEN:-262144}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Chat template. It is mounted into the container by path, so it must exist ON THE HOST: this was
# hardcoded to a file under ~/.cache/huggingface that only ever existed on the box it was written
# on, which made a fresh clone fail at startup with a missing-file error from vllm rather than
# anything pointing at the cause. The repo's own template is the default now; point CHAT_TEMPLATE
# at your own to override, e.g. the qwen-fixed series if you have it.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$SCRIPT_DIR/qwen3.8-enhanced.jinja}
CHAT_TEMPLATE="$(realpath -m "$CHAT_TEMPLATE")"
PATCHES_DIR="$(realpath -m "${PATCHES:-$SCRIPT_DIR}")"
# A template inside the repo rides the /patches mount that is already there (already SELinux
# relabelled by its :z); anything else gets its own read-only mount.
CT_MOUNT=()
case "$CHAT_TEMPLATE" in
  "$PATCHES_DIR"/*) CT_PATH="/patches/${CHAT_TEMPLATE#"$PATCHES_DIR"/}" ;;
  *) CT_PATH=/chat-template.jinja; CT_MOUNT+=(-v "$CHAT_TEMPLATE:$CT_PATH:ro,z") ;;
esac

preflight
# A libr4d checkout DIRECTORY whose r4d.so is copied over the image's at container start. Leave
# unset and it is built for you (see AUTO_R4D just below); set it to use your own checkout.
# Needed because the GDN overflow fixes are upstream (StillDeadcode/libr4d PR #1, merged) but the
# only tag is still v0.4.0 and the 0.7.4 image pins v0.4.0 -- so the SHIPPED kernel predates the
# fix and NaNs the gated-delta-net output on this model: WikiText-2 PPL 653586 vs 8.3706. Once
# deadcode tags a release and ships an image pinning it, all of this can go away.
R4D_SO=${R4D_SO:-}
# Built automatically when R4D_SO is unset: libr4d is cloned at the pinned commit and compiled
# inside $IMAGE once, then cached and reused. Costs a few minutes on the first launch only.
# AUTO_R4D=0 opts out and runs the image stock kernel (broken on this model -- see above), and
# setting R4D_SO by hand still wins, so an existing checkout is never rebuilt behind your back.
R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_CACHE=${R4D_CACHE:-$HOME/.cache/radiance-libr4d}
# r4d_radiance_extras.patch carries this repo's libr4d additions on top of the pinned commit:
# the 8-bit prefill attention legs (R4D_ATTN_FP8) and the fused GDN decode step
# (RADIANCE_GDN_FUSED_UPDATE). The build cache key carries a suffix so patched and stock builds
# coexist; bump the suffix whenever the patch content changes, or a stale build serves silently.
R4D_PATCH="$SCRIPT_DIR/r4d_radiance_extras.patch"
R4D_KEY="$R4D_PIN"
if [ -f "$R4D_PATCH" ]; then R4D_KEY="$R4D_PIN-rx4"; fi
if [ -z "$R4D_SO" ] && [ "${AUTO_R4D:-1}" = 1 ]; then
  if [ ! -f "$R4D_CACHE/$R4D_KEY/r4d.so" ]; then
    echo "[radiance] building libr4d $R4D_KEY in $IMAGE -- one time, a few minutes"
    rm -rf "$R4D_CACHE/.build"
    mkdir -p "$R4D_CACHE/.build"
    git clone -q https://codeberg.org/StillDeadcode/libr4d.git "$R4D_CACHE/.build"
    git -C "$R4D_CACHE/.build" checkout -q "$R4D_PIN"
    if [ "$R4D_KEY" != "$R4D_PIN" ]; then
      git -C "$R4D_CACHE/.build" apply "$R4D_PATCH"
    fi
    "$RUNTIME" run --rm --entrypoint bash -v "$R4D_CACHE/.build":/work:z -w /work \
      "$IMAGE" -c ./build.sh
    # publish only after a successful build, so an interrupted one is not cached as good
    mv "$R4D_CACHE/.build" "$R4D_CACHE/$R4D_KEY"
  fi
  R4D_SO="$R4D_CACHE/$R4D_KEY"
  echo "[radiance] libr4d $R4D_KEY -> $R4D_SO"
fi
if [ "${PREPARE_ONLY:-0}" = 1 ]; then
  echo "[radiance] prepared: image pulled and libr4d built -- ready to serve"
  exit 0
fi
# Where the hand-written W4A8 kernel takes over from aiter's W4A4 Triton path.
# DEFAULT 0 = never fall back; our kernel serves every M. The comparison is `x.shape[0] > MIN_M`,
# so MIN_M=1 would still route M=1 to aiter -- use 0, not 1.
#
# This was 16 until the decode kernel landed, for two separate reasons that are now both resolved:
#
#   CORRECTNESS. aiter's W4A4 path returns a WRONG result for N=5120 K=3072 (o_proj): captured from
#   a live serve and replayed against an fp32 reference, aiter lands at rel=1.066 with ~1/35th of the
#   correct magnitude, while ours is at rel=0.0017. That shape has no tuned table in mxfp4-configs/,
#   so it takes aiter's generic bands. At MIN_M=16 it went unnoticed in prefill (M=17, our kernel)
#   and poisoned decode (M=9, aiter) -- the fluent-looking garbage this build shipped with for an
#   afternoon.
#
#   SPEED. MIN_M=0 used to be a ~55% decode regression (54.3 ms/step against 35.1) because the only
#   kernel available at M<=16 was the prefill-tiled one, which at M=5 issues 51x more matrix MACs
#   than useful. RADIANCE_MXFP4_DECODE_MAX_M below fixes exactly that, so MIN_M=0 is now both
#   correct AND faster than the old default.
#
# Set it absurdly high to route everything to aiter -- only useful for bisecting.
MIN_M=${MIN_M:-0}
# The decode-kernel band must cover MAXSEQS x (SPEC+1) rows or the biggest verify batches fall
# onto the prefill tile: 64 covers the 8-stream default exactly (dflash SPEC=7 -> 8x8), 128
# covers 16 streams. Defaulted from MAXSEQS so the 8-and-under band routes IDENTICALLY to today.
if [ "${MAXSEQS:-8}" -gt 8 ]; then
  RADIANCE_MXFP4_DECODE_MAX_M=${RADIANCE_MXFP4_DECODE_MAX_M:-128}
fi

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
# Cudagraph capture sizes; empty/none = vLLM's default list ([1,2,4] + multiples of 8).
# Finer sizes (3,5,6,7,10,12,14) were tried 2026-08-29 to un-pad dynamic-width single streams and
# measured NEUTRAL (181.3 vs 184.7 weighted, inside noise): the decode-band GEMMs are
# weight-stream-bound and nearly M-invariant below M~16 (tier7: gate_up 88.5 us at M=5 vs 88.7
# at M=8), so there was no single-stream width cost hiding behind the padding to recover --
# dynamic width's value is batching, where M crosses real cost and split-K boundaries. The knob
# stays for capture experiments; the default stays stock. SPEC=8 + dynamic width was measured in
# the same session: single-stream 184.9 (even), conc-8 405-427 vs 444-461 (LOSES -- cold-start
# batches run full width into the M=72>64 kernel cliff before the EMAs settle). 7 stays.
CAPTURE_SIZES=${CAPTURE_SIZES:-none}
# Compilation-config entries accumulate into ONE flag: two --compilation-config instances would
# not merge (argparse keeps the last).
CC_ITEMS=""
if [ -n "$CAPTURE_SIZES" ] && [ "$CAPTURE_SIZES" != none ]; then
  CC_ITEMS="\"cudagraph_capture_sizes\":$CAPTURE_SIZES"
fi
if [ "$NQF" = 1 ]; then
  CC_ITEMS="${CC_ITEMS:+$CC_ITEMS,}\"pass_config\":{\"fuse_norm_quant\":true,\"fuse_act_quant\":true}"
fi
if [ -n "$CC_ITEMS" ]; then
  EXTRA="$EXTRA --compilation-config {$CC_ITEMS}"
fi
# PROFILE_DIR=1 arms the torch profiler (vLLM 0.27 moved it from VLLM_TORCH_PROFILER_DIR to CLI
# flags); traces land in $CACHE/prof, driven by POST /start_profile and /stop_profile.
if [ -n "${PROFILE_DIR:-}" ]; then
  mkdir -p "$CACHE/prof"
  # PROFILE_STACK=1 adds python stacks to the trace (bigger, slower flush; use for ATTRIBUTION
  # runs, not timing runs -- with_stack inflates the very gaps being measured).
  if [ "${PROFILE_STACK:-0}" = 1 ]; then WITH_STACK=true; else WITH_STACK=false; fi
  EXTRA="$EXTRA --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/prof --profiler-config.torch_profiler_with_stack=$WITH_STACK"
fi

SNAP="$(realpath -m "${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}")"
# -f follows symlinks, so a checkpoint assembled as a symlink farm into the HF cache fails
# this test on the HOST even though it resolves fine in the container, where the cache is
# bind-mounted at /root/.cache/huggingface. Accept a dangling symlink too and let the
# container be the judge; a genuinely absent checkpoint still has neither.
if [ ! -f "$SNAP/config.json" ] && [ ! -L "$SNAP/config.json" ]; then
  echo "no checkpoint at $SNAP" >&2
  echo >&2
  echo "Run the one-time setup, which downloads AMD's release and builds this checkpoint from it:" >&2
  echo >&2
  echo "  ./setup-mxfp4.sh" >&2
  echo >&2
  echo "It is not an optimization you can skip. AMD's release does not load as-is: its bf16 mtp.*" >&2
  echo "layers are named in neither \`exclude\` nor \`layer_quant_config\`, so vLLM applies the mxfp4" >&2
  echo "scheme to them and asserts on a half-width parameter. ./fp8_mtp.py requantizes that head to" >&2
  echo "fp8 and writes the matching layer_quant_config; setup-mxfp4.sh just drives it for you." >&2
  exit 1
fi
# HF_HUB_OFFLINE=1 inside the container and the cache mounts at /root/.cache/huggingface, so vllm
# must be handed the CONTAINER path -- a host path fails HF repo-id validation, not "not found".
# Derived from SNAP rather than hardcoded, so overriding SNAP actually redirects the server
# instead of silently serving whatever sits at the default name inside the mount.
case "$SNAP" in
  "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
  *) echo "SNAP ($SNAP) must be under MODELS ($MODELS): only MODELS is mounted into the" >&2
     echo "container. Move the checkpoint there, or set MODELS to a directory containing it." >&2
     exit 1 ;;
esac

if [ "$R4D_ATTN" = "1" ]; then ATTN=R4D; else ATTN=ROCM_AITER_UNIFIED_ATTN; fi

# Async scheduling overlaps the host's scheduling work with GPU execution, which is the standard
# answer to a large launch gap. vLLM refuses it together with disable_padded_drafter_batch, so the
# two are one switch here. The unpad lever is worth ~+50% single-stream on the 27B hybrids under
# MTP, where the drafter runs a SERIAL loop of forwards and the padding is paid once per position.
# Under dflash the drafter emits the whole block in one graphed pass, so it is worth re-testing
# which side of that trade wins.
ASYNC=${ASYNC:-0}
if [ "$ASYNC" = 1 ]; then ASYNC_FLAG="--async-scheduling"; UNPAD=false; else ASYNC_FLAG="--no-async-scheduling"; UNPAD=true; fi

# Speculative config, built here so the drafter path is validated before podman is invoked rather
# than surfacing as an HF repo-id error inside the worker.
if [ "$SPEC_METHOD" = dflash ]; then
  DRAFTER="$(realpath -m "$DRAFTER")"
  if [ ! -f "$DRAFTER/config.json" ]; then
    echo "no dflash drafter at $DRAFTER" >&2
    echo >&2
    echo "Fetch it (2 GiB), or let ./setup-mxfp4.sh do it:" >&2
    echo "  hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir $DRAFTER" >&2
    echo >&2
    echo "Or serve without it, using the MTP head inside the target checkpoint instead:" >&2
    echo "  SPEC_METHOD=mtp ./serve-mxfp4.sh" >&2
    exit 1
  fi
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) echo "DRAFTER ($DRAFTER) must be under MODELS ($MODELS): only MODELS is mounted." >&2
       exit 1 ;;
  esac
  # disable_padded_drafter_batch is the single-stream lever (~+50% on the 27B hybrids) and the
  # image bakes the vLLM unpad patch it relies on; it applies to dflash as well as mtp.
  # DRAFT_SAMPLE=probabilistic drafts stochastically with vLLM's shared-Gumbel coupling
  # instead of argmax. The serve samples at temperature 0.7, and greedy one-hot drafts accept
  # with only p_target(argmax); matched sampling accepts with sum(min(p,q)). Costs the full
  # draft-logits head (bypasses the int2 argmax fast path) until the sparse draft_logits_spec
  # integration exists -- measure acceptance vs that cost before defaulting.
  DRAFT_SAMPLE=${DRAFT_SAMPLE:-greedy}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$CDRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}"
else
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
fi

# The AR size gate compares the raw bf16 byte count: CHUNK x hidden(5120) x 2. Derive it rather
# than hardcoding it, so changing CHUNK cannot silently drop prefill back onto RCCL.
AR_MAX_KB=$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))


mkdir -p "$CACHE"/{vllm,inductor,triton,aiter}

echo "[run] $RUNTIME $IMAGE | port $PORT | $SPEC_METHOD spec=$SPEC | model $CSNAP"
echo "[run] attn=$ATTN chunk=$CHUNK ar_max_kb=$AR_MAX_KB fast_draft=$FAST_DRAFT rerank=${RADIANCE_DRAFT_RERANK:-32} vhead=${RADIANCE_VERIFY_HEAD:-0} min_m=$MIN_M fuse_rms=${RADIANCE_FUSE_RMS_QUANT:-1} preshuf=${RADIANCE_PRESHUFFLE:-1} util=$GPU_UTIL kv_mem=${KV_MEM:-profiled}"
echo "[run] cache=$CACHE"
echo "[run] chat-template=$CHAT_TEMPLATE"
echo "[run] follow the log with: $RUNTIME logs -f $NAME    stop with: $RUNTIME stop $NAME"

# docker has no --replace, so a container left behind by a previous run has to go first.
if [ "$RUNTIME" != podman ]; then "$RUNTIME" rm -f "$NAME" >/dev/null 2>&1 || true; fi

# DRY_RUN=1 prints the command instead of running it -- for checking what a set of environment
# overrides actually produces, and for lifting the invocation into a unit file.
exec ${DRY_RUN:+echo} "$RUNTIME" run "${RT_FLAGS[@]}" --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
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
  -e RADIANCE_FAST_DRAFT="$FAST_DRAFT" -e RADIANCE_DRAFT_TAU="${RADIANCE_DRAFT_TAU:-0.20}" \
  -e RADIANCE_DRAFT_RERANK="${RADIANCE_DRAFT_RERANK:-32}" \
  -e RADIANCE_DFLASH_SELECTOR_TOPK="${RADIANCE_DFLASH_SELECTOR_TOPK:-}" \
  -e RADIANCE_VERIFY_HEAD="${RADIANCE_VERIFY_HEAD:-0}" \
  -e RADIANCE_VERIFY_HEAD_MAX_M="${RADIANCE_VERIFY_HEAD_MAX_M:-32}" \
  -e RADIANCE_MXFP4_DEBUG="${RADIANCE_MXFP4_DEBUG:-0}" \
  -e RADIANCE_MXFP4_PUREQUANT="${RADIANCE_MXFP4_PUREQUANT:-0}" \
  -e RADIANCE_MXFP4_SYNC="${RADIANCE_MXFP4_SYNC:-0}" \
  -e RADIANCE_MXFP4_CLONE="${RADIANCE_MXFP4_CLONE:-0}" -e RADIANCE_MXFP4_CHECKX="${RADIANCE_MXFP4_CHECKX:-0}" \
  -e RADIANCE_MXFP4_PADOUT="${RADIANCE_MXFP4_PADOUT:-0}" \
  -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-0}" \
  -e RADIANCE_GDN_MERGE_INPROJ="$GDN_MERGE" \
  -e R4D_ATTN_FP8="${R4D_ATTN_FP8:-3}" \
  -e RADIANCE_AR_OVERLAP="$AR_OVERLAP" \
  -e RADIANCE_GDN_FUSED_UPDATE="${RADIANCE_GDN_FUSED_UPDATE:-1}" \
  -e RADIANCE_DYNAMIC_WIDTH="${RADIANCE_DYNAMIC_WIDTH:-1}" \
  -e RADIANCE_DYNW_ALPHA="${RADIANCE_DYNW_ALPHA:-0.35}" \
  -e RADIANCE_DYNW_MARGIN="${RADIANCE_DYNW_MARGIN:-2}" \
  -e RADIANCE_DYNW_MIN="${RADIANCE_DYNW_MIN:-2}" \
  -e RADIANCE_DYNW_MIN_BATCH="${RADIANCE_DYNW_MIN_BATCH:-3}" \
  -e RADIANCE_AR_QNB="${RADIANCE_AR_QNB:-96}" \
  -e RADIANCE_AR_QNT="${RADIANCE_AR_QNT:-1024}" \
  ${PYTORCH_CUDA_ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
  -e RADIANCE_AR_OVERLAP_MIN_M="${RADIANCE_AR_OVERLAP_MIN_M:-2048}" \
  -e RADIANCE_AR_OVERLAP_SLICES="${RADIANCE_AR_OVERLAP_SLICES:-4}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" \
  -e RADIANCE_MXFP4_R4D_DECODE_MAX_M="${RADIANCE_MXFP4_R4D_DECODE_MAX_M:-0}" \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS="${RADIANCE_TOPK_TRITON_MIN_ROWS:-1}" \
  -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_DFLASH_CALIB="${RADIANCE_DFLASH_CALIB:-}" \
  -e RADIANCE_DFLASH_CALIB_TOKENS="${RADIANCE_DFLASH_CALIB_TOKENS:-200000}" \
  -e RADIANCE_MXFP4_HOIST_QUANT="${RADIANCE_MXFP4_HOIST_QUANT:-$NQF}" \
  -e RADIANCE_MXFP4_TRACED_QUANT="${RADIANCE_MXFP4_TRACED_QUANT:-$NQF}" \
  -e RADIANCE_FP8_STREAM="$FP8S" \
  -e RADIANCE_RMS_QUANT_FUSION="${RADIANCE_RMS_QUANT_FUSION:-$NQF}" \
  -e RADIANCE_MXFP4_SHADOW="${RADIANCE_MXFP4_SHADOW:-}" \
  -e RADIANCE_MXFP4_SANITIZE="${RADIANCE_MXFP4_SANITIZE:-0}" \
  -e RADIANCE_GDN_PATHS="${RADIANCE_GDN_PATHS:-both}" \
  -e RADIANCE_GDN_NANTRACE="${RADIANCE_GDN_NANTRACE:-0}" \
  -e RADIANCE_MXFP4_KERNEL_N="${RADIANCE_MXFP4_KERNEL_N:-}" \
  -e RADIANCE_MXFP4_KERNEL_NK="${RADIANCE_MXFP4_KERNEL_NK:-}" \
  -e RADIANCE_MXFP4_CHECKALL="${RADIANCE_MXFP4_CHECKALL:-}" \
  -e RADIANCE_MXFP4_MHIST="${RADIANCE_MXFP4_MHIST:-0}" \
  -e RADIANCE_MXFP4_DECODE_KS="${RADIANCE_MXFP4_DECODE_KS:-}" \
  -e RADIANCE_MXFP4_DECODE_BK="${RADIANCE_MXFP4_DECODE_BK:-}" \
  -e RADIANCE_MXFP4_CHECK_MAX_M="${RADIANCE_MXFP4_CHECK_MAX_M:-128}" \
  -e RADIANCE_MXFP4_PERBLOCK_NK="${RADIANCE_MXFP4_PERBLOCK_NK:-}" \
  -e RADIANCE_MXFP4_REFLINEAR="${RADIANCE_MXFP4_REFLINEAR:-0}" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1 \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "$MODELS":/models \
  -v "$CACHE":/cache \
  -v "${PATCHES:-$SCRIPT_DIR}":/patches:z \
  ${CT_MOUNT[@]+"${CT_MOUNT[@]}"} \
  ${R4D_SO:+-v "$R4D_SO":/r4d:z} \
  ${R4D_SO:+-e R4D_SO="$R4D_SO"} \
  --entrypoint bash \
  "$IMAGE" -lc '
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
    python3 patch_gdn_merge_inproj.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    # Non-fatal: fixes content=null on thinking-off requests; not required to serve.
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    # radiance_drafthead.py is copied too so RADIANCE_DRAFT_RERANK can be swept without an
    # image rebuild. The repo copy was byte-identical to the 0.9.3 one before that knob existed.
    cp radiance_mxfp4.py radiance_gdn.py radiance_rmsquant.py radiance_drafthead.py \
       radiance_verifyhead.py radiance_gdnmerge.py radiance_aroverlap.py radiance_topk.py \
       radiance_arnq.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    # Optional patched libr4d. R4D_SO is the DIRECTORY of a libr4d checkout built from main --
    # it is bind-mounted at /r4d and its r4d.so replaces the one in the image. For an image
    # rebuild, the Dockerfile supports the same substitution through R4D_REPO / R4D_VERSION.
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
    ${KV_MEM:+--kv-cache-memory "$KV_MEM"} \
    --max-model-len "$MAXLEN" --max-num-seqs "${MAXSEQS:-8}" --max-num-batched-tokens "$CHUNK" \
    --attention-backend "$ATTN" \
    --speculative-config "$SPEC_CFG" \
    $ASYNC_FLAG $EXTRA \
    --enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
    --chat-template "$CT_PATH" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
