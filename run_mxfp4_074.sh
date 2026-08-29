#!/bin/bash
# PRODUCTION launcher: native MXFP4 body on gfx1201 with an FP8 drafter, on radiance 0.9.3 (libr4d).
#
# The name still says 074 because that is what the file was called when it targeted the 0.7.4 image;
# the defaults below track production and are what actually decide which server you get. The script
# began as an evaluation harness for 0.7.4 and became the thing that starts the real server, which
# is how its defaults came to lag two image versions behind what they launch.
#
# The 0.5.8 form of this run is run_mxfp4_minm.sh, kept as-is because it is the only way to
# reproduce the baseline these numbers are measured against:
#   prefill 3602 / 3409 / 2283 / 1730 / 1390 tok/s at 7.8k/26k/104k/182k/259k
#   real decode 55.4 / 61.7 tok/s | weights 9.24 GiB/GPU | KV ~856k tok | WikiText-2 PPL 8.3335
#
# Checkpoint built by this repo's ./fp8_mtp.py from amd/Qwen3.8-27B-Quark-AWQ-MXFP4; run it once
# before this script (it prints the command if the checkpoint is missing). AMD's release will not
# load as-is: its mtp.* layers are bf16 but named in neither `exclude` nor `layer_quant_config`,
# so vLLM applies the global mxfp4 scheme to them and asserts on a half-width parameter.
#
# The drafter is FP8, not MXFP4, and that is a settled result: MXFP4 RTN cost acceptance
# 2.5 -> 2.21 and AWQ did not rescue it (MXFP4's per-32 E8M0 block exponent already does most of
# what per-channel scaling would), while FP8 e4m3 per-channel holds acceptance at 2.60-2.80. Do
# not point this at Qwen3.8-27B-MXFP4-mtpq or -mtpawq; those exist only for that comparison.
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
#   - --kv-cache-memory IS set now, to 18563072000 (17.29 GiB), but only when GPU_UTIL is the
#     throughput default. Do NOT re-derive it from vLLM's "fit into requested memory" line: that
#     number is computed against the GPU_UTIL budget (0.98 x 31.86 = 31.22 GiB), not against the
#     card, and on 0.9.3 it suggests 15.47-15.61 GiB while the profiled run is already using 16.37
#     and fits. Derive it from the card instead -- `rocm-smi --showmeminfo vram` INSIDE the
#     container, under an 8-way 8192-chunk load, then hand KV everything but ~0.3 GiB. Measured:
#     profiled 0.98 peaks at 30.64 of 31.86 GiB and the free 1.22 GiB stays free under load (KV is
#     pre-allocated and the activation pool is reserved at profiling time), so 17.29 GiB peaks at
#     31.57 with 0.29 free. Worth 892,799 -> 943,581 tokens. Re-derive after anything that moves
#     weights, cudagraph sizes or CHUNK; getting it wrong OOMs at startup, not under load.
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
#   DECODE_MAX_M     the small-M decode GEMM (M<=64, TM=ceil(M/16), split-K). ON BY DEFAULT at 64.
#                      Needs MIN_M=0 to be reachable at all -- at MIN_M=16 the M=5 decode call
#                      never enters our launcher. Measured: single-stream step time 35.1 -> 33.4 ms
#                      (-4.8%), aggregate throughput +28.5% at 4 concurrent and +19.7% at 8, prefill
#                      unchanged within 1.2%. GSM8K 500q paired: 486 both correct, 3/3 discordant,
#                      sign test p=1.00, at 14% less wall. Same numbers in the README table.
#                      Set 0 to fall back to the prefill-tiled kernel for every M.
#
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
# Built with if-appends, NOT $([ ... ] && echo ...): a command substitution that "fails" (the
# test arm) makes the ASSIGNMENT fail, and under set -e that exits the script silently before a
# single line of output. It bit exactly when a flag was 0.
CACHE_SUF=""
if [ "$GDN_MERGE" = 1 ]; then CACHE_SUF="$CACHE_SUF-gdnm"; fi
if [ "$AR_OVERLAP" = 1 ]; then CACHE_SUF="$CACHE_SUF-arov"; fi
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
if [ -z "$KV_MEM" ] && [ "$GPU_UTIL" = "0.98" ]; then KV_MEM=18563072000; fi
if [ "$KV_MEM" = "0" ]; then KV_MEM=""; fi
# Which drafter to speculate with.
#   mtp    -- the multi-token-prediction head inside the target checkpoint. One draft forward per
#             speculative position, so RADIANCE_DYNAMIC_DRAFT can stop the loop early.
#   dflash -- a separate block-diffusion drafter (DFlash2) that emits the whole block in ONE graphed
#             pass. Depth is fixed when its CUDA graph is captured, so DYNAMIC_DRAFT is inert and
#             num_speculative_tokens becomes a real tuning knob again.
SPEC_METHOD=${SPEC_METHOD:-mtp}
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
if [ -f "$R4D_PATCH" ]; then R4D_KEY="$R4D_PIN-rx2"; fi
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
    podman run --rm --entrypoint bash -v "$R4D_CACHE/.build":/work:z -w /work \
      "$IMAGE" -c ./build.sh
    # publish only after a successful build, so an interrupted one is not cached as good
    mv "$R4D_CACHE/.build" "$R4D_CACHE/$R4D_KEY"
  fi
  R4D_SO="$R4D_CACHE/$R4D_KEY"
  echo "[radiance] libr4d $R4D_KEY -> $R4D_SO"
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
if [ -n "$CAPTURE_SIZES" ] && [ "$CAPTURE_SIZES" != none ]; then
  EXTRA="$EXTRA --compilation-config {\"cudagraph_capture_sizes\":$CAPTURE_SIZES}"
fi
# PROFILE_DIR=1 arms the torch profiler (vLLM 0.27 moved it from VLLM_TORCH_PROFILER_DIR to CLI
# flags); traces land in $CACHE/prof, driven by POST /start_profile and /stop_profile.
if [ -n "${PROFILE_DIR:-}" ]; then
  mkdir -p "$CACHE/prof"
  EXTRA="$EXTRA --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/prof --profiler-config.torch_profiler_with_stack=false"
fi

SNAP="$(realpath -m "${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}")"
# -f follows symlinks, so a checkpoint assembled as a symlink farm into the HF cache fails
# this test on the HOST even though it resolves fine in the container, where the cache is
# bind-mounted at /root/.cache/huggingface. Accept a dangling symlink too and let the
# container be the judge; a genuinely absent checkpoint still has neither.
if [ ! -f "$SNAP/config.json" ] && [ ! -L "$SNAP/config.json" ]; then
  echo "no checkpoint at $SNAP" >&2
  echo >&2
  echo "Build it from AMD's MXFP4 release (one-off, ~15 min, needs the fp8 MTP head -- the stock" >&2
  echo "checkpoint asserts on load because its bf16 mtp.* layers fall through to the mxfp4 scheme):" >&2
  echo >&2
  echo "  hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4" >&2
  echo "  ./fp8_mtp.py \\" >&2
  echo "    \"\$(ls -d ~/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/*/ | head -1)\" \\" >&2
  echo "    \"$SNAP\"" >&2
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
    echo "  hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir $DRAFTER" >&2
    exit 1
  fi
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) echo "DRAFTER ($DRAFTER) must be under MODELS ($MODELS): only MODELS is mounted." >&2
       exit 1 ;;
  esac
  # disable_padded_drafter_batch is the single-stream lever (~+50% on the 27B hybrids) and the
  # image bakes the vLLM unpad patch it relies on; it applies to dflash as well as mtp.
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$CDRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
else
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
fi

# The AR size gate compares the raw bf16 byte count: CHUNK x hidden(5120) x 2. Derive it rather
# than hardcoding it, so changing CHUNK cannot silently drop prefill back onto RCCL.
AR_MAX_KB=$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))


mkdir -p "$CACHE"/{vllm,inductor,triton,aiter}

echo "[run] image=$IMAGE attn=$ATTN chunk=$CHUNK ar_max_kb=$AR_MAX_KB fast_draft=$FAST_DRAFT rerank=${RADIANCE_DRAFT_RERANK:-32} vhead=${RADIANCE_VERIFY_HEAD:-0} min_m=$MIN_M fuse_rms=${RADIANCE_FUSE_RMS_QUANT:-1} preshuf=${RADIANCE_PRESHUFFLE:-1} util=$GPU_UTIL kv_mem=${KV_MEM:-profiled} cache=$CACHE"

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
  -e RADIANCE_FAST_DRAFT="$FAST_DRAFT" -e RADIANCE_DRAFT_TAU="${RADIANCE_DRAFT_TAU:-0.20}" \
  -e RADIANCE_DRAFT_RERANK="${RADIANCE_DRAFT_RERANK:-32}" \
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
  ${PYTORCH_CUDA_ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
  -e RADIANCE_AR_OVERLAP_MIN_M="${RADIANCE_AR_OVERLAP_MIN_M:-2048}" \
  -e RADIANCE_AR_OVERLAP_SLICES="${RADIANCE_AR_OVERLAP_SLICES:-4}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" \
  -e RADIANCE_MXFP4_R4D_DECODE_MAX_M="${RADIANCE_MXFP4_R4D_DECODE_MAX_M:-0}" \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS="${RADIANCE_TOPK_TRITON_MIN_ROWS:-1}" \
  -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_DFLASH_CALIB="${RADIANCE_DFLASH_CALIB:-}" \
  -e RADIANCE_DFLASH_CALIB_TOKENS="${RADIANCE_DFLASH_CALIB_TOKENS:-200000}" \
  -e RADIANCE_MXFP4_HOIST_QUANT="${RADIANCE_MXFP4_HOIST_QUANT:-0}" \
  -e RADIANCE_RMS_QUANT_FUSION="${RADIANCE_RMS_QUANT_FUSION:-0}" \
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
    python3 patch_gdn_merge_inproj.py
    python3 patch_dynwidth.py
    # Non-fatal: fixes content=null on thinking-off requests; not required to serve.
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    # radiance_drafthead.py is copied too so RADIANCE_DRAFT_RERANK can be swept without an
    # image rebuild. The repo copy was byte-identical to the 0.9.3 one before that knob existed.
    cp radiance_mxfp4.py radiance_gdn.py radiance_rmsquant.py radiance_drafthead.py \
       radiance_verifyhead.py radiance_gdnmerge.py radiance_aroverlap.py "$SP"/
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
    --chat-template /root/.cache/huggingface/qwen-fixed-v22.3.jinja
