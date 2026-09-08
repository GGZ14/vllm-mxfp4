# Performance history and provenance

The headline numbers for the current build are in the
[README](README.md#performance). This document holds the longer form: what each landed change was
worth, how it was gated, the cross-image A/B that established the stack, and the gated-delta-net
overflow bug that made a pinned libr4d build necessary.

None of this is needed to run the server.

## Contents

- [Where the speed came from](#where-the-speed-came-from)
- [Two results worth carrying forward](#two-results-worth-carrying-forward)
- [ParoQuant against MXFP4](#paroquant-against-mxfp4)
- [Provenance: the 0.5.8 -> 0.7.4 baseline](#provenance-the-058---074-baseline)
- [The gated-delta-net NaN (fixed upstream)](#the-gated-delta-net-nan-fixed-upstream)

## Where the speed came from

Each row is a separate landed change with its own gate, measured against the build immediately
before it. This is not a decomposition of one A/B, and the rows are not additive: several move the
same wall time.

| Change | Measured | Gate |
|---|---|---|
| Decode launch-gap stack: traced quant, fp8 residual stream, fused AR epilogue (`db9dba6`) | **25.4 -> 22.66 ms/step**; decode launches 1477 -> ~1080/step; weighted single-stream +14%, conc-8 aggregate +24% | GSM8K 500q 97.8, paired sign test p=0.219; epilogue kernels bit-identical to the traced reference |
| Dynamic verify width (`f68d215`) | conc-8 steps 52-57 -> **46-47 ms**, aggregate 391-413 -> 444-461 t/s (**+11-13%**); single-stream a wash | Lossless by construction: speculative verification preserves the output distribution at any proposal length |
| fp8 QK + PV legs in prefill attention (`afa21b5`) | prefill 4442 -> 4648 t/s @ 40k, 3448 -> **3831 t/s @ 106k** (+10.8%); kernel-level 96.2 -> 150.5 TF at a hot 8k chunk | ppl 8.3708 -> 8.3707, top-1 54.09 -> 54.13%; GSM8K paired p=1.000. Upstream deleted these legs on *kernel* accuracy; the end-task gates say the error is free |
| KV cache group size by capacity, not smallest bucket (`1bf3914`) | **739,544 -> 892,799 KV tokens** (+20.7%); concurrency 2.82x -> 3.41x | Allocator-derived group size, unchanged for the n:1 layouts upstream targeted |
| Explicit KV pin over profiling | 892,799 -> **943,581 KV tokens** (+5.7%); 3.60x | Survives a full 260k-prefill sweep with no OOM. See [KV cache calibration](README.md#kv-cache-calibration) |
| int2 target verify head (`f882ea2`) | combined decode 170.0 -> **174.9 t/s** (+2.9%), all 8 categories +2.7-3.4%; conc 1/2/4 +2.8/+2.5/+1.6% | Equivalence, not a score: 24/24 seeded sampled completions byte-identical against a sequential self-consistency control |
| Epilogue store width T=512 at prefill M (`5950b38`) | 726 -> **685 us** at M>=2048 (405 -> 429 GB/s); TTFT @ 32k 7443-7878 -> 7196/7260 ms | GSM8K 97.60, in band (the 512-way striding changes per-row summation order); decode untouched and byte-identical |
| Fused GDN gated-norm + quant (`radiance::gdn_norm_quant`, `RADIANCE_GDN_NORM_QUANT=1`, default 2026-09-02) | single-stream 22.51 -> **22.32 ms/step** (-0.8%), 25.8 -> 25.1 @32k; one launch replaces two inductor kernels on each of the 48 linear-attention layers | GSM8K 500q 97.60%; BetterBench single-pass update p50 -0.2 ms in every category, tok/update neutral. Not bit-exact (silu 1 ulp), so judge on multi-prompt tok/update |
| Fragment-order weights + nontemporal decode loads (`RADIANCE_MXFP4_WPERM=1`, `RADIANCE_MXFP4_DECODE_NT=1`, defaults 2026-09-02) | single-stream 23.95 -> **22.66 ms/step** (-5.4%), 27.23 -> 25.65 @32k; BetterBench combined 195 -> 227 t/s single-pass | Acceptance byte-identical; GSM8K 500q 97.40%; prefill +0.3..+3.3% vs WPERM=0 (the A-tiled kernel is layout-neutral) |
| GDN `in_proj` single-GEMM merge (`588d5e6`) | single-stream 26.25 -> **25.50 ms/step** (-2.9%), -6.5% stacked with `WPERM=1`; removes 96 GEMM launches and 48 activation quants per forward | GSM8K 500q paired; drift is split-K reassociation only |
| GDN decode conv+recurrent fused into one launch (`9a84208`) | 25.01 -> **24.91 ms/step** (-0.4%) | **Bit-identical**: 4/4 byte-equal greedy completions |
| Decode band extended to M<=128 (`44d48f0`, opt-in at `MAXSEQS>8`) | conc-16 **549-622 t/s** at 75-79 ms steps, +30% over the pre-extension attempt | dks1 bit-identical to the folded tile at every shape and M in {72, 96, 127, 128} |

## Two results worth carrying forward

**`SPEC` depth is content-dependent, and tuning it on one content class picks the wrong default.**
A sweep on non-repetitive prose flipped the dflash default to 5. On BetterBench's weighted mix, back
to back on the same build, `SPEC=7` scores 184.3 t/s combined against `SPEC=5`'s 159.4 (+15.6%),
because code/json/file_edit run 4.7-6.0 tok/update at depth 7 and a cap at 5 truncates exactly the
high-acceptance tail the mix rewards. `SPEC=5` keeps the edge on prose-heavy content and on batch
throughput (conc-8 562 vs 544). Dynamic verify width mostly dissolves the trade-off. (`5692fed`)

**Fill the GPU before judging a tiling.** The epilogue prefetch above was justified by an M=2048
microbench on an idle, VRAM-squeezed GPU and shipped as a regression: at the real prefill shape
(M=8192) it measures 828 us against the original 726, **14% worse**, because its +40 VGPRs/thread
costs occupancy exactly when 8192 workgroups compete. (`eafcac9`, reverted in `5950b38`)

## ParoQuant against MXFP4

The second int4 format this stack serves, measured on the same box against MXFP4 production. The
path itself — format, kernels, knobs — is documented in [PAROQUANT.md](PAROQUANT.md); this is what
it is worth.

| | MXFP4 prod | ParoQuant | |
|---|---|---|---|
| GSM8K 500q | 97.8% | **97.4-98.0%** | per-token vs per-group activation scales; inside binomial noise |
| decode @ ctx25 | 22.3 ms/step | 24.19 ms/step | -7% |
| combined decode | 186.0 t/s | **226.2 t/s** | +22% |
| conc 1/2/4/8/16 | — | 168 / 276 / 399 / **512** / 518 | |
| prefill @ 2k/8k/16k/32k/64k | — | 3782 / 3700 / 3725 / 3621 / 3450 t/s | -8.8% at 8k when it shipped, parity from ~64k |
| KV cache profile | — | 622k tokens | |

Weight-side traffic is 4.25 bits/weight in both formats (ParoQuant's asymmetric zero point rides in
the same 4-byte load as the scale), which is why they land so close at the memory-bound end. The
gap that remained was launch count, not arithmetic: at ship time the rotation was 192 separate
launches per decode step, ~2 ms of a 26 ms step.

Where the ParoQuant speed came from, each measured against the build before it:

| Change | Measured | Gate |
|---|---|---|
| A-tiled prefill GEMM + fragment-order decode + prologue v2 | prefill 3220/3144/3148/3139/2984 -> **3789/3723/3668/3630/3427 t/s** (+15-18%); 24.27 ms/step (was 25.74, -5.7%) | GSM8K 500q 97.60%; harness bit-exact incl. the bf16 scratch rounding |
| Rotation stream 1 — residual add + RMSNorm + rotate + quant as one producer kernel | 24.27 -> **24.03 ms/step**; combined decode 186.0 t/s (= MXFP4 prod); KV profile 432k -> 479k tokens | GSM8K 500q **98.00%**; 96 rotation launches per step removed |
| MXFP4 weights + z-lab rotations (`paroquant_mxfp4`, 2026-09-08) | GEMM inner loop 16 VALU -> **0**; A-tiled prefill enabled (layouts identical). TP=1 eager A/B vs int4 PARO, same launcher: prefill **+9.8/+8.5/+6.8/+6.2%** at 2k/8k/16k/30k (2434 vs 2217 tok/s @2k), eager decode +27% (indicative); decode traffic unchanged (4.25 bits/weight both) | CHECKALL rel 0.00000 on all four gated shapes, every partition; loader at the e4m3 floor (0.011-0.012) at every band; one-shot pseudo GSM8K 96.96% (W4A16 bound), fine-tune in progress |
| Rotation stream 2 — silu-mul, gated norm, attention gate producers | 24.53 -> **24.19 ms/step** (-1.4%); combined **226.2 t/s**; conc-8 500 -> 512; KV profile 479k -> **622k tokens** | Harness 24/24 bit-exact vs the unfused chain; GSM8K 97.40% |

The KV profile moving 432k -> 479k -> 622k tokens across the two stream landings is worth noting on
its own: fusing producers removed intermediates from the compiled graph, and the capacity came back
as cache.

**Rejected: rotation stream 3**, the two-rank all-reduce fused into the norm+rotate producer. It is
*correct* — bit-exact single-rank loopback, an in-serve all-reduce check passing on every call on
both ranks, GSM8K 97.40, identical sanity completions — and still slower: 25.10 vs 24.19 ms/step,
202 vs 226 t/s combined. With one workgroup per row the rotation chains serialize on a single CU
and the push uses M CUs instead of r4d's 24 blocks, costing +7 us per site. Two earlier designs
deadlocked outright, in both cases because a spin-wait on peer flags outgrew residency or changed
its slice mapping with M. The MXFP4 `exact_nq` all-reduce wins with the same structure only because
its epilogue has no rotation to serialize.

One methodology note carried over from the MXFP4 work and re-earned here: **judge layout A/Bs with
short benches.** Under the fixed 210 W per-card cap a long run and a short run of the same build do
not measure the same machine, and a GSM8K comparison run against the wrong chat template moves the
score by two points regardless of the kernel — the 97-98% band needs `qwen-fixed-v22.3.jinja`.

## Provenance: the 0.5.8 -> 0.7.4 baseline

The cross-image A/B that established the stack, kept because it is the one clean comparison this box
can make and every number in the README is measured downstream of it. Both arms ran under
`SPEC_METHOD=mtp` at `SPEC=4`, the default at the time; the shipped default is now the `dflash`
drafter.

| | 0.5.8 | 0.7.4 | |
|---|--:|--:|--:|
| prefill 7.8k | 3873 | **4387** | +13.3% |
| prefill 26k | 3445 | **4138** | +20.1% |
| prefill 104k | 2310 | **3143** | +36.1% |
| prefill 182k | 1736 | **2511** | +44.6% |
| prefill 260k | 1393 | **2089** | +49.9% |
| decode short / medium | 63.0 / 67.4 | **67.1 / 67.5** | +6.5% / +0.1% |
| WikiText-2 PPL | 8.3335 | 8.3719 | +0.46% |

All 304 linear layers run the W4A8 fp8-WMMA kernel; `aiter` is not used at all now that
`RADIANCE_MXFP4_W4A8_MIN_M` defaults to 0. The prefill gain scales with context because it is mostly
R4D's paged attention, whose share of prefill grows with sequence length.

### The decode GEMM, measured against the same 0.7.4

`RADIANCE_MXFP4_DECODE_MAX_M`, default 64, toggled against the same build with it off.

| | Off | On | |
|---|--:|--:|--:|
| single stream, ms/step | 35.06 | **32.16** | -8.3% |
| ms/step at 32k context | 36.53 | **33.47** | -8.4% |
| aggregate tok/s, 4 concurrent | 170.1 | **218.5** | +28.5% |
| aggregate tok/s, 8 concurrent | 295.0 | **353.1** | +19.7% |
| prefill, all five lengths | -- | -- | unchanged (-0.3 to -1.2%) |
| GSM8K 500q, greedy | 97.80% | 97.80% | 3/3 discordant, sign test p=1.00 |

Batched workloads gain most because at M=20-40 aiter's tuned band uses `NUM_KSPLIT=1`, which leaves
the grid underfilled, while this kernel keeps split-K. GSM8K also ran **14% faster wall**
(375.5s -> 322.7s) on slightly *more* generated tokens.

## The gated-delta-net NaN (fixed upstream)

libr4d v0.4.0 produces NaN in the gated-delta-net output on this model: WikiText-2 PPL **653586**
with the W4A8 path and no mitigation. This is why `setup-mxfp4.sh` builds libr4d at a pinned commit
rather than using the one baked into the image.

### The three overflows

All three are the same shape: an unguarded `__expf` on an inactive lane or a split-form half,
giving `0 * INF = NaN`.

1. **`kkt_solve`, padding rows.** `gi` is forced to 0 for `i >= rows` while `gb[j]` keeps its real
   negative cumsum, so `d = -gb[j]` is large POSITIVE, the opposite of the "never positive"
   invariant the code asserts, which holds only for live rows. The NaNs land in padding rows of the
   64x64 tile, and the blocked inverse merges the whole tile with WMMA, so they reach live rows.
2. **`chunk_scan`, split-form halves.** `e^{g_i-c}.e^{c-g_j}` with `cref` at the chunk midpoint
   gives each half +/-(gate span)/2; a span past ~176 sends one to +INF and the other to 0.
3. **`chunk_scan`, `V'` staging.** The dominant one, and only visible across chunks. `V' = V.gv[t]`
   is staged in **bf16**, so `gv` must leave room for `V` under bf16's 3.4e38 ceiling. Clamping at
   `e^88` still NaNs; `e^80` leaves margin.

Clamp value, measured over 208,539 WikiText-2 tokens with no other mitigation:
70 -> 8.3841, **80 -> 8.3706**, 83 -> 8.3728, reference 8.3335, stock 653586. (Those were measured
before the fold was widened; on the current build the same configuration reads 8.3719, against
8.3736 with the original fold table.)

### The fix, and what the clamp does not fix

Fixed in **StillDeadcode/libr4d PR #1** (merged 2026-08-24). Not in a tag yet, which is why setup
builds libr4d at a pinned commit. The build is verified reproducible: it produces an `r4d.so`
byte-identical (sha256 `3026297b...`) to the one every number here was measured with.

The pin is deliberate. Nothing version-checks the library it loads, so a later commit that renames
an entry point or changes a compiled-in geometry constant makes `radiance_gdn.py` set
`ENABLED = False` and **fall back to the Triton path with one line on stderr**, costing performance
rather than raising.

**The clamp bounds the damage; it does not remove the cause**, and upstream sharpened this point
when merging. The original note here claimed the clamped product "evaluates to 0, which is the
correct answer". That is wrong: what leaves range is the distance from `cref`, not `g_i-g_j`, so on
a span-200 chunk the last token's own diagonal, and its `e^{gl-g_t} ~ 1` weight into the state which
the next chunk reads, are *attenuated* by `e^{80-(cref-g_t)}` rather than correctly vanishing. The
real fix is to stop splitting a weight that is provably <= 1 into a huge x tiny pair: stage `V'` in
fp32, or apply `e^{gl-g_t}` directly on the state path.

Fixing this also removed a second symptom: `RADIANCE_FAST_DRAFT` used to hang a worker at chunk 8192
because the draft head was being fed NaN like everything else downstream of the GDN core.
