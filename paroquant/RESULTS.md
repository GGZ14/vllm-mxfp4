# ParoQuant (PARO) W4A8 on gfx1201 — engineering log

**Goal:** serve z-lab/Qwen3.8-27B-PARO (ICLR'26 ParoQuant: int4 g128 asymmetric + learned
pairwise Givens rotations + channel scaling) through the radiance stack on the 2x R9700, on a
W4A8 path, at the best speed the format allows.

## Format (verified against the real checkpoint, not the paper)

Per projection: AWQ-packed `qweight [K, N/8]` / `qzeros [K/128, N/8]` (nibble order 0,2,4,6,1,3,5,7;
zeros are NOT offset-by-one) / `scales [K/128, N]` f16, plus `pairs [8, K]` i16 (Givens pair
indices, **local to each 128 group**, a permutation of 0..127 per group per layer),
`theta [8, K/2]` f16, `channel_scales [1, K]` f16 **stored pre-inverted** (multiply activations).
Zeros are genuinely asymmetric: 2..14, only 38% equal 8. Unquantized: visual tower,
`linear_attn.in_proj_a/b`, lm_head. No MTP/drafter tensors — the external DFlash2-FP8 drafter
carries over unchanged.

**Inference identity** (verified to 9e-16 on real tensors):
`y = ((x * channel_scales) R^T) dequant(Q)^T` — i.e. rotate+scale the ACTIVATIONS forward
(layer 0..7, within each 128-group), then a plain uint4-asym group GEMM. R^T on weights =
flipped layers with negated angles.

## Design (par_kernels.h, forked from ar_kernels.h)

1. **Zero point as a row-sum correction.** Keep the AutoRound (c-8) LUT; true value
   (c - zp) = (c-8) - d. Per group: `acc += asg * (sc*WMMA - (sc*d)*rowsum)`. `sc*d` ("zscale")
   rides in the same 4-byte load as the scale (SZ interleaved [G, N, 2] f16). Both a*c and d*a
   products are exact in fp32, so this only moves addition roundoff. Weight-side traffic:
   4 + 32/128 = 4.25 bits/weight — exactly MXFP4's.
2. **Per-group fp8 activation scales** (not per-token): lets the whole prologue be ONE kernel
   with no cross-workgroup sync, folds into the per-slab rescale that already exists, and is
   finer-grained fp8 than the per-token path (the paper's kernel is W4A16; this is the closest
   W4A8 gets). The epilogue As multiply disappears — the scale folds per slab.
3. **Fused prologue `pq_rotate_quant`**: rotate + channel-scale + amax + e4m3 encode + code-domain
   row-sums, one launch, replacing the old scaled_fp8_quant launch → net-zero added launches on a
   stack whose launch gap is ~20% of decode. One workgroup per (partition, group, token-chunk);
   rotation records ({i|j<<8, cos f16, sin f16} 8B) read once per chunk. Software e4m3
   encode/decode (RNE, OCP, saturating) so the row-sum is the sum of what the codes decode to BY
   CONSTRUCTION; gated exhaustively (256-code round-trip + nearest-value optimality).
4. **Partition select**: rotations are per projection, so merged linears (QKV, gate_up,
   in_proj_qkvz) carry A/ASG/RS with a leading partition axis; each n-block derives its partition
   from two boundary columns. One GEMM launch regardless. qwen3_5's in_proj_qkvz loads as 4 vLLM
   partitions (q,k,v,z) but q/k/v share the in_proj_qkv rotation — the loader dedups identical
   adjacent rotations into runs (in_proj → P=2, attn QKV → P=3).

## Gates passed

- Harness (run.sh): e4m3 round-trip + optimality exact; prologue bit-exact vs CPU reference
  (after matching `amax * (1/448.f)` rounding); decode+prefill GEMM rel ≤ 1.7e-3 (= bf16 output
  rounding) on all per-rank model shapes incl. 3-partition QKV and subnormal-heavy codes;
  rows past M untouched.
- Semantic identity on real checkpoint tensors: 9.3e-16.
- Module test (test_module.sh): CHECKALL (kernel vs same-codes fp64 reference) ≤ 4e-5 on all five
  layer families; vs EXACT activations rel ≈ 2.6e-2 — that is the e4m3 activation quantization
  itself (dot products do not average per-element fp8 noise down), same class as the AutoRound
  W4A8 path.
- In-serve CHECKALL (eval boot, eager): rel = 0.00000 on all five gated shapes, both TP ranks.
- Prod boot (MODE=prod): compiled + CUDA graphs captured (incl. dflash2 drafter graphs) around
  the custom op; spec decode live, early acceptance ~2.7 tok/draft.

## Traps hit (do not re-hit)

- Ubuntu ships `/usr/lib/python3.12/sitecustomize.py`; a sitecustomize dropped in site-packages
  is silently shadowed and never imported. APPEND to the stdlib one.
- The CHECKALL `_exact_ref` originally materialized int64 codes [N, K] (~700 MB on gate_up) and
  OOM'd a 0.92-util worker at runtime when a new M triggered it → chunked to 512 columns, fp32.
  CHECKALL remains an eval-boot tool; leave it off in prod.
- vLLM 0.27.1 requires >3-partition acceptance at create_weights (in_proj_qkvz) with dedup at
  process_weights — the GEMM's ≤3 limit applies to DISTINCT rotations.

## Numbers

Baselines: MXFP4 prod (Aug-27 sweeps): single-stream ~101.4 tok/s @ 28.63 ms/step, acc/draft
1.904; GSM8K 500q 97.8%; prefill ~3690 t/s @ 8K. AutoRound int4 kernel (same-conditions harness):
gate_up prefill M=2048 = 2046 us.

**PARO quality — GSM8K 500q (prod config, dflash SPEC=5, greedy): 98.0% (490/500), 1 truncated,
0 errors.** Above MXFP4 (97.8%) and the current prod tweak set (97.4-97.6%).

**PARO serve, first prod boot (pre fold-fix kernels):**
- decode vs ctx: 26.58 ms/step @ ctx25 (MXFP4 28.63), 28.94 @ 33k, 31.36 @ 102k (MXFP4 ~39),
  34.98 @ 208k (MXFP4 ~42.7). tok/s 100.3 / 102.9 / 89.8 / 84.3. acc/draft 1.67-1.98 —
  short-ctx acceptance below MXFP4's 1.90 (drafter/target mismatch, same drafter).
- BetterBench quick: conc-1 per-req 167.6 t/s med; conc-8 aggregate 457 t/s (MXFP4 ~434);
  conc-16 466. Prefill sweep 2247/2204/2169/2162/2044 t/s @ 2k/8k/16k/32k/64k — the one deficit
  (MXFP4 ~3690 @ 8k).

**Prefill kernel attribution + fix (harness --bench, gate_up N=17408 K=5120 M=2048):**
- v1 fold (per-element b32 LDS + 3-op chain per slab): 4797 us, 209 VGPRs / 7 waves.
- v2 fold (RS:=rowsum*asg in prologue; per-fragment float4 asg/rsa hoist; 2 VALU/elem/slab +
  1 FMA/elem/GROUP correction; stage on even slabs only): **2910 us, 144 VGPRs / 10 waves**
  (-39%). Decode unchanged (gate_up M=8 ~60 us).
- Ablation (skip fold+staging, ABLATE bit 4): 2272 us → fold now costs 22% of the kernel; the
  ablated kernel is within 11% of AutoRound (2046), so the structure is right.
- Rotate-quant prologue: M=8192 K=5120 P=3: 1.64 ms (~10% of a prefill chunk). P=1: 0.57 ms.
- PER-TOKEN PREFILL (v3, BUILT): pass A (`pq_rotate_quant<ROTOUT>` → bf16 XR + group amaxes),
  pass C (`pq_token_quant`: As = max_g asg, encode, plain row-sums), `PTOK` GEMM template
  (AutoRound-cost fold, correction once per group, As in the epilogue) + sc/zsc carried in
  registers across the group's two slabs (half the scale loads either variant paid before).
  Dispatch: M > RADIANCE_PQ_DECODE_MAX_M (64) → per-token; decode band keeps the fused
  per-group single-launch path. gate_up prefill M=2048: **2499 us** (1.22x AR), 128-135 VGPRs /
  10 waves. All shapes -13%. Serve prefill 3156 → **3367 t/s @ 8k** (-8.8% vs MXFP4), 2855 @
  104k, 2207 @ 260k — parity-or-better vs MXFP4 from ~64k depth. Decode untouched (25.74
  ms/step). Harness gates: ptok-prologue bit-exact (incl. mirroring the bf16 scratch rounding),
  PTOK GEMM at the bf16 floor on all shapes.
- TN=4 prefill arm: dead end, do not revisit — AR's sweep closed it (256 VGPRs of accumulator,
  spills).
- MEASUREMENT TRAP: `VAR= cmd` (set-but-empty) makes getenv() return non-NULL — an ablate arm
  gated on bare getenv() silently ran in a "clean" bench. Gate on `abf && *abf`.

## Final state (2026-08-31, PTOK build serving)

- GSM8K 500q: 97.4% (487/500) per-token / 98.0% per-group — both ≥ the prod tweak band
  (97.4-97.6) and the delta is inside binomial noise; RADIANCE_PQ_PTOK=0 is the rollback lever.
- BetterBench quick (final): conc-1 per-req 169.8 t/s med, conc-8 aggregate **466.9 t/s**
  (MXFP4 ~434, +7.6%), conc-16 476.2; TTFT p50 conc-8 155.7 ms. Prefill sweep
  3128/3083/3120/3099/2954 @ 2k/8k/16k/32k/64k — above MXFP4 at 64k, −10..−16% below at ≤32k.
- bench_prefill_clean: 3367 @ 8k, 2855 @ 104k, 2207 @ 260k.
- bench_decode_ctx: 25.74 ms/step @ ctx25 (MXFP4 28.63), 34.10 @ 207k (MXFP4 ~42.7).
- Serving: `MODE=prod SPEC=5 ~/mxfp4_work/paro/run_paroquant.sh` (container vllmparo, id
  Qwen3.8-PARO). MXFP4 prod restore: `podman start vllmmxfp4074`.
