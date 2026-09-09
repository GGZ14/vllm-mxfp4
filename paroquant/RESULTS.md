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

## 2026-09-02: parity pass with the MXFP4 stack (A-tiled prefill, fragment-order decode, prologue v2)

BetterBench single pass on the 08-31 build, measured this afternoon against the MXFP4 prod of the
same day (which had gained the A-tiled prefill GEMM, WPERM+NT decode, fused GDN norm and the
v22.3 template since Paro shipped): prefill **3220/3144/3148/3139/2984 t/s @ 2k/8k/16k/32k/64k
vs 4883/4955/4854/4725/4480 (-33..-37%)**; update p50 25.7-26.2 ms vs 22.3; conc 1/2/4/8/16
aggregate 140/242/357/456/471 vs 163/280/405/529/536. `results/paro-single-0902-baseline.json`.

Ported, all harness-gated (`run2.sh` builds once; `--quick` skips the CPU GEMM reference;
`--bench2 rot|passc|pre|dec|wp` are DRAM-fed ABAB benches with rotated weight copies -- the old
`--bench` keeps ONE 44.6 MB weight inside the 64 MB Infinity Cache and reads it at 1170-1280 GB/s):

1. **A-tiled per-token prefill GEMM** `pq_int4_fp8_gemm_atiled<TN, WPERM, LBK, WHOIST>`: pass C
   (`pq_token_quant_tiled`) writes the MXFP4 fragment-tiled layout, the GEMM loads A fragments
   straight into the WMMA registers, W alone goes through LDS. LBK=128 makes the slab the scale
   group: sc/zsc once, temp accumulator over 8 WMMA steps, fold + zero-point correction once per
   128 k. 206 VGPRs / occupancy 7 (hoisted W fragments), no spills.
   vs the shipped PTOK kernel, best-of-4 DRAM-fed, TF/s in parentheses:

   | shape M=2048 | ptok-row | at64 hoist | **at128 hoist** | at128 nohoist |
   |---|--:|--:|--:|--:|
   | qkv     | 1019 (148) | 0.99 | **0.83 (178)** | 0.86 |
   | o_proj  |  450 (143) | 0.91 | **0.79 (178)** | 0.85 |
   | gate_up | 2458 (149) | 0.99 | **0.85 (176)** | 1.10 |
   | down    | 1239 (147) | 1.01 | **0.83 (177)** | 0.87 |
   | in_proj | 1172 (147) | 1.00 | **0.84 (175)** | 0.88 |

   M=512: 0.72-0.78; M=1024: 0.79-0.82; M=4096: 0.85-0.86. at128+hoist wins every cell;
   LBK=64 (32 fewer VGPRs, 2x barriers and folds) is a wash; nohoist loses 3-27%.
   Serve: `RADIANCE_PQ_ATILED=1` (default), `RADIANCE_PQ_AT_LBK=128`, `RADIANCE_PQ_AT_HOIST=1`.
2. **Pass C tiled**: first cut (lane reads 16 B from sixteen rows per k-step) was 1.3-2.0x the
   row pass. Rewritten as row-contiguous reads + 16x128 LDS tile at a 136 B stride + 256 B
   fragment stores, groups split across gridDim.z so small M fills the GPU: 0.85x at M=64,
   0.96-1.02x at M=2048, 1.09-1.17x at M=8192 (+45 us/linear against ~1.5 ms of GEMM saved).
3. **Fragment-order weights + streaming loads at decode** (`RADIANCE_PQ_WPERM=1`,
   `RADIANCE_PQ_DECODE_NT=1`, both default; `pq_stage_w<...>` is one staging helper for both
   layouts, all three GEMMs): decode M=1/5/8/16/40/48: qkv 0.87/0.89/0.88/0.87/0.84/0.85,
   o_proj 0.83/0.81/0.80/0.80/0.84/0.89, gate_up ~0.9 to M=48, down 0.90/0.86/0.88/0.87/0.82/0.82,
   in_proj 0.89/0.88/0.88/0.89/0.85/~0.85; M=64 neutral (as on MXFP4). Bit-identical to the row
   layout (same sW tile). The tiled prefill kernel is layout-neutral within +-2-4% (`--bench2 wp`:
   gate_up 1.005-1.019 at WSLOTS=2, 1.002-1.045 at 4; qkv 0.95-1.01; down 0.99-1.03) -- an earlier
   +22% reading came out of the long sustained `pre` sequence (power cap), not the kernel. The
   BK=64 row-major prefill kernels DO lose 25% under WPERM; they are fallbacks only now.
4. **Rotation prologue v2** (`pq_rotate_quant2`, `RADIANCE_PQ_ROT_V2=1` default): the lane's 16
   pair records live in registers, no record LDS fill, no __syncthreads, 4+4 LDS ops per layer
   instead of 10+4. Bit-exact vs v1 (gated, both modes). Decode M=5-8: 0.91-0.95 (the rest is
   launch overhead); M=40-64: 0.79-0.87; prefill pass A M=2048-8192: 0.67-0.70.
5. **Fused GDN update** (rx5 libr4d, `RADIANCE_GDN_FUSED_UPDATE=1`, merge hook installed with
   the merge itself OFF -- run_autoround.sh has run this way since 08-30; the "fp8-linear-only"
   note above was wrong): "all-R4D decode(fused) path live" on both ranks.
6. Decode band extended to M<=128 (PQ_DEC_MAX_TM 8, AutoRound's M>64 split-K rule);
   RADIANCE_PQ_DECODE_MAX_M stays 64 in the unit until conc-16 is re-swept.

Gates: harness correctness PASS (0 failures: every atiled instantiation at the bf16 floor on
all shapes x 14 Ms, WPERM decode/prefill byte-identical, prologue v2 and tiled pass C bit-exact),
module test PASS on the real checkpoint (CHECKALL rel <= 3e-5 decode band, <= 4e-5 tiled),
7k-token prompt answered correctly in serve.

**Served 2026-09-02** (`vllm-switch paro` = unit -> run_paroquant.sh, all new knobs at their
defaults, cache `~/.radiance-cache-paro-093-fu`). Gates back to back on the live server:

- GSM8K 500q (greedy, conc 8, qwen-fixed-v22.3): **97.60% (488/500), 1 truncated, 0 errors**
  (08-31 build: 98.0 per-group / 97.4 per-token; inside binomial noise).
- bench_decode_ctx GEN400: **24.27 ms/step @ ctx25** (was 25.74, -5.7%), 25.80 @ 8k, 26.91 @ 32k,
  28.80 @ 105k, **32.14 @ 206k** (was 34.10, -5.7%); acc/draft 1.74-1.91 unchanged.
- BetterBench single pass (`results/paro-single-0902-new.json` vs `-baseline.json`, same day):

  | | 08-31 build | today | MXFP4 prod (09-02) |
  |---|--:|--:|--:|
  | prefill PP t/s @2k/8k/16k/32k/64k | 3220/3144/3148/3139/2984 | **3789/3723/3668/3630/3427** (+15..+18%) | 4883/4955/4854/4725/4480 |
  | update p50 (ms) | 25.7-26.2 | **24.4-24.7** | 22.3 |
  | combined decode t/s | 176.6 | **184.0** | 186.0 |
  | conc 1/2/4/8/16 aggregate | 140/242/357/456/471 | **150/270/382/506/496** | 163/280/405/529/536 |
  | TTFT p50 single | 79 ms | 69 ms | |

  Gap to MXFP4 prod: prefill -25% (was -37%), decode step -9% (was -15%), conc-8 -4%
  (was -14%), combined decode -1%.

**What is left, priced from today's ledger.** Prefill: the GEMM is now at 175-182 TF/s against
MXFP4's tiled 215-220 -- the remaining 20% is the per-slab scale fold (a temp accumulator and
2 FMA/elem/128k that MXFP4 folds into the weight bytes) plus the rotation prologue (pass A+C,
~8% of a chunk after v2); TN=4 needs a register budget this fold does not leave. Decode: the
2 ms/step to MXFP4 is the rotation launch (192 x ~4 us kernel time; fusing it into the norm
producers is the next lever), the GDN in_proj merge + norm-quant fusion (fp8-linear-only),
and drafter acceptance (1.74 vs 2.07 on the same drafter).

## 2026-09-02/03: rotation stream (fused add + RMSNorm + rotate + quant producers)

Decode-band producer for every norm-fed linear as ONE kernel: `pq_add_rms_rot<ROT>`
(residual add + Gemma RMSNorm in vLLM's exact op order + channel scale + rotation + per-group
quant; one workgroup per (row, 8-group chunk, partition), records in registers, gpw=1 chain
per wave to M=40 and 2 above). `radiance::pq_add_rms_rot` -> (hs, residual, A, ASG, RS);
`radiance::paroquant_linear_pre` consumes the tuple in the decode band and takes the tiled
prefill path from hs above it (the M branch stays inside the ops -- no dynamo guard).
Patched decoder-layer forward (mirror of radiance_arnq._stream_forward) + a tuple-aware GDN
forward_hip (in_proj_ba keeps the bf16 hidden); installed from radiance_gdnmerge.merge_model;
`RADIANCE_PQ_ROT_STREAM=1`, cache suffix `-rs`. 64 input + 64 mid epilogues over 64 layers.

Kernel (DRAM-fed `--bench2 fuse`, M/P, fused vs norm-kernel + pq_rotate_quant2 chain):
M=1-8: 5.0-5.2 vs 8.1-9.0 us (0.59-0.63); M=16: 5.3-6.9 vs 9.7-9.9; M=40: 6.5/8.8/11.9 vs
13.0-13.6 (P=1/2/3); M=64 (gpw 2): 7.6/11.6/15.2 vs 16.4-17.6. Gates: harness fused ==
chain bit-exact at every M x gpw; vs CPU reference residual exact, hs <= 20 ppm; module test
vs vLLM GemmaRMSNorm + bf16 linear: bit-identical at M<=17, hs 1-7 per M at 40-300 (ppm),
out-rel <= 5e-4.

Serve (vs the fused-GDN build of the same day): bench_decode_ctx 24.27 -> **24.03 ms/step**
@ctx25, 25.80 -> 25.52 @8k, 26.91 -> 26.31 @32k, 28.80 -> 28.73 @103k, 32.14 -> 31.90 @206k;
BetterBench single pass update p50 24.4-24.7 -> **24.2-24.5 ms**, combined decode 184.0 ->
**186.0 t/s (= MXFP4 prod)**, conc 1/2/4/8/16 152/263/398/500/506 (noise-level vs before);
**GSM8K 500q 98.00% (490/500)**, 1 truncated. KV cache profile 432k -> 479k tokens (smaller
compiled-graph activation footprint). First cut cost prefill -1..-1.5% (3789/3723/3630/3427 ->
3753/3671/3587/3377): the plain-norm fallthrough re-read the row for its second pass; fixed by
carrying the row in registers (radiance_add_rms_quant's shape). Re-check below.

Gap to MXFP4 prod after this: decode step 24.0 vs 22.3 (-7%), combined decode equal, conc-8
500 vs 529 (-5%), prefill -25%. Remaining decode ledger: silu_mul -> down (48 rotations),
GDN gated norm -> out_proj (48), attention out -> o_proj (16) still launch the standalone
prologue; GDN in_proj merge + fp8-only norm-quant fusions; drafter acceptance 1.5-2.1.

TRAP (cost an hour): the harness gate launched the fused kernel with a hard-coded gpw=1 while
sizing the grid for 2/4, and the un-cleared ASG/RS/HS buffers from the gpw=1 pass masked the
unwritten groups -- "codes wrong, scales right" was a harness bug, not a kernel bug. Clear
EVERY output between variants of a gate.

**Re-check after the register-carry fix (2026-09-03, BetterBench prefill+decode single pass,
`results/paro-0902-rs2-predec.json`):** prefill 3772/3702/3687/3648/3450 @2k/8k/16k/32k/64k
(pre-stream 3789/3723/3668/3630/3427: parity, +-0.5%); update p50 24.0-24.4 ms. Served config.

## 2026-09-03: prefill GEMM ablation ledger (what the fp16 group scale costs, and what does not help)

`--bench2 abl` (DRAM-fed, order rotated per rep, best of 4), TN=2 LBK=128 fragment-order, M=2048:

| variant | qkv | in_proj | gate_up |
|---|--:|--:|--:|
| shipped at128 (scale FMA + zero-point FMA per tile-group = 16 VALU) | 180 TF/s (1.00) | 179 (1.00) | 184 (1.00) |
| no fold, temp accumulator kept (8 VALU) | 200 (0.90) | 199 (0.90) | 200 (0.92) |
| accumulate straight into the WMMA output (0 VALU, the MXFP4 loop) | 224 (0.80) | 224 (0.80) | 225 (0.81) |
| zero point out of the loop, fp16-WMMA epilogue product (ZPE, 8 VALU + epilogue) | 186 (0.97) | 183 (0.98) | 150 (1.22) |
| ZPE loop alone, no epilogue | 194 (0.93) | 193 (0.93) | 152 (1.20) |

Reading: the cost is LINEAR in VALU per tile-group (each 8 ops ~10%); the WMMA path is the
same fp8 stream as MXFP4 and reaches MXFP4's number the moment the fold is gone. Measured and
REJECTED on the way: (a) in-place rescale (acc *= s[g-1]/s[g], WMMAs write acc directly, zero
point as an integer FMA) -- 16 VALU like the shipped kernel, no gain, plus the gate_up penalty;
(b) the zero point as a rank-G epilogue product -- 4% for the epilogue on top of the 7% loop
gain, and the loop variant without the zero-point FMA shows a +20% schedule pathology on the
widest shape (same binary, 0.93 on qkv/in_proj, 1.20 on gate_up; not order, not the operand
reads -- LDS-staged zero-scales did not move it). Left as ABL bit 4 for the bench only.
Also rejected earlier today: TN=4 (register budget with the fold), pass A+C fusion (records
re-read per row), chunk-size changes (the fold is per element, not per chunk).

Conclusion: with fp16 group scales the kernel is at 180-188 TF/s against a 225 TF/s loop;
the remaining 20% needs power-of-two (e8m0) scales folded into the weight bytes, i.e. a
re-quantized checkpoint (ParoQuant toolchain scoped: layer-wise optimizer, pow2 constraint is a
few lines in UniformAffineQuantizer, CUDA rotation kernel JIT-builds via cpp_extension --
untested on ROCm; bf16 base model 55.6 GB downloaded to ~/models/Qwen3.8-27B-bf16). On hold.

## 2026-09-03: SPEC re-sweep on the rotation-stream build

`spec_sweep.sh` (manual serve per SPEC, bench_decode_ctx 0/8k/32k + BetterBench decode single
pass; SPEC=5 = the unit, numbers from the same-day gates):

| SPEC | ms/step ctx0 / 8k / 32k | tok/s ctx0 / 8k / 32k (greedy, 1 prompt) | BetterBench combined t/s (8 prompts, temp 0.7) |
|--:|--:|--:|--:|
| 5 | 24.03 / 25.52 / 26.31 | 105.6 / 122.4 / 111.0 | 184-186 |
| 6 | 24.42 / 26.39 / 27.06 | 113.7 / 119.9 / 102.6 | 198 |
| 7 | 24.53 / 26.31 / 27.03 | 104.9 / 109.9 / 108.8 | 204 (weighted from the rows) |

Single-stream: 7 > 6 > 5 by ~+10% combined -- the extra tokens per update outweigh the +2%
step. The one-prompt greedy tok/s columns are trajectory noise (the memory's warning about
judging by acc/draft on one prompt applies). Concurrency check at SPEC=7 vs today's SPEC=5
(152/263/398/500/506 aggregate at conc 1/2/4/8/16) follows before the unit changes.

**SPEC=7 chosen (2026-09-03):** concurrency at SPEC=7 (stream 1) 166/274/410/503/502 vs SPEC=5
152/263/398/500/506 -- equal or better at every level, +10% single-stream. Unit updated.

## 2026-09-03: rotation stream 2 (silu-mul, GDN gated norm, attention gate fused with rotate+quant)

`pq_ew_rot<MODE, ROT>`: one wave per 128-group producer (no row reduction) + the rotate_quant2
body; MODE 0 silu(g)*u (bf16-rounded silu, bf16 product), MODE 1 x*sigmoid(gate) (eager
rounding), MODE 2 per-head RMSNormGated (head = group, fp32, ((x*rsqrt)*w)*silu(z) rounded once).
Hooks: mlp.act_fn -> tuple into down_proj; linear_attn._output_projection -> tuple into
out_proj; Qwen3NextAttention.forward tail -> tuple into o_proj. 64 + 48 + 16 sites.
`RADIANCE_PQ_ROT_STREAM2=1` (default), cache suffix `-rs2`. The partition-count guard in
`_linear_impl` exists because the module test once fed a single-partition tuple into a
2-partition layer and the GEMM read past the activation buffer -- a silent GPU hang, not a fault.

Gates: harness `ewrot` 24/24 bit-exact vs the unfused chain and matching the CPU reference
(mode 2 at 1 ppm); module test on the real down_proj: modes 0/1 bit-identical to torch at every
M, mode 2 within 5-26 flips per million above M=40; serve sanity (17*23, 7k-token prompt) OK.

Serve (unit: SPEC=7 + stream 2, same day, vs SPEC=5 + stream 1):
- bench_decode_ctx: 24.19 ms/step @ctx25 (SPEC=7 alone 24.53; stream 2 = -0.34 ms = -1.4%),
  25.74 @8k, 26.70 @32k, 28.71 @103k, 32.55 @206k.
- BetterBench single pass: update p50 24.3-24.7 ms; combined decode **226.2 t/s** (was 186.0;
  MXFP4 prod 186.0); conc 1/2/4/8/16 **168/276/399/512/518** (was 152/263/398/500/506);
  prefill 3782/3700/3725/3621/3450 @2k-64k (unchanged); KV cache profile 479k -> **622k tokens**
  (fewer intermediates in the compiled graph).
- GSM8K 500q: **97.40%** (487/500), 1 truncated.

## 2026-09-03: rotation stream 3 (two-rank all-reduce fused into the norm+rotate producer) -- REJECTED

`pq_ar_add_rms_rot`: the r4d one-shot P2P push/flag/reduce protocol with the fused norm +
rotate + quant epilogue, own IPC scratch/flags/counters. Three designs on the way:
1. (row, chunk, partition) grid with gpw keyed on M: DEADLOCK at capture size 12 -- the peer-flag
   wait compares the peer's per-slice sequence numbers against this workgroup's own, which only
   works when every slice index of a row has the same launch history; a mapping that changes with
   M breaks that.
2. (row, chunk) grid with a fixed mapping: DEADLOCK at capture size 8 -- the grid outgrew
   residency and resident workgroups spun on peer slices whose pushers were never scheduled (the
   classic spin-wait deadlock; r4d's kernels are one block per row for exactly this reason).
3. One 512-thread workgroup per row, 16 waves splitting the groups and looping the partitions:
   correct (single-rank loopback bit-exact vs pq_add_rms_rot at M=1/5/40, reduction-order ulps
   at 64; in-serve AR_CHECK: fused == vLLM AR + plain kernel on every call, both ranks; GSM8K
   97.40%; sanity prompts identical) but SLOWER: 25.10 ms/step ctx0 vs 24.19 with stream 2,
   26.79 vs 25.74 @8k, update p50 25.4-25.7 vs 24.3-24.7, combined 202 vs 226 t/s, conc-8 506
   vs 512. +7 us per site: with one workgroup per row the rotation chains (2-3 groups x up to 3
   partitions per wave) serialize on one CU, and the push uses M CUs instead of r4d's 24 blocks.
   The MXFP4 exact_nq wins with the same structure because its epilogue has no rotation.
   Restoring the parallelism means either the deadlock risk of design 2 or a two-launch split
   (push kernel + wait/rotate kernel), which gives back the launch saving that was the point.
Also found on the way: an installer bug (the per-layer flag init ran AFTER the previous
iteration granted the next layer's fused-input flag, so every layer but the last normalised an
un-reduced partial -- semi-coherent, looping output, bit-identical fused/unfused checks). Fixed;
the pattern (initialise all flags in a pre-pass) is worth remembering.
`RADIANCE_PQ_ROT_STREAM3` stays default 0; kernel, launcher, op, check mode (`RADIANCE_PQ_AR_CHECK`)
and fallback (`RADIANCE_PQ_AR_FALLBACK`) remain in the tree, dark.

**Served config after today:** SPEC=7, stream 1 + 2 on, stream 3 off, cache `-fu-rs-rs2`.

## 2026-09-08: MXFP4-PARO prod decode -- the launch-count regression and its three fixes

First prod boot of `paroquant_mxfp4` (compiled, dflash SPEC=7, TP=2): **35.40 ms/step** @ctx25,
80 tok/s single stream, BetterBench combined 126.3 t/s -- vs int4 PARO 24.19 / 105-114 / 226.2.
Ruled out first: acceptance (acc/draft 1.85 vs int4's 1.57-1.78 on the same bench_decode_ctx) and
the GEMM (`bench_linear_tp2.py`, real TP=2 shapes, M=8: int4 linears 12.56 ms/step, MXFP4 two-pass
prologue 18.96, MXFP4 fused prologue 12.01). The whole gap was launch count.

1. **`pq_rotate_tokquant`**: channel-scale + rotate + token amax + e4m3 encode in one launch, the
   rotated row parked in dynamic LDS between the two phases (the per-token scale needs the whole
   row). Byte-identical to pass A + pass C (`--bench2 tokq`, 54 shapes; max_g(amax_g)/448 ==
   max_g(amax_g/448) exactly). Prod 35.40 -> **28.48 ms/step**, combined 126 -> 184 t/s.
2. **Per-token stream producers** `pq_add_rms_rot_tok`, `pq_ew_rot_tok<0|1|2>`: the stream-1/2
   producers emitting (A [P,M,K], AS [P,M]) instead of the int4 per-group tuple. One workgroup per
   (row, partition), W waves split the groups, block-reduce the token amax, encode. Byte-identical
   to [plain producer -> hs] + `pq_rotate_tokquant(hs)` (`--bench2 tokstream`, 45 cases: hs, ro,
   A, AS all 0 diff). `install_stream` generalized: it asks the consumer's quant method for
   `stream_norm` / `stream_ew`, so both loaders share the patched forwards; stream 3 stays
   int4-only. Installed 64/64 epilogues + 64/48/16 stream-2 sites on both ranks. Prod 28.48 ->
   **26.29 ms/step**, combined 200.3 t/s, GSM8K 500q 97.40%.
   Waves per row: the small-M cost is the serial rotation chain per wave (K=5120: 40 groups = 5
   chains at 8 waves). Measured 8/16/32 (harness, idle GPU): norm site M=8 10.5 / 8.3 / 7.9 us,
   down-proj producer (N=8704) 17.2 / 12.0 / 9.7, o_proj producer (N=3072) 7.7 / 6.3 / 5.6; at M=64
   16 wins, at M=128 8. Rule `M<=16 -> 32, M<=64 -> 16, else 8` (`RADIANCE_PQ_TOK_WAVES` forces).
   Still 2-4 us/site behind the int4 producers (5.4 / ~6 us): interleaving two chains per wave
   (independent LDS latency chains) is the untried lever.
3. **Single-launch merged GEMM**: `radiance_mxfp4_fp8` decode / folded / A-tiled kernels take
   `pb1, pb2, astride`; an n-block in [pb1, pb2) shifts A and As to rotated copy 1, etc. Stock
   `launch` passes 1<<30 (bit-identical to before); `launch_p` / `launch_at_p` are the merged
   entries; `mxfp4_linear_pqp` the op. Replaces P GEMM launches + P slice copies (or a cat) per
   merged linear. Not bit-identical to the loop -- the decode band picks split-K from the launch's
   N -- 1-2 one-ulp flips per million outputs (loader test). M=8 real shapes: qkv P=3 51.7 -> 28.6
   us, gate_up 67 -> 59, in_proj 40 -> 30, P=1 sites 27 -> 23 (no copy); per-step linears
   10.9 -> **9.2 ms** (int4 12.7). Prod 26.29 -> **24.53 / 25.89 / 26.80 ms/step** @ctx25/8k/32k
   (int4 24.19 / 25.74 / 26.70), combined 203.4 t/s, GSM8K 500q 97.40% (487/500, 0 errors).

Prefill (TTFT proxy on the same decode benches, 8k / 32k prompts): int4 3850 / 3630 tok/s;
MXFP4-PARO final 4524 / 4245 (**+17%**) -- above the +6-10% the TP=1 eager kernel A/B gave, since
the stream and the single launch also take launches out of the prefill step.
BetterBench prefill sweep, single pass, final build: **4376/4449/4423/4291/4068 PP t/s @2k/8k/16k/32k/64k** vs int4 PARO's 3782/3700/3725/3621/3450 (+16%/+20%/+19%/+18%/+18%). `bench_prefill_clean_m`: 4528 @8k, 4349 @26k, 3545 @104k, 3013 @181k.

Housekeeping: the duplicate per-partition scale slabs (`ws_cat`, ~0.4 GiB/rank) are gone -- the
single launch reads the full [K/32, N] and the A/B loop slices on the fly; KV profile had dropped
698k (v1) -> 626k (stream) -> 578k (single) tokens at GPU_UTIL 0.92 while int4 sits at 622k, so the
unit now runs GPU_UTIL 0.95 (AMD MXFP4 runs 0.98): KV profile **862,604 tokens** (int4 622k, 3.29x
concurrency at 262k), decode unchanged (24.71 / 25.98 / 26.96 ms/step on that boot). Cache dir keyed
`-rs-rs2-sl`. SPEC not re-swept:
step time and acceptance match int4 PARO, whose sweep chose 7.

**Lesson:** measure prod decode step time BEFORE shipping a loader; prefill A/B + GSM8K said "ship"
while decode was 46% slower. And under a hipGraph the CPU is hidden but every kernel is not: a
2-3 us node x 400-500 extra nodes per step is the entire gap.

## 2026-09-08 (late): fused TILED prologue for prefill; chain interleaving measured neutral

**Fused tiled prologue (shipped).** Above the A-tiled threshold the MXFP4-PARO linear ran pass A
(rotate -> bf16 row, 460 us per partition at M=8192 K=5120) + tiled pass C (re-read the row, token
scale, encode into the fragment-tiled slab, 250 us). Rotating twice to avoid the round trip would
lose (pass A is compute/LDS-bound, not memory-bound: 168 MB moved in 460 us = 365 GB/s). Instead
`pq_rotate_tokquant<W, TILED>` keeps one workgroup per row, parks the rotated row in LDS, and writes
the fragment-tiled layout directly: each lane's four codes are one aligned 4 B piece of an 8 B
fragment chunk (`pq_tiled_off`), so a wave store scatters 32 x 4 B over 16 fragment lines that the 15
neighbouring rows' workgroups fill in -- L2 write-combining absorbs it. `--bench2 tokqt` (24 shapes,
K=5120/8704, M=64..8192, P=1..3): AT and AS byte-exact vs pass A + `pq_token_quant_tiled`;
**1.3-1.4x faster** at prefill M (M=8192 K=5120 P=2: 1687 -> 1279 us; K=8704 P=2: 3001 -> 2280 us).
The stream producers got the same TILED variant, so in the A-tiled band they hand the consumer the
tiled tuple directly and the linear runs zero prologue launches; producer and consumer take the same
`_tiled(M)` decision. Waves-per-row rule gained a K term (K>=8192 wants 16 above M=64).
**Prod (TP=2, compiled, SPEC=7, 8192 chunk): BetterBench prefill sweep 4770/4827/4649/4495/4273 PP t/s @2k/8k/16k/32k/64k**
-- +9%/+8%/+5%/+5%/+5% over the single-launch build (4376/4449/4423/4291/4068) and +26%/+30%/+25%/+24%/+24% over int4 PARO
(3782/3700/3725/3621/3450). Decode unchanged: 24.53 / 26.43 / 26.97 ms/step @ctx25/8k/32k. KV 850k.

**Two-chain interleaving (NOT shipped, kept dark as template `IL`).** Hypothesis: the per-token
producers trail int4's (7.9 vs 5.4 us norm, 9.7 vs ~6 down-proj at M=8) because each wave runs its
rotation chains serially; issuing two groups' LDS read/fma/write pairs per layer before one waitcnt
should hide half the LDS latency. Built for all three per-token kernels (`pq_tok_rotate_park2`),
byte-identical to the one-chain kernels at every W (tokstream `il=0`). Measured: at the wave counts
the launcher uses it is neutral -- norm w32 7.89 -> 7.93 us, ew N=3072 w32 5.63 -> 5.62, ew N=8704
w32 9.68 -> 9.65; only the unused w8 configs gain (norm 10.5 -> 10.1, ew N=8704 17.0 -> 15.4), and
at M>=40 it loses 5-20%. So the remaining producer cost at W=32 is not chain latency; it is the
phase structure the per-token scale forces (row pass -> barrier -> chains -> barrier -> LDS re-read
+ encode) against the int4 producer's single register-resident pass. ~2-2.5 us x 256 sites =
~0.5 ms/step; parked.

## 2026-09-09: DFlash2 drafter fine-tuned on the MXFP4-PARO target (self-distillation)

The remaining decode gap to int4 PARO is acceptance, not step time. No public DFlash2 training code
exists (z-lab's repo is inference-only), so the loop was written against vLLM's own forward
(`paroquant/drafter/train_drafter.py`), validated by reproducing the original drafter's per-position
top-1 on real captures. Recipe (DFlash paper): CE on the 7 mask positions weighted exp(-(k-1)/4),
random anchors per sequence, target embed / lm_head and the candidate selector frozen, fp32 master
+ AdamW offloaded to the CPU (1.8B trainable params on one R9700, 7 s/step).

Data: 2,400 prompts (1,200 ultrachat_200k, 700 CodeAlpaca, 500 GSM8K-train) answered by the served
MXFP4-PARO target (prod sampling, reasoning on, <=1024 tokens), captured in-serve by
`radiance_dflash_capture.py` (aux hidden states of layers 5/19/33/47/61 as e4m3 + per-token scale,
rejected draft slots trimmed): 2,410 sequences, 1.62M completion tokens, 44 GB. 2 epochs, lr 5e-5.

Held-out proxy (96 seqs, prefix-expected accepted/block): **1.979 -> 2.041** (+3.1%); weighted CE
2.00 -> 1.71; top-1 by position 0.807/0.687/0.586/0.523/0.462/0.415/0.372 ->
0.819/0.693/0.597/0.532/0.472/0.420/0.380. The proxy's 1.98 matched the served 1.84-1.90 acc/draft.

Served A/B, same prod config (SPEC=7, TP=2), FP8 block-128 export of the fine-tune vs tcclaviger's:

| | old drafter | **fine-tuned** |
|---|---|---|
| bench_decode_ctx acc/draft @ctx25 / 8k / 32k | 1.844 / 1.844 / 1.703 | **2.053 / 2.077 / 1.985** (+11 / +13 / +17%) |
| single-stream decode tok/s @ctx25 / 8k / 32k | 116.2 / 109.4 / 100.9 | **125.0 / 118.0 / 111.2** (+8 / +8 / +10%) |
| ms/step | 24.47 / 26.00 / 26.78 | 24.42 / 26.07 / 26.84 (unchanged) |
| BetterBench combined (single pass) | 207.5 | 209.2 (+0.8%) |
| BB by category: chat / code / json / math | 101 / 214 / 231 / 240 | **104 / 222 / 257 / 255** |
| BB by category: file_edit / prose / reasoning / summarization | 217 / 105 / 242 / 227 | 205 / 104 / 237 / 208 |
| GSM8K 500q | 97.60 | 97.60 |

Reading: acceptance rose where the training mix has coverage (chat, code, json, math) and slipped on
the categories it does not (summarization, file_edit, long reasoning prompts). Output quality is
unchanged (lossless drafting; GSM8K identical). The overnight gate's bar was +2% BetterBench
combined, so prod was restored with the OLD drafter pending a decision. The fine-tuned drafter is at
`~/models/Qwen3.8-27B-DFlash2-FP8-paro` (bf16 at `~/drafter_ft/ft_bf16`); serve it with
`DRAFTER=Qwen3.8-27B-DFlash2-FP8-paro`. Next round, if wanted: widen the prompt mix to summarization
/ file-edit / long-document prompts and raise the lr (5e-5 barely moved top-1 in 1,154 steps).
