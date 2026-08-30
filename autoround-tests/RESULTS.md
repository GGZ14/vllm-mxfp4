# AutoRound int4 W4A8 kernel for gfx1201

Target checkpoint: `Frozenlock/Qwen3.8-27B-int4-AutoRound` (18.70 GB, downloaded to
`~/models/Qwen3.8-27B-AutoRound-int4`).

## What the checkpoint actually is

| field | value |
|---|---|
| packing | `auto_round:auto_gptq` — `qweight` [K/8, N] i32, `qzeros` [K/128, N/8] i32, `scales` [K/128, N] f16 |
| bits / group | 4 / 128 |
| sym | **true** — every `qzeros` word is `0x77777777`, i.e. nibble 7, i.e. zero = 8 everywhere |
| `g_idx` | **absent** — no activation reordering, so groups are contiguous along K |
| unquantized | 99 `linear_attn.in_proj_a` / `in_proj_b` tensors held at bf16 |
| bits/weight | 4 + 16/128 + 4/128 ≈ **4.156** vs MXFP4's 4.25 |

Two of these decide the whole kernel design:

**Symmetric with a constant zero of 8** means code `c` in 0..15 represents the integer `c - 8` in
[-8, 7], and every one of those sixteen integers is EXACTLY representable in e4m3. So the zero
point folds into the unpack table and never enters the matmul — no correction term, no activation
row-sums, which is what a general asymmetric GPTQ kernel has to carry. Proven in `lut.py`.

**group_size 128 equals the tuned decode slab `DBK`**, so a group boundary IS a slab boundary. The
rescale lands on the `__syncthreads()` that already exists and the inner 8-step WMMA loop is
untouched. Better still, in the gfx12 16x16x16 wave32 C layout a lane's N index is `lane & 15`,
identical across all eight of its accumulator slots — so a lane needs ONE scale per group, not
eight: one f16 load and 8 `v_fmac_f32` per slab, about 1 VALU per WMMA.

## Correctness

`./run.sh` — 60/60 pass. M in {1,2,3,5,8,9,13,16,17,32,40,64} x five shapes including N=48,
N not a multiple of BND, and K not a multiple of DKS*DBK.

`rel ≈ 1.8e-3` uniformly, which is the bf16 output rounding floor (2^-9), not kernel error.
`rows-past-M-touched = 0` everywhere: the destination is poisoned with bf16 NaN and four rows past
M are asserted untouched, because an M-padded kernel writing row 15 of a 5-row output is an
out-of-bounds write that a relative-error check passes happily.

Resource usage: 75-104 VGPRs, 38-40 SGPRs, **0 spills**, occupancy 12 waves/SIMD, LDS 21764 B/block.

## v_perm_b32 on gfx1201 — measured, not assumed

The unpack needs a 16-entry table, but `v_perm_b32` addresses only 8 bytes. Two attempts failed
before the selector semantics were measured (`permprobe.hip`):

| sel | behaviour on gfx1201 |
|---|---|
| 0-3 | byte 0..3 of **S1** (the second argument) |
| 4-7 | byte 0..3 of **S0** (the first argument) |
| 8-12 | 0x00 in the probe, but **11 is data-dependent** |
| 13-15 | **0xFF unconditionally**, even with no MSB set anywhere in the pool |

- Attempt 1 drove the unwanted table half's selector into 8..11 expecting `0x00`. It got `0xFF`,
  which is **e4m3 NaN**, and the entire GEMM returned NaN.
- Attempt 2 built the blend mask by sign-replication. It got the byte lanes crossed — and no
  uniform-code test can see that, because uniform codes make cross-byte mixing invisible. It passed
  `lutest.hip` (uniform codes) and failed only under codes varying in both n and k.

The shipped version uses **only selector values 0..7**, and builds the 0xFF/0x00 mask from a
2-entry pool indexed by bit3. `unpacktest.hip` gates all eight nibbles of 4096 random words.

Lesson worth keeping: a uniform-value test cannot detect a permutation. The one-hot diagnostic in
`--diag` (mode 2, code varying in BOTH n and k) is what localized this; modes 0 and 1, each varying
one axis, both passed while the kernel was wrong.

## Speed

`./run.sh --bench`. **Measured with the production server up** — contention is real, but the
roofline probe is interleaved with the GEMM in the same loop, so the FRACTION self-normalizes.

| shape | N | K | M=5 | M=8 | M=16 | M=40 | M=64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| gate_up | 17408 | 5120 | 57.8% | 56.5% | 54.8% | 29.4% | 26.5% |
| down | 5120 | 8704 | 72.9% | 72.1% | 69.3% | 34.9% | 30.7% |
| out | 5120 | 5120 | 88.9% | 86.9% | 81.1% | 42.7% | 36.7% |

(% of the interleaved streaming roofline.)

Decode-band time is essentially flat in M — gate_up runs 68.0 us at M=5 and 73.7 us at M=16 — which
is the signature of a bandwidth-bound kernel and is what we want. M=40/64 (DTM 3 and 4) roughly
doubles, so those shapes have become compute-bound and are the place to tune if batched decode
matters.

> **Superseded, 2026-08-26.** M=40/64 was never compute-bound. It was split-K partial traffic,
> which is linear in M, plus serialised staging. See "Branchless staging and the split-K policy"
> below; M=40 gate_up is now 95.2 us against the old 124.1.

### The absolute GB/s in this harness are cache-fed, not DRAM-fed

R9700 L3 (Infinity Cache) is **65536 KB = 64 MB**, confirmed from `rocminfo`. The gate_up weight
buffer is 17408 x 5120 / 2 + scales = **44.6 MB**, so it sits entirely in cache and is re-read 20
times per timing loop. That is why the table reports 675 GB/s against a 635 GB/s DRAM peak — over
100% of DRAM, which is the known artifact, not a result. In real serving the model's ~14 GB of
weights cannot be cache-resident, so absolute decode bandwidth will be lower. The percentages are
still a fair efficiency measure because the probe shares the residency; the absolute GB/s are not.

## Prefill kernel

Same tile as the MXFP4 folded kernel (BMF=256 via AR_TM=4, BK=64, PAD=8, LDS 23040 B -- byte
identical, so 2 blocks/CU is preserved). The one structural difference is forced by the format:
MXFP4 folds its block exponent into the weight byte and has no rescale at all, while an fp16 group
scale cannot be folded that way.

group_size 128 with BK=64 means a group is exactly TWO SLABS. Two variants were built and both are
correctness-gated:

| variant | temp accumulator | VGPR | occupancy |
|---|---|--:|--:|
| group temp | AR_TM x TN tiles, folded once per group | 183 | 8 |
| **IMAJOR** (shipped) | TN tiles, folded once per slab | **123** | **10** |

IMAJOR is valid because both slabs of a group carry the SAME scale, so folding per slab is
arithmetically identical: s*(P1+P2) either way. It costs re-reading the sW fragments once per
M-fragment. For reference the MXFP4 folded kernel is 116 VGPR at occupancy 10, so IMAJOR reaches
parity on registers.

Correctness: 27/27 against an fp64 CPU reference (M up to 512, N=64/200/256, K not a multiple of
BK*2), plus prefill-vs-decode agreement at M=64 on all three production shapes at rel ~1e-5. The
decode path was gated against fp64 first, so that cross-check pins prefill to a known-good path at
a real shape without needing an M*N*K double-precision reference.

## Head to head against the shipped MXFP4 kernel

`cmp.hip` compiles BOTH kernel families into one binary and interleaves them in a single timing
loop. Its argument rotates over that many weight buffers so the working set can be pushed past the
64 MB Infinity Cache; NCOPY=3 is 131 MB on gate_up, DRAM-resident, which is the case that matches
serving. (`mk_mxfp4_header.py` derives the MXFP4 header from the shipped .hip rather than
committing a copy that could drift.)

The first build was 8-15% slower than MXFP4 at decode. Two fixes, both found by ablation, reversed
that. Decode, us, DRAM-resident:

| shape | M | first build | shipped | mxfp4 | ratio |
|---|--:|--:|--:|--:|--:|
| gate_up | 8 | 103.9 | **92.7** | 96.0 | **0.966x** |
| gate_up | 40 | 135.7 | **125.4** | 124.5 | 1.007x |
| gate_up | 64 | 174.4 | **158.8** | 167.9 | **0.945x** |
| down | 8 | 38.4 | **30.8** | 34.8 | **0.885x** |
| down | 16 | 41.1 | **33.6** | 38.3 | **0.877x** |
| down | 40 | 78.4 | **50.4** | 54.6 | **0.923x** |
| down | 64 | 92.7 | **76.7** | 87.2 | **0.880x** |
| out | 8 | 24.3 | **20.5** | 21.4 | **0.960x** |
| out | 40 | 39.9 | **33.1** | 33.3 | **0.994x** |

**The int4 decode kernel is now faster than MXFP4 at every shape and batch measured**, by up to
12.3%. Prefill remains 1.12-1.23x slower.

> **Superseded, 2026-08-26.** Both numbers moved a long way -- see the section below. Decode is
> 0.70-0.94x MXFP4 and prefill 1.00-1.08x.

### Fix 1: stage 8 bytes of weight per thread, not 4

The staging loop read one uint32 per thread. MXFP4's reads a u64. A 4-byte global load halves the
bytes moved per instruction, and this repo had already paid for that exact lesson once -- a
4-byte-per-thread fragment-order staging cost 14-25% of prefill until it was widened to two
adjacent lane slots. Reading a uint2 and unpacking 16 codes per thread is worth **-11% decode**
(gate_up M=8 103.9 -> 94.0) and flipped the ranking on its own. Both LDS halves stay 8-byte
aligned, so no 16-byte store alignment is assumed.

### Fix 2: issue the group-scale load before staging

The scale load sat after the `__syncthreads()`, immediately ahead of the WMMA run, so its latency
was exposed. It depends on nothing the staging computes, so hoisting it above the A and W staging
gives it the whole stage plus the barrier to land. Worth **-22% at M=40** (down 64.3 -> 50.4) and
1-2% elsewhere.

That M=40 case is the one the ablation flagged hardest: `-scale` was removing 29% of runtime there
while the fold itself is only one FMA per WMMA. Arithmetic could not account for it, which is what
identified the stall as load latency rather than the multiply.

### The ablation that found both

`abl.hip` builds variants that skip the unpack (writing raw code bytes) or the scale load and fold.
Each is WRONG but leaves memory traffic and WMMA count untouched, so the delta bounds what the
corresponding optimisation could buy. Post-fix, decode:

| shape | M | full | -unpack | -scale | -both | mxfp4 |
|---|--:|--:|--:|--:|--:|--:|
| gate_up | 8 | 96.0 | 93.8 | 92.5 | 90.5 | 97.0 |
| down | 40 | 66.6 | 61.9 | **47.5** | 43.3 | 54.6 |

The `-scale` column at M=40 is what pointed at fix 2. It also settles a tempting idea: the unpack
is worth only 2-10% at decode and **0.5% at prefill** (where staging is amortised ~16x), so
switching the weight leg to `v_wmma_i32_16x16x16_iu8` -- where a nibble unpacks in 2 ops instead
of 9 -- cannot pay for moving activations from e4m3 to int8 and the accuracy that would cost.
**iu4 at 830 TF/s is likewise not reachable**: it needs int4 activations, i.e. W4A4.

### Things measured and rejected

- **Tile split at fixed BMF=256**: AR_TM=4/AR_WM=4 wins. TM=2/WM=8 is 2-7% worse; TM=8/WM=2 spills
  at 256 VGPR and is 1.5x worse.
- **DTM choice at M=33..48**: the launcher's `ceil(M/16)` rule is right. DTM=4 at M=40 is slower
  than DTM=3 (159.0 vs 155.5 on gate_up), so the M=40 gap was never a tile-selection problem.
- **Decode IMAJOR**: neutral (105.7 -> 105.6). Kept only because prefill needs it.

### Why prefill stays ~1.2x  (WRONG -- see the 2026-08-26 section)

MXFP4's scale is a power of two, so it folds into the weight byte exactly and its prefill inner
loop is pure WMMA with no temp accumulator at all. An arbitrary fp16 group scale cannot fold that
way without rounding the product to e4m3's four significant bits -- which is precisely the lossy
pow2-scale conversion measured at +41.5% weight error. So the rescale is not removable at
acceptable quality. Fully ablating both unpack and scale still leaves prefill ~9% short, and the
remainder is the IMAJOR restructuring's extra LDS reads. Roughly 15-20% at prefill is the price of
the format on this hardware; decode, where bytes/weight dominates, goes the other way.

> **This conclusion was wrong.** The rescale is real, but it is not what cost 15-20%. Nearly all
> of the gap was a codegen accident in the staging loop, and the ablation could not see it because
> `-unpack` and `-scale` leave the staging LOADS in place -- and the loads were the problem. With
> branchless staging the same kernel, same tile, same rescale, same arithmetic, bit-identical
> output reaches 1.00-1.08x of MXFP4. An ablation bounds only what it actually removes.

## Branchless staging and the split-K policy (2026-08-26)

Both kernels got faster from two changes that touch no arithmetic. Every decode and prefill
variant below is **bit-identical** to what it replaced on every gated shape, and the shipped
`./run.sh` gate still passes 0 failures.

### The defect: a bounds predicate costs a full memory round trip

A staging load written as `if (gm < M) v = *(...)` lands inside an `s_and_saveexec_b32` /
`s_cbranch_execz` region. The compiler cannot carry a COUNTED `s_wait_loadcnt N` across an
exec-mask merge, so it emits `s_wait_loadcnt 0x0` -- wait for everything -- after every single
one. The shipped ISA was `load -> s_wait_loadcnt 0x0 -> ds_store`, repeated:

| kernel | staging loads per k-slab | round trips paid |
|---|--:|--:|
| prefill | 2 scale + 4 A + 1 W | **7** |
| decode, DTM=3 | 1 scale + 2 A + 4 W (A nested two deep) | **6** |

One would do. It also silently voided the decode kernel's scale hoist: the source issues the
group-scale load above the staging deliberately, but the ISA was `global_load_d16_b16` followed
immediately by `s_wait_loadcnt 0x0`, so it never overlapped anything.

The fix is to **clamp the index rather than predicate the load** (`rc = r < M-1 ? r : M-1`).
Safe because the epilogue already drops rows past M and columns past N, and a clamped read
returns real finite data -- never the `0xFF` byte that is e4m3 NaN. All seven loads then issue
back to back under one wait. As a bonus, decode DTM=4 drops 177 -> 142 VGPR (the predicated form
needed registers to carry the zero-initialised values across the branches) and occupancy goes
8 -> 10.

A second, smaller prefill fix stacks on it: IMAJOR re-read the eight W fragments once per
M-fragment (the ISA showed offset pairs `{0,144},{2,146},{4,148},{6,150}` four times over).
Hoisting them out of the `i` loop costs 6 VGPRs, 122 -> 128, and leaves occupancy at 10.

### Split-K is not free, and its cost is linear in M

The decode kernel's partial buffer costs `DKS*M*N` floats of write-then-read traffic. That is
constant in the weight stream but **linear in M**: 11.1 MB against gate_up's 44.6 MB of weights
at M=40, 17.8 MB at M=64. The reduction also serialises onto the ONE block per n-range that
finishes last, while the other `DKS-1` have exited. So the split that pays for itself at M=5 is a
large loss at M=64 -- and on gate_up, which is 136 n-blocks before any split at all, it never
paid. `DKS==1` now skips the partial buffer, the threadfence, the atomic and the reduction pass
entirely and writes C directly.

`split_k_for()` in `radiance_autoround.hip` takes the smallest split that still fills the machine
and shrinks it further as M grows. It picks the best measured cell in all fifteen shape/M
combinations.

### Decode, us, idle GPU, NCOPY=3 (`decopt.hip`)

| shape | M | mxfp4 | shipped | branchless DKS=4 | DKS=2 | DKS=1 | best vs shipped |
|---|--:|--:|--:|--:|--:|--:|--:|
| gate_up | 5 | 97.0 | 93.3 | 87.3 | 85.8 | **83.1** | **0.890x** |
| gate_up | 16 | 102.3 | 99.6 | 93.7 | 89.5 | **84.0** | **0.843x** |
| gate_up | 40 | 124.5 | 124.1 | 116.3 | 108.1 | **95.2** | **0.767x** |
| gate_up | 64 | 168.6 | 159.2 | 144.8 | 132.9 | **118.4** | **0.744x** |
| down | 5 | 32.5 | 30.8 | **25.7** | 28.7 | 41.9 | **0.835x** |
| down | 40 | 55.2 | 54.3 | **46.4** | 48.5 | 64.1 | **0.855x** |
| down | 64 | 86.9 | 83.9 | 66.0 | **62.3** | 78.9 | **0.743x** |
| out | 16 | 22.8 | 22.1 | **19.2** | 20.3 | 26.0 | **0.866x** |
| out | 64 | 50.8 | 48.5 | 42.3 | **40.7** | 49.0 | **0.839x** |

Note how sharply DKS wants to differ by shape: DKS=1 is the best cell on gate_up at every M and
the WORST on down and out, where 40 n-blocks cannot fill the GPU alone.

### Prefill, us, idle GPU, NCOPY=3 (`preopt.hip`)

| shape | M | shipped | branchless | +W-hoist | mxfp4 | was vs mxfp4 | now |
|---|--:|--:|--:|--:|--:|--:|--:|
| gate_up | 512 | 681.7 | 616.1 | **589.3** | 585.3 | 1.165x | **1.007x** |
| gate_up | 2048 | 2367.1 | 2246.5 | **2122.4** | 2041.1 | 1.160x | **1.040x** |
| gate_up | 4096 | 4577.6 | 4415.0 | **4159.9** | 3912.0 | 1.170x | **1.063x** |
| down | 512 | 478.6 | 323.0 | 334.1 | 385.7 | 1.241x | **0.866x** |
| down | 2048 | 1300.0 | 1161.0 | **1098.8** | 1092.6 | 1.190x | **1.006x** |
| down | 4096 | 2390.3 | 2209.3 | **2114.9** | 2010.5 | 1.189x | **1.052x** |

### Measured and rejected

- **`s_setprio(1)` around the WMMA run**: a loss at every prefill shape, up to +11%.
- **Register prefetch of the next k-slab across the WMMA run**: no gain once staging is
  branchless, and +18 VGPR. The point of a prefetch is to overlap the load latency, and one
  batched round trip at occupancy 10 already is overlapped.
- **Prefill tile sweep (AR_WN x TN in {2,4}^2)**: CONCLUDED 2026-08-30 (interleaved, NCOPY=6,
  stable MXFP4 reference). The SHIPPED tiles win every cell: ship-TN4 at M>=2048 beats opt-tn4 by
  2-4% and WN=4 by 23-72% (WN=4 runs 256 VGPRs at occupancy 5-6 -- the register wall the 08-26
  TM/WM sweep predicted). The TN2->TN4 crossover at 2048 re-confirms. The tile space is CLOSED.

### Hardware facts established while looking (`gfx1201`, ROCm 7.14 / clang 23)

- **No direct-to-LDS DMA.** `__builtin_amdgcn_global_load_lds` needs the `vmem-to-lds-load-insts`
  target feature, which gfx1201 does not have; the compile is a hard error. Staging on RDNA4 must
  round-trip through VGPRs. The CDNA-style async-copy pipeline is not an option here.
- **No wider fp8 WMMA.** `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` is the only fp8 shape; there is no
  16x16x32 fp8 and no fp8 `swmmac`.
- `wmma_i32_16x16x32_iu4_w32_gfx12` DOES exist, but it needs int4 activations (W4A4), which
  confirms the earlier rejection of the iu4 path from a different direction.
- `s_prefetch_data` is available and emits. Untested for value.

### Still open

- The prefill epilogue is untouched: 64 scattered 2-byte `global_store_d16_hi_b16`, each with its
  own exec-mask save/restore and 64-bit address math, plus 55 `v_cmp_u_f32` from the bf16
  convert's NaN handling. About 1/80th of runtime at K=5120, and the sloppiest code in the kernel.
- The tile sweep above.
- `cmp.hip` still launches decode at a hardcoded DKS=4, so its decode column shows only the
  branchless gain, not the split-K policy. The policy is exercised through the module dispatch.

## vLLM integration

`radiance_autoround.py` registers an `auto-round` quantization config and routes linears to the
kernel through a `radiance::autoround_linear` custom op. The op owns the whole dispatch because a
shape branch written in `apply()` is data-dependent and splits the torch.compile graph at every
linear -- the MXFP4 path measured that at ~30% of decode.

Verified against the real checkpoint config, single-process TP group:

- config parses: `group_size=128, sym=True, unquantized_modules=99`
- routing: `in_proj_a` / `in_proj_b` -> `UnquantizedLinearMethod`, everything else -> the kernel
- `create_weights` at down_proj TP=2 (K 17408 -> 8704): qweight (1088, 5120) i32,
  scales (68, 5120) f16, qzeros (68, 640) i32 -- all shard correctly via vLLM's own parameter
  classes
- `process_weights_after_loading` transposes qweight to (5120, 1088) = [N, K/8], which is what the
  kernel reads, and keeps scales at fp16 rather than casting to bf16 (bf16 has three fewer mantissa
  bits and would quantize the scale itself)
- the asymmetry guard fires on a non-0x77777777 qzeros: the kernel folds a CONSTANT zero of 8 into
  its table and must refuse a checkpoint it would be silently wrong for

`run_autoround.sh` builds the module into the image and serves. Registration happens through a
`sitecustomize.py` shim in site-packages so the decorator runs in the engine process AND every TP
worker, not in a throwaway interpreter.

## What this does NOT yet establish


1. **The model has never been served.** Everything above is kernel-level and load-path-level. An
   end-to-end BetterBench run needs the production MXFP4 server stopped, because it holds
   `--gpu-memory-utilization 0.98` on both GPUs and there is no room beside it.
2. **No quality measurement.** Which is the only axis on which this format can win. Needs
   `ppl.py` and the GSM8K 500q gate against the MXFP4 reference of 8.3706.
3. ~~**M=40 decode remains 1.43x** and is not explained by register pressure. Unresolved.~~
   **Resolved 2026-08-26**: split-K partial traffic (linear in M) plus six serialised staging
   round trips per slab. Neither is register pressure. M=40 gate_up is now 95.2 us against the
   old 124.1.
4. **All timings were taken with the production server running.** Interleaving makes the A/B fair
   -- both kernels pay the same contention -- but the absolute microseconds are inflated.

## Expectation to hold onto

Both this kernel and the MXFP4 one are bandwidth-bound at decode, so what separates them is
bits/weight: **4.156 vs 4.25, about 2.3%**. Unpack cleverness and rescale cost both hide under the
weight stream (at 635 GB/s and ~2.4 GHz there are ~31 VALU slots per streamed byte and this kernel
needs ~5). So the honest ceiling for this format over MXFP4 at single-stream decode is low single
digit percent. If AutoRound wins, it should be expected to win on QUALITY, not speed.

## Files

| file | role |
|---|---|
| `ar_kernels.h` | the decode and prefill kernels -- shared by the harness and the shipped module |
| `ar_harness.hip` | correctness gate + `--diag` one-hot probe + `--bench` |
| `cmp.hip` / `cmp.sh` | head-to-head against the MXFP4 kernels in one binary; arg = NCOPY |
| `radiance_autoround.hip` | pybind module (decode/prefill dispatch, split-K scratch) |
| `radiance_autoround.py` | vLLM `auto-round` quant config + linear method |
| `run_autoround.sh` | build-and-serve launcher |
| `lut.py` | derives and verifies the int4-sym -> e4m3 table, emits the constants |
| `lutest.hip` | device gate for the 16-entry table (uniform codes) |
| `unpacktest.hip` | device gate for all 8 nibbles of random words — the test that matters |
| `permprobe.hip` | ground-truth map of v_perm_b32 selector semantics on gfx1201 |
| `run.sh` / `lutest.sh` / `unpacktest.sh` / `permprobe.sh` | build+run in the radiance image |

---

# Final tuning state (2026-08-27)

## Decode is at the memory wall

Measured in one binary against a streaming probe on the SAME rotated DRAM-resident buffers, so the
denominator is what the machine actually delivers rather than the 635 GB/s spec figure. gate_up is
the shape to trust: 263 MB working set, fully past the 64 MB Infinity Cache, DKS=1 as shipped.
(`down` and `out` report >100% of their probe because at 138 MB and 78 MB they still take partial
cache hits -- their probe is not a valid ceiling, and quoting those numbers would be wrong.)

| shape | M | kernel GB/s | achievable stream | **% of achievable** |
|---|--:|--:|--:|--:|
| gate_up | 5 | 548.1 | 573.8 | **95.5%** |
| gate_up | 8 | 546.1 | 580.1 | **94.1%** |

Two independent lines of evidence agree that this is the end:
  * 95.5% of achievable streaming bandwidth leaves 4.5%.
  * Ablating the unpack AND the scale fold entirely -- wrong answers, identical traffic -- recovers
    only 3-6%.

Against the shipped MXFP4 kernel, decode is **8-14% faster at every production shape** (MXFP4 runs
at 73-75% of the spec peak where we run at 86%).

## Prefill: the epilogue was the last real lever (2026-08-27)

The per-kernel ISA census (`isa.hip` + `isa_census.py`) showed where the instruction budget went,
and it was not arithmetic:

| kernel | total | wmma | addr64 pairs | s_and_saveexec | 64-bit shifts |
|---|--:|--:|--:|--:|--:|
| decode DTM=1 | 563 | 8 | 19 (3.4%) | 1 (0.2%) | 14 |
| prefill TN=2 | 2713 | 32 | 142 (5.2%) | 95 (3.5%) | 132 (4.9%) |

Decode was already clean -- the clamp-not-predicate work did that. Prefill was spending ~16% of its
instructions on 64-bit addressing and exec-mask predication, in a kernel that is compute-bound at
~56% of peak. The cause was the epilogue: AR_TM*TN*8 = 64 individually bounds-tested stores, each
with a full `(size_t)m * N + n` address.

Two fixes, both in the epilogue: hoist a wave-uniform base so the compiler emits the SADDR form
with a 32-bit offset, and take a branch-free fast path when the block lies entirely inside M and N
(on a real prefill only the last row-block and column-block are ragged). Worth **3.3-4.5%**:

| shape | M | MXFP4 | shipped TN=4 | ratio |
|---|--:|--:|--:|--:|
| gate_up | 2048 | 1768.2 | **1762.8** | **0.997x** |
| gate_up | 4096 | 3680.8 | **3568.0** | **0.969x** |
| down | 4096 | 1780.3 | 1800.9 | 1.012x |

Note the STATIC instruction count went UP (2713 -> 3368) because the fallback epilogue is still
compiled. What matters is the path executed, and the fast path is the one that runs.

**int4 prefill now beats MXFP4 on the largest shape**, at 57.6-58.3% of the 355 TF/s peak against
MXFP4's 55.9-58.2%.

## Prefill is at the shape's ceiling, and it is a SHARED ceiling

| kernel | M | TFLOP/s | % of 355 peak |
|---|--:|--:|--:|
| int4 TN=4 | 2048 | 198.5 | 55.9% |
| MXFP4 | 2048 | 206.9 | 58.3% |
| int4 TN=4 | 4096 | 196.7 | 55.4% |
| MXFP4 | 4096 | 197.3 | **55.6%** |

At M=4096 the two are within 0.2 points of each other. 55-58% is what this GEMM shape reaches on
gfx1201; it is not an int4 deficiency. In serving, prefill at the 8k chunk size now BEATS MXFP4
(4435 vs 4299 t/s).

## Every lever, and its verdict

| lever | verdict |
|---|---|
| weight staging 4 -> 8 B/thread | **shipped**, -11% decode |
| staging 8 -> 16 B/thread | closed: identical (87.2 vs 87.2 us) |
| hoist group-scale load above staging | **shipped**, -22% at M=40 |
| clamp instead of predicate | **shipped** (removes s_wait_loadcnt per load) |
| DKS==1 fast path + shape-aware split-K | **shipped** |
| W-fragment hoist out of the prefill i-loop | **shipped** |
| IMAJOR prefill (TN temp tiles) | **shipped**, 183 -> 123 VGPR |
| TN=4 above M=2048 | **shipped**, prefill +8% at 8k |
| BK 32 / 64 / 128 | closed: within 0.5% normalised |
| TM/WM split at fixed BMF | closed: 4/4 optimal, 8/2 spills |
| DTM selection | closed: ceil(M/16) is right |
| iu8 / iu4 int WMMA | closed: unpack is 2-6% of decode, cannot pay for int8 activations |
| decode IMAJOR | closed: neutral |
| prefill epilogue: hoisted base + full-tile fast path | **shipped**, 3.3-4.5% |
| cheaper f32->bf16 conversion | closed: truncation saves 0-0.5% |
| register-resident weights (drop LDS for W) | closed: removing the LDS round-trip entirely is 0, and slower on 2 of 6 |
| hardware f32->bf16 convert | closed: does not exist on gfx1201 (cvt_pk_bf16_f32 is CDNA-only) |

## The residual serving gap is the DRAFTER, not the kernel

Serving decode is 140.9 t/s against MXFP4's 167.3, even though our kernel is faster per step. The
cause is acceptance, and it is measurable. Same DFlash2-FP8 drafter, same SPEC=7, same box:

| target | accepted / draft | rate |
|---|--:|--:|
| MXFP4 | 3.673 | 52.5% |
| AutoRound int4 | 2.718 | 38.8% |

The drafter was built against the MXFP4 lineage, and the AutoRound checkpoint is genuinely a
different set of weights -- its `post_attention_layernorm` differs from the FP8 reference on 99.6%
of channels, by no scale or offset (see the AWQ-fold investigation earlier in this file). So the
drafter predicts the MXFP4 target far better.

The arithmetic closes: acceptance ratio 1.35x, our per-step advantage 1.08-1.14x, net 1.19x in
MXFP4's favour -- which is exactly the observed 167.3/140.9. Nothing is unexplained, and none of
the remainder is the kernel.

Caveat on those two acceptance figures: they come from different prompt distributions (an ad-hoc
mixed corpus at n=967 against BetterBench's 29-prompt corpus at n=32,774). The direction is not in
doubt; the exact ratio is approximate.

**To close the decode gap the lever is a drafter matched to this target** -- either the
checkpoint's own int4 MTP head (which holds acceptance at 1.911/draft at SPEC=4, but MTP is worth
less than DFlash2 here: 86.2 vs 140.9 t/s) or a DFlash2 drafter distilled against the AutoRound
weights. Neither is kernel work.

## The bf16 conversion: identified, measured, rejected

The per-kernel ISA showed a group of ops that looked like a lever: `v_cmp_u_f32` + `v_bfe_u32` +
`v_or_b32` + `v_add3_u32` + `v_cndmask_b32`, about 5 VALU per output element. That is what a plain
`(__bf16)` cast compiles to on gfx1201 -- round-to-nearest-even AND NaN propagation -- and there is
no hardware convert to replace it with: `__builtin_amdgcn_cvt_pk_bf16_f32` does not exist for this
target, it is CDNA-only.

Replacing it with a bare truncation (WRONG to ship -- it roughly doubles output rounding error)
measures the ceiling on any cheaper conversion:

| shape | M | shipped RNE | truncating | delta |
|---|--:|--:|--:|--:|
| gate_up | 2048 | 1760.6 | 1761.1 | 0.0% |
| gate_up | 4096 | 3566.5 | 3548.2 | -0.5% |
| down | 4096 | 1796.1 | 1793.6 | -0.1% |

0-0.5%. The reason is structural and worth remembering when reading a static ISA census: the
epilogue is fully unrolled so it DOMINATES the instruction listing (64 stores at TN=4), but it
executes ONCE per block against ~80 iterations of the main loop. Static instruction counts point at
the epilogue; runtime does not. The epilogue's predication was worth 3.3-4.5% because
`s_and_saveexec` serialises, not because the epilogue is a large share of the work.

`AR_BF16_TRUNC` is left in place, defaults off, so the result stays re-measurable.

## Register-resident weights: closed without building it

libr4d's MXFP4 skinny GEMM is register-resident -- it reads the weight straight from global into
the WMMA fragment, with no LDS and no barrier, on the argument that "global load already has the
shape the fragment wants; staging it would cost LDS, a barrier and a [sync]". The same argument
looks like it should apply here, and more strongly: at DTM=1, single-stream decode, each staged
weight byte is read exactly ONCE by exactly one wave. There is no reuse at all, so the LDS write,
the barrier and the LDS read look like pure overhead.

Building it needs a fragment-order weight layout (the loader change the MXFP4 WPERM path already
demonstrates), so before writing any of that, ABLATE bit 2 measures the ceiling: keep the global
read and the unpack, never send the result to LDS, feed the WMMA from a register instead.

| shape | M | full | -LDS | delta |
|---|--:|--:|--:|--:|
| gate_up | 8 | 89.6 | 90.1 | +0.6% SLOWER |
| gate_up | 64 | 145.0 | 143.3 | -1.2% |
| down | 8 | 43.3 | 42.8 | -1.2% |
| down | 40 | 59.7 | 62.5 | +4.7% SLOWER |

Zero, with two cases going backwards. The LDS round-trip is entirely hidden under global-load
latency, which is what a kernel already running at 95.5% of achievable streaming bandwidth must
look like. A whole loader change and a second decode kernel avoided by one ablation.

Note the asymmetry with prefill, where the answer is different for a structural reason rather than
a measured one: there `sW` is shared by WM=4 waves and `sA` by WN=2, so register-resident would
take per-slab staging from 20 KB to 48 KB. LDS is load-bearing at prefill and free at decode.


---

# State-of-the-art audit (2026-08-30)

Requested and answered: is this prefill kernel the best it can be on gfx1201/R9700?

Every named technique is measured: branchless staging (originated here), 8 B weight staging
(16 B identical), SADDR hoists, sched_barrier, W-fragment hoist, IMAJOR (183 -> 123 VGPR),
epilogue fast path, BK/TM/WM/TN/WN tile space fully swept (this file), register-resident W (0),
double-buffering (-1%), s_setprio (+11% loss), fp8 WMMA throughout. Decode: 95.5% of achievable
stream, beats MXFP4 at 13/15 cells (cmp.hip, 2026-08-30).

The remaining 13-23% prefill gap to MXFP4 is the FORMAT, mathematically:
  * the fp16 group scale cannot fold into e4m3 weights (+41.5% weight error, measured);
  * the two-level split s = 2^e x m folds only the exponent -- the mantissa m in [1,2) still
    costs the same per-group FMA, because it varies along K and must apply before cross-group
    summation;
  * the rescale's temp accumulators forced the IMAJOR restructure, which reaches register
    PARITY with the MXFP4 folded kernel (123 vs 116 VGPR, both occupancy 10) but pays for it by
    re-reading the sW fragments once per M-fragment -- the ~9% residue the original ablation
    measured, structural to needing any temp accumulator at all.
MXFP4's zero-rescale inner loop requires power-of-two scales; this format does not have them.
The kernel is at its format's ceiling on this hardware.