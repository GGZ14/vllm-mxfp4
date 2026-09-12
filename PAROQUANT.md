# ParoQuant W4A8 on RDNA4

A second int4 weight format served by this stack, alongside native MXFP4:
[ParoQuant](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) (ICLR'26) — int4, group 128,
**asymmetric**, with learned **pairwise Givens rotations** and per-channel scaling applied to the
activations before the GEMM.

The reference implementation is CUDA-only and W4A16. This is an independent **W4A8**
reimplementation for gfx1201: hand-written HIP rotation and GEMM kernels, registered as a real vLLM
quantization method. There is no PyTorch dequant fallback and no AWQ GEMM underneath — the rotation
is a kernel and the zero point never enters the matmul.

It serves `z-lab/Qwen3.8-27B-PARO` at **GSM8K 97.4-98.0%** (MXFP4 97.8) with **combined decode
226 t/s against MXFP4's 186**, on 2 x R9700.

## Contents

- [Setup and running](#setup-and-running)
- [The format](#the-format)
- [The rotation kernel](#the-rotation-kernel)
- [The GEMM](#the-gemm)
- [Merged linears and partition select](#merged-linears-and-partition-select)
- [Checkpoint loading](#checkpoint-loading)
- [Configuration](#configuration)
- [Traps](#traps)
- [Results](#results)
- [MXFP4 weights: the zero-VALU loop](#mxfp4-weights-the-zero-valu-loop)

## Setup and running

```bash
./setup-paroquant.sh                      # host check, image, checkpoint, drafter, kernels
MODE=prod SPEC=7 ./paroquant/run_paroquant.sh
```

`setup-paroquant.sh` is idempotent and shares the image, the drafter and libr4d with
`setup-mxfp4.sh`, so running both costs one download of each.

Two modes:

| `MODE` | What it is |
|---|---|
| `eval` (default) | `--enforce-eager`, 32K context, no speculative decoding, `RADIANCE_PQ_CHECKALL` comparing every gated shape against an exact fp32 dequant. Correctness gating only |
| `prod` | 262K context, prefix caching, compiled graphs, the DFlash2 FP8 drafter at `SPEC` tokens |

On a host with the systemd units installed, `vllm-switch paro` starts it as `qwen_vllm_paro` and
stops whatever else holds the GPUs. Both stacks want both cards and port 8080, so only one runs at
a time.

The kernel module is compiled inside the container at start (`hipcc --offload-arch=gfx1201` over
`radiance_paroquant.hip`) from this directory, exactly as the MXFP4 kernel is. Nothing is baked
into the image.

## The format

Verified against the real checkpoint rather than taken from the paper. Per projection:

| Tensor | Shape | Notes |
|---|---|---|
| `qweight` | `[K, N/8]` int32 | AWQ nibble order `0,2,4,6,1,3,5,7` |
| `qzeros` | `[K/128, N/8]` int32 | **Not** offset-by-one. Genuinely asymmetric: values 2..14, only 38% equal 8 |
| `scales` | `[K/128, N]` f16 | |
| `pairs` | `[krot, K]` i16 | Givens pair indices, **local to each 128-channel group** — a permutation of 0..127 per group per layer |
| `theta` | `[krot, K/2]` f16 | Rotation angles |
| `channel_scales` | `[1, K]` f16 | Stored **pre-inverted**: multiply activations by it, do not divide |

`krot` is 8 for this checkpoint. Unquantized and left fp16: the visual tower,
`linear_attn.in_proj_a/b`, `lm_head`. There are **no MTP or drafter tensors** — the external
DFlash2-FP8 drafter carries over from the MXFP4 stack unchanged.

The whole implementation rests on one identity, verified to 9.3e-16 on real checkpoint tensors:

```
y = x W^T = ((x * channel_scales) R^T) dequant(Q)^T
```

Rotate and scale the **activations** forward through layers 0..krot-1 within each 128-group, then
run a plain uint4-asymmetric group GEMM. (`R^T` on the weights would instead be flipped layers with
negated angles; doing it on activations is what makes a single fused prologue possible.)

## The rotation kernel

`par_kernels.h`, forked from the AutoRound symmetric-int4 kernels; every divergence is marked
`PARO`.

The rotation is a genuine Givens sweep, not a Hadamard:

```c
const int i = ij & 0xFF, j = ij >> 8;
const float xi = s_x[wave][i], xj = s_x[wave][j];
s_x[wave][i] = fmaf(c, xi,  sn * xj);
s_x[wave][j] = fmaf(c, xj, -sn * xi);
```

Each layer's 64 pairs **partition** the 128 channels of a group. That disjointness is what makes it
cheap: no workgroup can observe a partial layer, so successive layers need only an
`s_waitcnt lgkmcnt(0)`, never a barrier.

`pq_rotate_quant` does rotate + channel-scale + amax + e4m3 encode + code-domain row-sums in **one
launch**, replacing the `scaled_fp8_quant` launch that was already there — net zero added launches
on a stack whose launch gap is ~20% of decode. One workgroup per (partition, group, token-chunk).

`pq_rotate_quant2` (`RADIANCE_PQ_ROT_V2=1`, the default) is the same math with the pair records
held in registers (`unsigned long long rec[8][2]`) instead of staged through LDS, and no
`__syncthreads` at all. Bit-exact against v1.

The e4m3 encode and decode are **software** (RNE, OCP, saturating) rather than `v_cvt_pk_fp8`, so
the code-domain row-sum is by construction the sum of exactly what the codes decode to. Gated
exhaustively: 256-code round-trip plus nearest-value optimality.

Four more producers fuse the rotation into whatever computed the activation, so the rotation stops
being its own launch:

| Kernel | Fuses |
|---|---|
| `pq_add_rms_rot` | residual add + RMSNorm + rotate + quant, feeding the norm-fed linears (qkv, in_proj_qkvz, gate_up) |
| `pq_ew_rot` | silu-mul -> `down_proj`, per-head gated RMSNorm -> `out_proj`, attention gate -> `o_proj` |
| `pq_token_quant`, `pq_token_quant_tiled` | the prefill per-token pass (tiled writes the fragment layout the A-tiled GEMM reads) |

Together these remove 192 rotation launches per decode step and replace inductor's norm pair with
one kernel.

## The GEMM

The asymmetric zero point is kept **out of the matmul entirely**. The AutoRound `(c-8)` LUT is
retained, and the true value `(c - zp)` is written as `(c-8) - d` with `d = zp - 8`, so per group:

```
acc += asg * ( sc * WMMA(a, c-8)  -  (sc*d) * rowsum )
```

`sc*d` ("zscale") is interleaved with the scale in `SZ [K/128, N, 2]` f16 so it rides the same
4-byte load. Both `a*c` and `d*a` products are exact in fp32, so this moves only addition roundoff.
Weight-side traffic is `4 + 32/128 = ` **4.25 bits/weight — exactly MXFP4's**, which is why the two
formats land so close at the memory-bound end.

Activation scales are **per group, not per token**, in the decode band. That is what lets the whole
prologue be one kernel with no cross-workgroup sync, it folds into the per-slab rescale that
already exists, and it is finer-grained fp8 than a per-token scale. (The paper's kernel is W4A16;
per-group fp8 is the closest W4A8 gets.)

Three bands:

| Kernel | Band |
|---|---|
| `pq_int4_fp8_gemm_decode<DWN,KS,TM,...,WPERM,NT>` | `M <= RADIANCE_PQ_DECODE_MAX_M` (64). Split-K, per-group scales |
| `pq_int4_fp8_gemm_prefill<TN,...,PTOK,WPERM>` | Row-major A, per-token or per-group |
| `pq_int4_fp8_gemm_atiled<TN,WPERM,LBK,HOIST>` | Prefill with fragment-tiled A read straight from global into the WMMA registers, no A tile in LDS. Default |

`RADIANCE_PQ_WPERM=1` stores the weight in fragment order (one 32-lane u32 slot per
(n-tile, k-step)) rather than the loader's `[N, K/8]`. This flag and the kernel template argument
**must agree** or the weight is read as garbage — and fragment order is what makes the decode
kernel's non-temporal loads (`RADIANCE_PQ_DECODE_NT`) pay at all.

The M branch lives inside `_linear_impl` (`radiance_paroquant.py`) rather than in the traced graph,
so dynamo never sees it.

## Merged linears and partition select

Rotations are **per projection**, so a merged linear carries P differently-rotated copies of the
activation. The GEMM derives the partition from two boundary columns and indexes into them:

```c
const int prt = (n0 >= pb1 ? 1 : 0) + (n0 >= pb2 ? 1 : 0);
A += (size_t)prt * M * K;  ASG += (size_t)prt * M * G;
```

One launch either way. Attention QKV is P=3; `gate_up` and the GDN `in_proj` are P=2.

qwen3_5's `in_proj_qkvz` arrives as four vLLM partitions (q, k, v, z) but q/k/v share the
`in_proj_qkv` rotation, so the loader dedups identical adjacent rotations into runs. The GEMM's
limit of 3 applies to *distinct* rotations, not to vLLM partitions — vLLM 0.27.1 requires accepting
more than three at `create_weights` and deduping at `process_weights_after_loading`.

The load path asserts, rather than assumes:

1. `group_size` is 128 and K per partition stays a multiple of 128 under TP.
2. `bits` is 4 and `krot <= 8` (the prologue's table is sized for 8).
3. Partition boundaries land on multiples of 128 (the decode n-block), so no GEMM block straddles
   two rotations.

## Checkpoint loading

`radiance_paroquant.py` registers the method through vLLM's **public plugin hook** —
`@register_quantization_config("paroquant")` — so no file in vLLM is edited and no fork is needed.
`config.json` declaring `quant_method: paroquant` is enough to route the model to it.

`create_weights` registers the AWQ buffers as `PackedvLLMParameter` / `GroupQuantScaleParameter`
(packed factor 8) and the rotation buffers with a custom `_rotation_weight_loader`. TP row-sharding
is a plain `narrow` and is **exact**, because pair indices are local to each 128-group and TP
boundaries are multiples of 128.

`process_weights_after_loading` undoes the AWQ nibble reorder (`_AWQ_INV = [0,4,1,5,2,6,3,7]`, the
argsort of AWQ's `0,2,4,6,1,3,5,7`), repacks to the kernel's `[N, K/8]` u32 layout, optionally
applies `permute_w` for fragment order, builds the interleaved `SZ`, dedups the rotation runs, and
materialises `rec [P, krot, K/2, 4]` as `{i | j<<8, cos f16, sin f16, 0}`.

Modules skipped to fp16 are matched by regex (`.*visual.*`, `.*in_proj_a.*`, `.*in_proj_b.*`),
extendable with `RADIANCE_PQ_SKIP`.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `RADIANCE_PAROQUANT` | `1` | Registration is import-time; this only gates logging |
| `RADIANCE_PQ_DECODE_MAX_M` | `64` | Top of the decode band. Must cover `MAXSEQS x (SPEC+1)` rows or the widest verify batches fall onto the prefill tile |
| `RADIANCE_PQ_WPERM` | `1` | Fragment-order weight layout. Must agree with the compiled kernel |
| `RADIANCE_PQ_DECODE_NT` | `1` | Non-temporal decode loads. Honoured only under `WPERM` |
| `RADIANCE_PQ_ATILED` | `1` | A-tiled prefill GEMM |
| `RADIANCE_PQ_AT_LBK` | `128` | A-tiled K blocking |
| `RADIANCE_PQ_AT_HOIST` | `1` | Hoist the scale load out of the A-tiled inner loop |
| `RADIANCE_PQ_I8` | `0` | int8 activations on the int8 WMMA (scale amax/127, integer row-sums); per-group with `PG`. Keys the cache (`-i8`) |
| `RADIANCE_PQ_PG` | `0` | Per-group activation scales on the A-tiled prefill band (fused rotate + group-quant producer). The best-fidelity configuration with `I8`; costs ~9% prefill vs per-token. Keys the cache (`-pg`) |
| `RADIANCE_PQ_ZPE` | `0` | Zero-point correction as a rank-G fp16 WMMA epilogue on the A-tiled band instead of one FMA per element per group in the loop: +6.5% prefill, numerics within rounding noise, any G. Keys the cache (`-zpe`) |
| `RADIANCE_PQ_PG_PRODUCER` | `3` | Per-group prefill producer: `3` = conflict-free ownership-layout rotate+quant (the Givens chain in LDS is bank-conflict-bound with the checkpoint's random pairs; this one is 1.9x, +11% prefill at 2k), `2` = pass-A records-resident, `1` = one workgroup per row. All byte-exact |
| `RADIANCE_PQ_PTOK` | `1` | Per-token activation scales above the decode band. `0` forces per-group everywhere: ~7% slower prefill, finer-grained fp8 (GSM8K 98.0 per-group vs 97.4 per-token — inside binomial noise) |
| `RADIANCE_PQ_ROT_V2` | `1` | Register-resident rotation records. Bit-exact against v1 |
| `RADIANCE_PQ_ROT_STREAM` | `1` | Fused add + RMSNorm + rotate + quant producers. **Changes the traced graph**, so it keys the cache directory (`-rs`) |
| `RADIANCE_PQ_ROT_STREAM2` | `1` | silu-mul / gated-norm / attention-gate producers. Needs `ROT_STREAM`; keys the cache (`-rs2`) |
| `RADIANCE_PQ_ROT_STREAM3` | `0` | All-reduce fused into the norm+rotate producer. Built and **measured slower** — see [Results](#results). Off |
| `RADIANCE_PQ_SKIP` | unset | Extra regexes for modules to leave in fp16 |
| `RADIANCE_PQ_CHECKALL` | unset | `"N:K,N:K"` — compare the kernel against an exact fp32 dequant. Needs `--enforce-eager`; eval only |
| `RADIANCE_PQ_CHECK_MAX_M` | `128` | Row cap for `CHECKALL` calls |

Compile caches validate on model plus torch/Triton version and **must not be shared across
checkpoints**; the launcher keys `~/.radiance-cache-paro-093` with a suffix per graph-changing flag.

MXFP4-only knobs are left at production values in the launcher but are inert here — no quark layers
load.

## Traps

- **Ubuntu ships `/usr/lib/python3.12/sitecustomize.py`, and it shadows any site-packages copy.**
  A `sitecustomize.py` dropped into site-packages is silently never imported and the quant method
  never registers. Append to the stdlib one. Each `podman run` starts from the pristine image, so
  the append does not accumulate.
- **A partition-mismatched activation tuple is a silent GPU hang, not a fault.** Feeding a
  single-partition tuple to a 2-partition layer makes the GEMM read past the activation buffer.
  Hence the partition-count guard in `_linear_impl`.
- **`CHECKALL`'s exact reference OOMs if it materialises int64 codes.** `[N, K]` on `gate_up` is
  ~700 MB and killed a 0.92-util worker when a new M first triggered it. It is chunked to 512
  columns in fp32 now, and remains an eval-boot tool.
- Split-K scratch is allocated from Python at first use and registered with the extension. A lazy
  `hipMalloc` from `launch()` lands inside CUDA-graph capture whenever the compile cache is warm.
- A stale `.so` in the working directory precedes site-packages on `sys.path`, so the launcher
  leaves the bind mounts before `exec`.

## Results

2 x R9700, TP=2, fp8 KV, R4D attention, DFlash2-FP8 drafter, `qwen-fixed-v22.3.jinja`.
MXFP4 production is the comparison column throughout.

| | MXFP4 prod | ParoQuant |
|---|---|---|
| GSM8K 500q | 97.8% | **97.4-98.0%** |
| decode @ ctx25 | 22.3 ms/step | 24.19 ms/step |
| combined decode | 186.0 t/s | **226.2 t/s** |
| conc 1/2/4/8/16 | — | 168 / 276 / 399 / **512** / 518 |
| prefill @ 2k-64k | — | 3782 / 3700 / 3725 / 3621 / 3450 t/s |
| KV cache profile | — | 622k tokens |

Prefill was the one real deficit, -8.8% at 8k when the stack first shipped, closing to parity from
around 64k; the A-tiled round and the rotation streams took the decode step from 25.74 to 24.19 ms
and the combined figure past MXFP4's.

Numerics gates, all passing: e4m3 round-trip and optimality exact; prologue bit-exact against a CPU
reference; GEMM relative error <= 1.7e-3 (bf16 output rounding) on every per-rank model shape
including 3-partition QKV and subnormal-heavy codes; semantic identity 9.3e-16 on real tensors;
module-level `CHECKALL` <= 4e-5 across all five layer families; **in-serve `CHECKALL` rel = 0.00000
on all five gated shapes, both TP ranks** at the boot shape, and <= 4e-5 at the wider M real
traffic reaches.

Measured against exact (unquantized) activations the relative error is ~2.6e-2. That is the e4m3
activation quantization itself — dot products do not average per-element fp8 noise down — and it is
the same class as the AutoRound W4A8 path.

**Rejected, so nobody rebuilds it:** rotation stream 3, the two-rank all-reduce fused into the
norm+rotate producer. Correct (bit-exact single-rank loopback, in-serve all-reduce check on every
call on both ranks, GSM8K 97.40) but slower — 25.10 vs 24.19 ms/step, 202 vs 226 t/s combined. With
one workgroup per row the rotation chains serialize on one CU and the push uses M CUs instead of
r4d's 24 blocks. Restoring the parallelism means either a spin-wait deadlock or a two-launch split,
which gives back the launch saving that was the point. The MXFP4 equivalent wins with the same
structure only because its epilogue has no rotation.

## MXFP4 weights: the zero-VALU loop

The int4 GEMM's prefill cost is linear in VALU per tile-group: 16 ops (the fp16 group scale FMA
and the zero-point FMA) run at 180-188 TF/s, 8 at 200, **0 at 225** -- MXFP4's loop. So the second
ParoQuant format keeps the learned rotations and changes only the weight grid to OCP MXFP4:
e2m1 elements, one e8m0 (power-of-two) scale per 32 along K. The e8m0 scale folds at weight
staging, e2m1 has no zero point, and the inner loop is fp8 x fp8 WMMA with nothing else in it --
the same W4A8 path AMD's MXFP4 release runs on. Weight traffic is unchanged at 4.25 bits/weight
(4 + 8/32 vs 4 + 32/128), so decode traffic is untouched; the decode *step* is a launch-count story, see below.

**Checkpoint** (`quant_method: paroquant_mxfp4`, `paroquant/build_hybrid.py`): per projection the
Quark MXFP4 buffers -- `weight [N, K/2]` u8 (two e2m1 per byte, even index in the low nibble),
`weight_scale [N, K/32]` u8 -- plus z-lab's `pairs`/`theta`/`channel_scales`. 18 GB, the same as
the int4 checkpoint and as AMD's MXFP4. The packing, scale bias and on-disk layouts match what
`radiance_mxfp4.py` already decodes, byte for byte.

**Where the rotations come from, and why not from scratch.** ParoQuant's optimizer initializes
rotations at identity. With the calibration this box can afford (256 samples x 2 epochs; 60 GiB
of RAM holding a 55.6 GiB fp16 model, so 512 samples thrash), the rotation stage stops helping
past layer ~10 and leaves angles at *exactly zero* on many layers -- layer 28 came out 100% zero
against z-lab's 3.1 rad mean. That run was MXFP4 plus weight fine-tuning behind a prologue that
rotates by nothing. The hybrid instead takes the bf16 base weights, applies z-lab's trained
rotations through the same kernel the optimizer uses (their dequantized int4 reproduces
`rotate(W_bf16 * cs)` to within int4 noise: 0.104 vs 0.1025 expected, inverse rotation 1.41), and
quantizes to MXFP4 -- 400 modules in 91 s. A rotation that tames per-group outliers for int4 does
the same for e2m1. `STAGE=finetune` then optimizes only the weights and the per-block e8m0 bias
under the MXFP4 grid with those rotations frozen: layer 0 went 1.92e-5 -> 3.02e-6 (-84%).

**Shared exponent.** AMD's release uses the OCP `floor(log2(amax)) - 2` rule, which clips the
block maximum (their per-block maxima are only 4.0 and 6.0, never 3.0). A never-clip rule
measured *worse* on real weights (+1.9% RMSE, every tensor sampled), so `ocp` is the default and
a comparison against AMD's checkpoint isolates the rotations.

**Serving** (`paroquant/radiance_paroquant_mxfp4.py`): the int4 module's per-token prologue
(`pq_rotate_quant` -> `pq_token_quant`) composed with `mxfp4_linear_pq`, once per distinct rotation
on its N-slice; weight prep mirrors the MXFP4 kernel class (scale transpose, per-row reference
exponent, fragment order). The prologue's tiled writer and the MXFP4 GEMM's tiled reader use the
same fragment layout -- `[m-tile][k-step][half][row 16][8 B]` -- so the A-tiled prefill path is
taken above `RADIANCE_MXFP4_A_TILED_MIN_M` with no relayout. `RADIANCE_PQM_CHECKALL` gates each
partition against an fp32 dequant. Serve with the same launcher:
`MODEL_DIR=Qwen3.8-27B-PARO-MXFP4-ft RADIANCE_PQ_ROT_STREAM=0 RADIANCE_PQ_ROT_STREAM2=0`. As prod: the
systemd unit `paroquant/qwen_vllm_paro_mxfp4.service` (its own container name and compile-cache dir;
`vllm-switch paromx`), with the int4 unit left intact so `vllm-switch paro` is the rollback.

**Gated so far (2026-09-08):**

| | |
|---|---|
| in-serve `CHECKALL`, fine-tuned checkpoint, TP=2, **real inputs** | **rel 0.0012-0.0021** on every shape and partition (gate_up P=2, in_proj_qkvz P=2, qkv P=3, down, o/out_proj; 110 calls each) = the bf16 output-rounding floor. Profile-run calls (zero activations) are labeled `zero-input`; an earlier all-zero "pass" was those, and a wrapper bug (double un-permute under WPERM) had reported rel 1.6-9 on real inputs |
| loader vs fp32 reference | rel 0.011-0.012 at M = 1 / 5 / 40 / 64 (decode), 200 (prefill), 600 / 2048 (A-tiled) -- the e4m3 activation floor |
| GSM8K 500q, one-shot hybrid, pseudo (W4A16 upper bound) | 96.96% (479/494) |
| GSM8K 500q, **fine-tuned**, pseudo (W4A16 bound) | 97.20% (486/500), 0 errors |
| GSM8K 500q, **fine-tuned, SERVED W4A8 path** (TP=2, this loader) | **97.60%** (488/500), 0 errors, 1 truncated -- int4 PARO 97.60 (09-02, same template), AMD MXFP4 97.8; n=500 sigma ~0.76 pt. 17.8 min vs 23 for the fp16 pseudo serve |
| weight-space error vs bf16, one-shot | +4-9% over z-lab's fine-tuned int4 (e.g. L0 down_proj 0.127 vs 0.118) -- what the fine-tune targets |

**Speed, kernel path isolated** (one R9700, TP=1, eager -- same launcher and mode for both
arms, both on their A-tiled prefill paths; eager numbers are far below prod's TP=2 compiled
serve, the *ratio* is the point):

| prefill tokens | int4 PARO | MXFP4-PARO | |
|---|---|---|---|
| 2k | 2217 tok/s | **2434** | +9.8% |
| 8k | 2223 | **2411** | +8.5% |
| 16k | 2186 | **2334** | +6.8% |
| 30k | 2095 | **2224** | +6.2% |

The prefill gain is the zero-VALU loop's +25% on the GEMM diluted by the unchanged prologue,
attention and everything else in a prefill step.

**Prod decode: the launch-count regression and its three fixes** (2026-09-08; compiled graphs,
DFlash2 SPEC=7, TP=2, `bench_decode_ctx` + BetterBench decode single pass, same tools as the int4
records). The first prod boot of this loader decoded at **35.40 ms/step** against int4 PARO's
24.19 -- with acceptance at parity (acc/draft 1.85 vs int4's 1.57-1.78 on the same bench) and the
GEMM at parity (`paroquant/bench_linear_tp2.py`, real TP=2 shapes). The whole gap was launch count:
two prologue kernels per linear, unfused norm / silu / gate kernels, one GEMM per rotated partition,
and an output copy per partition. hipGraph hides CPU cost, not kernel count.

| prod build | ms/step @ctx25 / 8k / 32k | BetterBench combined | GSM8K 500q |
|---|---|---|---|
| int4 PARO (reference) | 24.19 / 25.74 / 26.70 | 226.2 t/s | 97.4-98.0 |
| MXFP4-PARO v1 (two-launch prologue, no streams) | 35.40 / 36.43 / 37.53 | 126.3 | 97.60 |
| + `pq_rotate_tokquant` (one launch: scale + rotate + token amax + e4m3) | 28.48 / 30.23 / 30.85 | 184.4 | -- |
| + per-token stream producers (`pq_add_rms_rot_tok`, `pq_ew_rot_tok<0/1/2>`) | 26.29 / 27.51 / 28.34 | 200.3 | 97.40 |
| + single-launch merged GEMM (partition select in-kernel) | **24.53 / 25.89 / 26.80** | 203.4 | 97.40 |
| + scale slabs dropped, `GPU_UTIL=0.95` (shipped) | 24.71 / 25.98 / 26.96 | -- | KV 862k tokens |
| + fused TILED prologue for the A-tiled band (shipped) | 24.53 / 26.43 / 26.97 | -- | prefill +5-9%, see below |
| + skinny split-K bf16 GEMM for the GDN gate projections (`RADIANCE_SKINNY_GEMM=all`, shipped) | **23.38 / 24.91 / 25.80** | 216.3 | 97.40 |

The same two changes went back to the int4 PARO unit on 2026-09-09: skinny gate GEMM + `GPU_UTIL=0.95` took it from 24.09 / 25.87 / 26.43 to **23.27 / 24.79 / 25.59 ms/step** (KV 812k -> 854k, GSM8K 97.40); the fused single-launch prologue (`pq_rotate_tokquant<..., WRS>`, byte-exact incl. row-sums, harness `tokqrs`) is on by default but served prefill moved only +0.1-0.5%: int4 prefill is bound by the GEMM's zero-point fold, not the prologue.

Each step is gated bit-identical to the path it replaced (`par_harness --bench2 tokq` and
`tokstream`: 54 + 45 shapes, codes / scales / hs / residual byte-exact) and output-identical at the
loader level (`test_mxfp4_loader.py`: stream tuple vs plain path `torch.equal` at every site and M).
The single launch is *not* bit-identical to the per-partition loop -- the decode band picks split-K
from the launch's N, so the fp32 partials reassociate -- and measures 1-2 one-ulp bf16 flips per
million outputs (rel 3e-8 .. 4e-6), the same class as any split-K change.

- **Stream plumbing is shared.** `radiance_paroquant.install_stream` now asks each consumer linear's
  quant method for its producers (`quant_method.stream_norm` / `stream_ew`), so the int4 method
  hands its GEMM the per-group tuple and `paroquant_mxfp4` hands its GEMM `(A [P,M,K], AS [P,M])`.
  Stream 3 (fused all-reduce) stays int4-only (`pq_ar_capable`). Same guards, same patched
  forwards, same "installed: 64/64/64/48/16" line.
- **Per-token producers need the whole row in one workgroup** (the scale is a row max), so they
  cannot split rows across workgroups the way the per-group producers do; the small-M cost is the
  serial rotation chain per wave (5 chains per wave at 8 waves for K=5120). Waves per row are chosen
  by M -- 32 at M<=16, 16 at M<=64, 8 above (`RADIANCE_PQ_TOK_WAVES` forces one) -- which takes the
  norm producer from 10.5 to 7.9 us and the down-proj producer from 17.2 to 9.7 us at M=8. They are
  still 2-4 us behind the int4 producers per site (5.4 / ~6 us); interleaving two chains per wave is
  the untried lever.
- **Single-launch merged GEMM.** `radiance_mxfp4_fp8` kernels (decode, folded prefill, A-tiled) take
  `pb1, pb2, astride`: an n-block in `[pb1, pb2)` reads rotated copy 1 of A and its scales, etc.
  Boundaries must be 128-aligned (loader-checked; every Qwen3.8 shape at TP=1/2 is). The stock entry
  `launch` passes `1<<30` and is bit-identical to before. At M=8 on the real TP=2 shapes this took the
  per-step linear cost from 10.9 to **9.2 ms** (int4: 12.7): qkv P=3 51.7 -> 28.6 us, gate_up 67 -> 59,
  in_proj 40 -> 30; the P=1 sites also lost their output copy (27 -> 23 us).

Prod prefill (BetterBench prefill sweep, single pass): the single-launch build measured
4376/4449/4423/4291/4068 PP t/s at 2k/8k/16k/32k/64k; the **fused tiled prologue** (one workgroup per
row writes the fragment-tiled A directly, `pq_rotate_tokquant<W, TILED>`, byte-exact vs pass A +
tiled pass C and 1.3-1.4x faster at M>=600) took it to **4770/4827/4649/4495/4273** (+9%/+8%/+5%/+5%/+5%), which is
+26%/+30%/+25%/+24%/+24% over int4 PARO's 3782/3700/3725/3621/3450. The stream producers emit the tiled tuple in
that band too, so an A-tiled linear now runs zero prologue launches. Two-chain interleaving in the
per-token producers was built and measured neutral at the wave counts in use (kept dark, `IL`).

Where it stands: step time at parity with int4 PARO; combined BetterBench 203 vs 226 on a single pass
(tokens per update trail on prose / file_edit, step time does not -- acceptance on this checkpoint,
not the kernels); prefill +6-10%. **Open:** the producers' 2-4 us per site; KV profile had come out at
578k tokens vs int4's 622k on the same `GPU_UTIL=0.92` (consumed memory 17.0 vs 16.1 GiB between the
stream and single-launch boots; the duplicate per-partition scale slabs, 0.4 GiB/rank, were part of
it) -- with the slabs removed and the unit at `GPU_UTIL=0.95` it is **862k tokens** (3.29x
concurrency at 262k ctx), decode unchanged; SPEC re-sweep not redone -- step time and acceptance match int4 PARO, whose sweep chose 7.

**Fine-tune** (`STAGE=finetune`: weights + e8m0 bias under the MXFP4 grid, rotations frozen at
z-lab's, 256 samples x 2 epochs): mean -4.4% layer-output error per layer (-84% at layer 0, -1.7 to
-9% at depth). The optimizer keeps the whole fp16 model CPU-resident; on 60 GiB that thrashed, so
each finished block's storage is now released as the loop passes it (memory only). 64 layers in
~5 min each once that landed.

**Traps, so nobody re-hits them:** the pseudo config must not carry `torch_dtype: float16` (R4D
attention is bf16-only); `GPUS=<one card>` needs `HIP_VISIBLE_DEVICES` to index into the
ROCR-filtered set; the `RADIANCE_MXFP4_*` kernel knobs must reach the container or the GEMM runs
with fragment order off and the decode band disabled; two gate scripts must never share a port
(one stopped the server under the other's last six requests -- the "errors" in that run); the
`RotateQuantizedLinear` + `strict=False` export path silently writes zeros for MXFP4 buffers.

The full change-by-change log, including the prefill ablation ledger and the SPEC re-sweep, is in
[paroquant/RESULTS.md](paroquant/RESULTS.md).

### int5 W5A8 (2026-09-10/11)

Five-bit codes are exact in e4m3 as `(c - 16)`, so int5 rides the int4 kernel with a fifth-bit plane
(`pq_stage_w<..., BITS=5>`, `ar_unpack8_5`); checkpoint layout "int5-bitplane" (`build_int5.py`,
`convert_int5.py`, `requant.sh NBIT=5`). Fine-tune runs incrementally on this box from an fp16 base
(`to_fp16.py`: memory-mapped load, no swap) at ~9.5 min/layer.

| served, TP=2, SPEC=7 | ms/step @ctx25 / 8k / 32k | KV | prefill @2k / 64k | GSM8K | KL vs bf16 (wiki, top-256) | top-1 |
|---|---|---|---|---|---|---|
| PARO-MXFP4 (prod) | 23.38 / 24.91 / 25.80 | 862k | 4770 / 4273 | 97.40 | 0.048 | 90.0% |
| int4 PARO | 23.27 / 24.79 / 25.59 | 854k | 3808 / 3349 | 97.40 | -- | -- |
| int5 RTN | 26.07 / 27.69 / 28.49 | 767k | 3490 / 3137 | 97.80 | 0.032 | 91.5% |
| **int5 fine-tuned** | 26.29 / 27.90 / 28.71 | 767k | 3569 / 3298 | 97.40 | **0.027** | **92.2%** |

Weights-only (no activation quant) the int5 checkpoint is at 0.011 nats; the rest is the e4m3
activation element, a floor shared by every W*A8 build here. Per-group activation scales recover 6% of
it for 19% of prefill (off). Details and the PTOK=0 loader fix: paroquant/RESULTS.md 2026-09-10/11.

### Same-stack fidelity ranking (2026-09-11)

Reference = the bf16 base served on this stack (eval mode, small context); cross-image references carry
a ~0.02-nat offset (the two images' bf16 outputs differ by that much), so only same-stack numbers are
comparable. Wikitext, 96 x 500-char chunks, top-256 support:

| served build | KL | top-1 | ms/step @ctx25 | prefill @2k | KV |
|---|---|---|---|---|---|
| MXFP4-PARO (prod) | 0.042 | 90.3% | 23.3 | 4770 | 862k |
| int4 PARO | 0.029 | 91.5% | 23.5 | 3808 | 854k |
| int5 fine-tuned, e4m3 per-token | 0.013 | 94.3% | 26.3 | 3569 | 767k |
| int5 fine-tuned, int8 per-group tiled (`RADIANCE_PQ_I8=1 RADIANCE_PQ_PG=1`) | **0.0097** | **95.2%** | 26.0 | 3328 | 767k |
| int5 fine-tuned, int8 per-group tiled + zero-point epilogue (`... RADIANCE_PQ_ZPE=1`) | 0.0099 | 95.25% | 25.7 | 3550 | 769k |
| **+ conflict-free producer (`RADIANCE_PQ_PG_PRODUCER=3`, default)** | **0.0100** | **95.21%** | 25.9 | **3941** | 760k |

GSM8K is identical within noise on every row (int5 + ZPE: 96.8 and 97.8 on two samples). The zero-point
epilogue (2026-09-12) moves the `sc*(zp-16)` row-sum term out of the loop into Gp/16 fp16 WMMAs per output
tile, +6.2-6.7% prefill at 2k-64k with decode and KL unchanged. The per-group producer was
LDS-bank-conflict-bound (the checkpoint's random Givens pairs; 478 -> 254 us per qkv linear at 2k rows
with a per-layer ownership layout and load-time write matchings, byte-exact), another +11% at 2k. The last
row is the prod candidate: 2k prefill 3328 -> 3941 (+18%) on 2026-09-12 with KL, decode and GSM8K
unchanged. Details, the I8 mode, the per-group tiled band, the epilogue and the producer:
paroquant/RESULTS.md 2026-09-11/12.

### Distribution-level quality: KL divergence against the FP8 serve (2026-09-09)

Per-position top-20 prompt logprobs from both serves over the same text (`~/pibench-local/kld.py`;
the OpenAI API exposes at most the server's `--max-logprobs`, 20 on both launchers, and 20 is also
prod's `top_k`, so this is the support the sampler draws from). KL(FP8 || PARO-MXFP4) over the
reference's top-K, both renormalized; a lower bound where the candidate's list did not cover the
reference token (coverage column). The reference is the FP8 unit (`Qwen/Qwen3.8-27B-FP8`, fp8 KV),
so the number contains the FP8 serve's own deviation from bf16 too.

| corpus | positions | top-5 KL | top-10 KL | top-20 KL (coverage) | top-1 agreement |
|---|--:|--:|--:|--:|--:|
| wikitext-2 | 11,165 | 0.0422 nats | 0.0492 | 0.0573 (87.9%) | 90.53% |
| code test set | 14,153 | 0.0420 | 0.0482 | 0.0541 (83.4%) | 92.67% |
| served traffic (target's own chat/code answers) | 12,100 | 0.0339 | 0.0387 | 0.0435 (87.2%) | 91.93% |

That is the ordinary 4-bit band (llama.cpp reports ~0.02-0.05 mean KLD for Q4_K_M against fp16
and ~0.001-0.003 for Q8), and it is consistent with the task-level results (GSM8K 97.40-97.60 vs FP8's
97.8, inside binomial noise). The prompt-logprob path materializes full-vocabulary logits for the whole
chunk, which is why the FP8 reference at 0.92 memory utilization OOMs above ~600 prompt tokens; it
was collected from a 0.80-utilization copy of its launcher with 1,500-character chunks, and the
chunking must be identical on both sides because positions are compared pairwise.
