# MXFP4 MoE on gfx1201: native experts, split-KV verify, prefill attention, MTP depth, exact GDN scan, W4A8 prefill, int2 draft head, fp8 dense layers, fused gate

Nine independent additions, each opt-in or scoped so the default Qwen3.8-27B serve is unchanged.
All numbers are from one Radeon AI PRO R9700 (gfx1201, 32 GB, TP=1) on the published image
`stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1). Each is a single run unless noted.
Sections 1-3 were measured with the launcher's old defaults (MTP-8, 8 sequences, GPU utilization 0.95).
Sections 4-9 were measured with the new ones (MTP-4, 16 sequences, 0.97), on a launcher that also ran
decode attention on AITER's unified kernel, which is not part of this repo; section 10 repeats the key
numbers with `serve-moe-mxfp4.sh` itself. Sections 7-9 were measured on top of sections 4-6 (exact scan and
W4A8 on), and 8 and 9 on top of 7: the int2 draft head was on in every arm of their A/Bs.

| Addition | Files | Default |
|---|---|---|
| MXFP4 MoE experts on gfx1201 | `patch_quark_moe_w4a16.py`, `patch_gfx12_aiter_a16w4.py`, `serve-moe-mxfp4.sh` | on in `serve-moe-mxfp4.sh` (`MOE_FIXES=1`) |
| Split-KV attention for spec-decode verify | `patch_attn_3d_multiq.py`, `moe-tests/test_attn_3d.py` | applied by `serve-moe-mxfp4.sh` only |
| R4D prefill attention on the MoE lane | `patch_moe_prefill_attn.py`, `moe-r4d/` | on in `serve-moe-mxfp4.sh` (`MOE_PREFILL_ATTN=r4d`) |
| MTP draft depth and sequence capacity | `SPEC`, `MAXSEQS`, `GPU_UTIL` in `serve-moe-mxfp4.sh` | on: `SPEC=4`, `MAXSEQS=16`, `GPU_UTIL=0.97` (were 8, 8, 0.95) |
| Exact GDN chunk scan (libr4d #4) | `patch_gdn_scan_fix.py`, `moe-gdn2/` | on in `serve-moe-mxfp4.sh` (`RADIANCE_GDN_SCAN_FIX=1`) |
| W4A8 expert GEMMs for prefill | `patch_moe_w4a8.py`, `moe-w4a8/` | on in `serve-moe-mxfp4.sh` (`RADIANCE_MOE_W4A8=1`) |
| Draft-only int2 lm_head | `patch_moe_drafthead.py`, `moe-drafthead/` | on in `serve-moe-mxfp4.sh` (`RADIANCE_MOE_DRAFT_HEAD=int2`, only with `SPEC > 0`) |
| FP8 dense layers | `patch_moe_densefp8.py`, `moe-densefp8/` | on in `serve-moe-mxfp4.sh` (`RADIANCE_MOE_DENSE_FP8=1`) |
| Fused shared-expert gate | `patch_moe_densefp8.py`, `moe-densefp8/` | on in `serve-moe-mxfp4.sh` (`RADIANCE_MOE_GATE_FIX=1`) |

## 1. MXFP4 MoE experts on gfx1201

`amd/Qwen3.5-35B-A3B-MXFP4` (Quark W4A4, experts only; attention, GDN and shared expert stay bf16)
loads on gfx1201 but runs its experts through vLLM's **EMULATION** MoE backend. The only native W4A4
MoE kernel vLLM knows is AITER's CK one, which is CDNA-only. EMULATION dequantizes all 256 experts of
every layer on every forward pass: a flat 3.35 ms per layer, about 16 tok/s decode even with MTP-8.

**`patch_quark_moe_w4a16.py`**
- `RADIANCE_MOE_W4A16=1` rewrites the Quark `w_mxfp4_a_mxfp4` scheme to `w_mxfp4`. The experts then
  run weight-only, with bf16 activations. Decode is bandwidth-bound, so this costs nothing, and it is
  more accurate than 4-bit activations. It is not bit-identical to EMULATION.
- `RADIANCE_MOE_BACKEND=<name>` picks the MXFP4 lane for the quantized expert layers only. The global
  `--moe-backend` flag also reaches the bf16 MTP drafter's MoE, which rejects MXFP4 lane names
  (`moe_backend='triton_unfused' is not supported for unquantized MoE`).

**`patch_gfx12_aiter_a16w4.py`** puts the experts on AITER's Triton a16w4 grouped GEMM
(`AITER_MXFP4_BF16` → `AiterW4A16ExpertsMonolithic` → `moe_gemm_a16w4`). That kernel unpacks e2m1
inside the matmul loop. Everything is gated on `on_gfx12x()`, so gfx950 / gfx1250 / CUDA keep their
old code paths:
- accepts gfx12 in the experts class, for SiLU only;
- interleaves the w13 rows, scales and bias gate/up along N at load. Qwen checkpoints store
  `[gate; up]`, while the kernel's fused SwiGLU reads column pairs. Without this the output is
  garbage (cosine ~0 against a reference).
- uses the Triton StridedLayout weight prep that gfx1250 gets, not the CK shuffle;
- passes `limit=None` for SiLU (the stock fallback clamps at 7.0, a gpt-oss constant);
- installs a gfx1201 tile table. Aiter's default `bn512/w8` tile is pathological at prefill sizes
  here (12.6 ms vs 2.4 ms on gate_up at M=4096), and small M wants narrow N tiles or split-K.

One MoE layer, real layer-20 weights, including routing, µs:

| Tokens (M) | EMULATION | TRITON_UNFUSED (unpatched lane) | a16w4 (this patch) |
|--:|--:|--:|--:|
| 1 | 3576 | 370 | **57** |
| 8 (MTP verify) | 4098 | 409 | **198** |
| 1024 | 6509 | 1978 | 1352 |
| 4096 | 7755 | 5227 | 3908 |

End to end, the same stream benchmark on all three, 1 card, MTP-8, fp8 KV, TRITON_ATTN (tok/s):

| | EMULATION | TRITON_UNFUSED | a16w4 |
|---|--:|--:|--:|
| 1 stream, prose / code | 17.1 / 24.6 | 64.8 / 85.0 | **81.2 / 107.1** |
| 8 streams, aggregate prose / code | 32.6 / 42.7 | 216.7 / 270.8 | **252.1 / 321.0** |

For reference, llama.cpp (Vulkan) decodes the same model's MXFP4_MOE GGUF at 88.5 tok/s (tg128).

Correctness checks:
- The a16w4 path matches a torch reference with relative error ≤ 3.6e-3 for M = 1 … 4096.
- Greedy 64-token completions against TRITON_UNFUSED: 11/20 identical. The rest are fluent
  alternatives, from a different accumulation order.
- Tool-call and structured-output checks pass.

Decode is now bounded by the bf16 non-expert weights, about 3.9 GB read per token.

## 2. Split-KV attention for spec-decode verify

TRITON_ATTN's `unified_attention` picks its split-KV 3D kernel only when `max_seqlen_q == 1`. So MTP
verify batches (1+k query tokens per sequence) and drafter catch-up passes run the 2D kernel. At batch 1
on a GQA-8 model its grid is about ten workgroups walking the whole KV, and decode collapses with
context.

The kernel already supports several query tokens per sequence in 3D mode:
- q blocks resolve their sequence from `cu_seqlens_q`, and rows are causally masked per token;
- fully masked rows are guarded in `softmax_step`;
- `reduce_segments` maps each token back to its sequence.

What doesn't fit is the scratch: `softmax_segm_*` are indexed by query token but sized per sequence.
**`patch_attn_3d_multiq.py`** therefore makes two changes:
- the scratch gets `max(seq_threshold_3D, RADIANCE_ATTN_3D_TOKENS)` rows (default 128);
- 3D is allowed while `max_seqlen_q <= RADIANCE_ATTN_3D_MAX_Q` (default 16) and the batch's query
  tokens fit those rows. Anything larger keeps the 2D kernel.

`RADIANCE_ATTN_3D_MAX_Q=1` restores stock behavior. It is read per call.

Kernel test, `moe-tests/test_attn_3d.py`:
- compares 2D vs 3D vs an fp32 reference, with fp8 KV, 16q/2kv heads, head 256 and block 2224;
- includes sequences that end just past a segment boundary, so rows are fully masked in the last
  segment;
- result: **PASS**, worst error 5.4e-4. At 18,293 context the call goes 8930 → 475 µs (one sequence,
  q_len 1), 9082 → 1891 µs (batch 4) and 9281 → 3492 µs (batch 8).

End to end, Qwen3.5-35B-A3B MXFP4, a16w4 lane, MTP-8, decode tok/s:

| Context | Unpatched | Patched | No speculation |
|--:|--:|--:|--:|
| 828 | 72.5 | 78.0 | 79.8 |
| 6,148 | 42.2 | **73.8** | 74.5 |
| 18,285 | 21.9 | **67.5** | 64.8 |

Aggregate at 8 streams also improves: 252.1 → 268.7 prose, 321.0 → 329.9 code.

`serve-mxfp4.sh` does not apply this patch. The 27B serve uses R4D attention for the target, and its
behavior stays byte-identical.

## 3. Prefill attention on the MoE lane (`MOE_PREFILL_ATTN`)

With the MoE fixes on, attention becomes the prefill bottleneck: on `amd/Qwen3.5-35B-A3B-MXFP4` it is
63% of prefill GPU time at 4k and 87% at 16k. TRITON_ATTN's 2D kernel runs a prefill chunk at
2.7 TFLOP/s at every context length, for two reasons:

- **Launch config.** On gfx1201 it picks BLOCK_M 16 (2 query tokens x 8 heads per program) with
  num_stages 2. The ISA shows 256 VGPRs, 182 of them spilled (552 B of scratch).
- **fp8 KV is decoded in software.** Triton 3.6 has no hardware fp8 convert on this target: about 15
  VALU ops per element plus a software f32-to-bf16 round. The same kernel on a bf16 cache is 10.5x faster.

libr4d's paged prefill kernel covers this. Its GQA 6 limit is a template argument, not a design limit
(384 rows = 48 query tokens x 8 heads per workgroup still divides), so `moe-r4d/r4d_moe.hip`
instantiates the unchanged v0.5.0 kernel at GQA 8 as a small pybind module. `moe-r4d/build.sh`
compiles it at container start with the image's hipcc (a few seconds) against a libr4d v0.5.0
checkout, which the launcher clones once into `R4D_SRC`. No libr4d source is copied into this repo.

**`patch_moe_prefill_attn.py`** (`RADIANCE_MOE_PREFILL_ATTN=r4d`) lets the R4D backend accept GQA 8.
Request runs of at least 17 query tokens go to the GQA-8 prefill kernel. Everything else goes to stock
Triton on a sub-batch, including `patch_attn_3d_multiq.py`'s split-KV kernel: decode, the 9-token
MTP-8 verify, drafter catch-up and graph capture. The R4D decode kernel holds 8 query tokens at GQA 8,
one short of an MTP-8 verify, so it is never used. The launcher switches the model and the MTP drafter
to `--attention-backend R4D`, which brings the HND cache layout and 16-token kernel blocks.

Kernel alone, one 2,224-token prefill chunk, paged fp8 KV (error vs fp32: 1.4-1.8e-3 R4D,
1.7-2.3e-3 stock):

| Context | Stock Triton | R4D at GQA 8 |
|--:|--:|--:|
| 2,224 | 14.7 ms (2.8 TFLOP/s) | 0.51 ms (79) |
| 16,384 | 200.7 ms (2.8) | 5.0 ms (112) |
| 32,768 | 419.9 ms (2.8) | 10.3 ms (112) |

End to end, `serve-moe-mxfp4.sh`, `MOE_PREFILL_ATTN=off` vs `r4d` (localeval speed, fresh-nonce
prompts, median of 3):

| Prompt | TTFT off | TTFT r4d | Prefill tok/s off | Prefill tok/s r4d |
|--:|--:|--:|--:|--:|
| 4k | 1.68 s | **0.38 s** | 2,434 | 10,760 |
| 16k | 11.07 s | **1.83 s** | 1,504 | 9,128 |
| 32k | 42.53 s | **4.33 s** | 797 | 7,833 |

- Decode is unchanged within noise: +0.7% / +8.7% at 4k / 16k in the sweep. On 64-token answers over
  4k / 16k / 32k code, medians go from 87.6 / 81.3 / 64.6 to 80.9 / 82.8 / 66.2 tok/s.
- Quality is unchanged within noise: gsm8k (50, thinking on) 0.96 to 0.98. A 24-prompt judged set
  went from 0.923 to 0.904, a paired difference of -0.023 against a run-to-run band of +/-0.032.
- This exact launcher, cloning and building from scratch, measured 0.41 / 1.85 / 4.31 s TTFT at
  4k / 16k / 32k once warm. The first 16k requests after a cold start pay one-time compilation.

`MOE_PREFILL_ATTN=off` restores TRITON_ATTN for everything. The patch also has a `triton` mode (tuned
tiles, BLOCK_M 128 / 8 warps / 1 stage, about 11x on the kernel). It still spills, and it was never
served, so the launcher does not expose it.

## 4. MTP draft depth and sequence capacity (`SPEC`, `MAXSEQS`, `GPU_UTIL`)

In align mode vLLM reserves `2 + num_speculative_tokens` GDN state blocks for every request
(`MambaSpec.max_memory_usage_bytes` in `vllm/v1/kv_cache_interface.py`; `num_speculative_blocks` is set in
`vllm/model_executor/layers/mamba/abstract.py`). The draft depth therefore decides how many requests fit
long before the pool is full of tokens. At MTP-8 one short request pinned 12.7% of the KV pool and only 7 fit.
The 8th queued, so 8 streams ran slower than 6. The launcher defaults are now `SPEC=4`, `MAXSEQS=16`,
`GPU_UTIL=0.97` (they were 8, 8, 0.95). `MAXSEQS=16` alone cost about 0.95 GiB of KV at 0.95, and 0.97 more
than gives it back.

localeval speed, 1k-token prompts, 1024 forced output tokens, one run per cell:

| | before: MTP-8, 8 seqs, 0.95 | MTP-4, 16 seqs, 0.95 | **MTP-4, 16 seqs, 0.97** |
|---|--:|--:|--:|
| KV cache | 5.18 GiB (242k tokens) | 4.23 GiB (249k) | **5.72 GiB (337k)** |
| KV pinned per short request | 12.7% | 8.6% | 6.4% |
| Requests that fit | 7 | 11 | **15** |
| 1 stream, tok/s | 105.8 | 138.2 | 122.8 (noisy) |
| 2 / 4 streams, aggregate tok/s | 162.7 / 255.4 | 195.0 / 274.1 | 177.2 / 308.1 |
| 8 streams, aggregate tok/s | 301.6 (1 queued) | 431.1 | **456.2** |
| 12 / 16 streams, aggregate tok/s | - | 477.3 (11 fit) / - | 569.3 / 534.1 (15 fit) |

- Acceptance drops from about 3.0 to 2.86 tokens per step at depth 4. Aggregate throughput still rises
  because more requests run at once.
- gsm8k (200, `--nonce`, thinking off): 0.380 against 0.365 at the old defaults, noise.
- No traceback or OOM line under 16-stream load. At 16 streams the 16th request waits and runs alone at the end.
- Measured on a separate production launcher (R4D prefill, decode attention on AITER's unified kernel, a GDN
  repair post-pass), not on `serve-moe-mxfp4.sh`. Section 10 has this launcher's own numbers.

**No-go: a larger CUDA-graph size.** A full batch is 16 x 5 = 80 tokens per decode step, above the largest
capture size (72). Adding 80 to `cudagraph_capture_sizes` made steps faster (+15% per stream at 16 streams)
but graph memory profiling reserved 1.1 GiB more: KV went from 5.72 GiB / 337k tokens to 4.63 GiB / 273k.
More requests queued, and aggregate throughput fell:

| streams | capture sizes up to 72 | with 80 added |
|--:|--:|--:|
| 12 | 585.5 | 568.9 |
| 14 | 609.5 | 488.5 |
| 16 | 560.3 | 463.4 |

`MAXSEQS=14` (14 x 5 = 70 fits the captured sizes) is an untested alternative. `SPEC=8 MAXSEQS=8 GPU_UTIL=0.95`
restores the old defaults.

## 5. Exact GDN chunk scan (`RADIANCE_GDN_SCAN_FIX`)

libr4d v0.5.0's `r4d_gdn_chunk_scan_k128_v128_c64_bf16` (libr4d issue #4) carries the in-chunk decay on the
chunk's midpoint `c = (G_first + G_last) / 2`: `e^{G_i-G_j} = e^{G_i-c} * e^{c-G_j}`. Each factor is clamped at
e^80, so any (sequence, head) whose chunk span `G_first - G_last` exceeds 160 gets finite but wrong outputs and
carried state. On this model that is common: 67 of 960 GDN heads exceeded it over about 25k tokens of real
prompts.

`radiance_gdn2` is the same kernel without the split. It never splits a factor that is <= 1:
- diagonal 16x16 tiles use elementwise `e^{min(G_i-G_j, 0)}`;
- below-diagonal tiles use `e^{G_i-c_J} * e^{c_J-G_j}`, with `c_J` the G at the end of the tile's 16-token block,
  so both factors are <= 1;
- the state path uses `e^{G_last-G_t}` directly.

`G` is a cumsum of non-positive values, so every exponent is <= 0. Nothing can overflow, and an underflow to 0
is the right value at fp32 precision. The extra tile fits in the LDS the kernel already has (59.5 of 64 KB), so
no barrier and no buffer are added. The ABI is r4d's 18 arguments, so `radiance_gdn.py` calls it from the same site.

**How it ships.** No libr4d source is stored in this repo, and the module is not a copy of libr4d's file.
- `moe-gdn2/radiance_gdn2_vs_v050.patch` is a unified diff against libr4d v0.5.0's
  `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip`. It carries the lines it adds, the 29 lines it removes, and 3 lines
  of context around each hunk.
- `moe-gdn2/build.sh` runs at container start. It checks that `R4D_SRC` is libr4d 0.5.0 and that the scan and
  `r4d_gdn_wmma.h` have the v0.5.0 md5s, applies the patch to a temporary copy of the scan
  (`moe-gdn2/apply_patch.py`, because the image has neither `patch` nor `git`), checks the patched file's md5,
  and compiles it with the image's hipcc into site-packages. Any mismatch is fatal.
- `patch_gdn_scan_fix.py` binds `radiance_gdn`'s `_CHUNK_SCAN` to it under `RADIANCE_GDN_SCAN_FIX=1`. A missing
  module is an import error at startup, never a silent fallback to the inexact scan. The log shows
  `[radiance.gdn] exact chunk scan ON` in the API server and in the engine core.
- The launcher needs the libr4d checkout for this even with `MOE_PREFILL_ATTN=off`, and applies it with
  `MOE_FIXES=0` too, because the GDN scan runs in every mode.

Error against an fp64 token-by-token recurrence (36 cases on the MoE 32/16 and dense 48/16 head layouts: decay
0.02-92 per token, mixed fast/slow heads, partial chunks, spans of exactly 160 and 170, batches, random beta and
initial state), relative error of output / state:

| case | libr4d v0.5.0 scan | exact scan |
|---|--:|--:|
| not exposed (span <= 160) | 0.29-0.34% / 0.16-0.25% | same, bit-identical |
| span 163.8 (2.6 per token) | 15.0% / 84.8% | 0.27% / 0.13% |
| span 201.6 (3.2 per token) | 46.0% / 100% | 0.26% / 0.09% |
| 8, 50, 92 per token | 82-99% / 100% | 0.23-0.24% / 0.00% |
| **max over 36 cases** | 98.6% / 100% | **0.343% / 0.253%** |

The remaining error is the kernel's own bf16 staging. The stock instantiation built from the same source is
bit-identical to the image's `r4d.so` in all 36 cases.

In the served model (per-head error against an fp64 recurrence, about 25k tokens of prompts, 420 scan calls),
whole-model output error goes from **7.45%** to **0.18%**, normed output from 26.96% to 0.12%, and state from
18.38% to 0.17%. That is the bf16 baseline.

Cost, localeval speed (128 forced tokens, 5 reps, fresh nonce):

| Prompt | libr4d scan | exact scan |
|--:|--:|--:|
| 4.1k | 10,842 tok/s | 10,894 (+0.5%) |
| 16.7k | 9,145 | 9,196 (+0.6%) |
| 33.9k | 7,974 | 7,982 (+0.1%) |

- The scan is 2-3% of MoE prefill, and the exact kernel runs at -4% to +10% of the stock one (single sequence,
  2,224 and 4,096 tokens, both head layouts), so the served difference is within noise.
- gsm8k (200, `--nonce`, thinking off): **0.375 against 0.215** (and 0.210 / 0.195 on repeats), +0.160 with a
  2-SE band of 0.090. The unfixed scan is also unstable between conditions: without a nonce it scores about as
  well as the fixed one, with a nonce it drops. The cause was not isolated.
- Container start took about 30 s longer on that production launcher (126 s against 91-96 s), because the
  module compiles at start.
- A post-pass that replays the flagged heads in fp32 is exact too, but costs 9-16% of prefill. This repo does
  not carry one.

`RADIANCE_GDN_SCAN_FIX=0` runs libr4d's scan as is, and then no libr4d checkout is needed unless
`MOE_PREFILL_ATTN=r4d`.

## 6. W4A8 expert GEMMs for prefill (`RADIANCE_MOE_W4A8`)

With the attention fix, the expert GEMMs are the largest block of MoE prefill. A rocprofv3 trace of one
request on an idle server (busy kernel time):

| Prompt | Steps | Expert GEMMs | Reduce | Dense bf16 GEMMs | Attention | GDN |
|--:|--:|--:|--:|--:|--:|--:|
| 3,991 | 1 | 42.0% | 2.8% | 32.3% | 4.8% | 6.0% |
| 15,979 | 7 | 42.8% | 2.2% | 27.3% | 13.8% | 4.7% |

AITER's a16w4 kernel feeds the 16-bit WMMA. gfx1201's fp8 WMMA has twice the peak (325 against 160 TFLOP/s),
so a W4A8 kernel (MXFP4 weights, per-token e4m3 activations) can win where the GEMMs are compute-bound. Triton
did not get there. `tl.dot` on e4m3 does emit the fp8 WMMA in this image, but converting e2m1 to e4m3 in
registers ran at 0.6-0.75x of a16w4, and the upper bound (weights pre-converted to e4m3, no dequant, twice the
weight bytes) was only 1.55-1.85x.

**`moe-w4a8/radiance_moe_w4a8.hip`** is a grouped HIP kernel derived from this repo's own
`radiance_mxfp4_fp8.hip` (LDS staging, `v_perm` e2m1-to-e4m3 upconvert with the MX block exponent folded in,
fragment layout), extended to the expert-grouped schedule of the a16w4 lane:
- `grid.y` walks AITER's routing `block_pid_map`, `grid.x` covers N blocks. It reads the same routing,
  weights and sorted buffers as the a16w4 lane, and its output goes through the same `reduce_grouped`.
- w13 (gate_up): rows are gathered from per-token fp8 activations. The gate row 2j and up row 2j+1 of the
  interleaved weight land in two column tiles of one wave, so SiLU(gate) * up is lane-local in the epilogue.
- w2 (down): per-row fp8 quantization of the intermediate, with the router gammas folded into the row scale.
- The e2m1 weights are upconverted losslessly to e4m3 with the block exponent taken relative to the row's
  maximum exponent (`Wref`, 0.75 MiB per layer, built on first use). That is exact for exponent gaps up to 14.
  On this checkpoint every block with nonzero weights has a gap of 11 or less.
- Tile configs are chosen per routing `block_m` (128x128 tiles, BK 128 or 64, a skip mode that drops padded 16-row tiles).

**`patch_moe_w4a8.py`** (`RADIANCE_MOE_W4A8=1`) inserts the dispatch into vLLM's
`aiter_triton_kernel_w4a16_moe_forward`. Calls with at least `RADIANCE_MOE_W4A8_MIN_TOKENS` tokens (default
1025, where AITER's routing reaches `block_m` 64: prefill chunks) take the W4A8 kernel. Everything else stays
on a16w4: decode, MTP verify, short prompts, every CUDA-graph capture, and any call with bias, clamp or
router-weight-on-input. At those sizes the expert GEMMs are weight-bandwidth bound (535-548 GB/s of
weight traffic at 60 and 80 verify tokens), so fp8 activations cannot speed them up. It anchors on code that
`patch_gfx12_aiter_a16w4.py` inserts, so it needs `MOE_FIXES=1`. A module that was asked for and is missing is
a startup failure. The kernel compiles at container start (about 5 s).

Per layer, layer-20 weights, cold cache, including activation quantization and the reduce (us):

| Tokens | block_m | a16w4 (gate_up + down) | W4A8 (quant + w13 + quant + w2 + reduce) | Layer speedup |
|--:|--:|--:|--:|--:|
| 1,056 | 64 | 913 + 530 = 1,443 | 1,044 | 1.38x |
| 1,536 | 64 | 1,048 + 625 = 1,673 | 1,181 | 1.42x |
| 2,224 | 128 | 1,798 + 1,063 = 2,861 | 1,404 | 2.04x |
| 3,991 | 128 | 2,337 + 1,443 = 3,780 | 2,040 | 1.85x |

Below block_m 64 the kernel is weight-streaming bound and loses to a16w4 (1.5-3x slower on w13 at 300-1,000
tokens), which is why the threshold exists. Served, the expert path takes 77 ms instead of 149 ms of a
3,991-token prefill and about 350 ms instead of 724 ms of a 15,979-token one.

End to end, A/B on the same stack with the exact scan (localeval; prefill 128 forced tokens, 5 reps, fresh nonce):

| | W4A8 off | W4A8 on | change |
|---|--:|--:|---|
| prefill 4,088 tokens | 10,916 tok/s (TTFT 0.37 s) | 13,607 (0.30 s) | **+24.6%** |
| prefill 16,656 tokens | 9,201 | 11,805 | **+28.3%** |
| prefill 33.9k tokens | 8,003 | 10,018 | **+25.2%** |
| decode, 1 stream, 1k / 4k | 100.8 / 102.3 | 98.8 / 102.3 | -2.0% / 0.0%, noise |
| decode, 8 / 12 streams | 405.4 / 537.8 | 413.8 / 546.0 | +2.1% / +1.5%, noise |
| gsm8k 200, `--nonce`, thinking off | 0.375 | 0.375 | 0.000 |
| KV cache | 5.72 GiB | 5.69 GiB | -0.03 GiB (the `Wref` tensors, 31 MB) |

Numerics. The W4A8 layer differs from an fp32 reference by 0.040-0.043 (relative Frobenius norm) at every
token count and tile config, on real layer-20 weights with router routing. a16w4 itself is 0.0023-0.0039, and
the emulated W4A4 scheme the checkpoint declares is 0.19-0.22. Each GEMM alone on identical fp8 inputs is
1.66e-3, which is bf16 output rounding, so the kernel is exact and the 4% comes from rounding the activations
to e4m3 at two points.

Limits:
- **gsm8k does not exercise the shipped path.** Its prompts are shorter than 1,025 tokens, so at the default
  threshold none of them reach W4A8. The gsm8k row above ran a test-only mode
  (`RADIANCE_MOE_W4A8_MIN_TOKENS=1 RADIANCE_MOE_W4A8_FORCE_ALL=1`) that sends every eager prefill through W4A8,
  using the block_m 16 tile configs rather than the shipped 64 / 128 ones. The evidence for the shipped
  configs is the per-layer check across all 26 tile configs, plus a long-context code set (18 rows, prompts
  up to 32k). Its judged score moved from 0.208 to 0.242 (n = 12, noise), and the judge's context rejected the
  six 32k rows.
- No perplexity or long-generation quality run was done. Per-layer numerics and timing use one layer (20).
- Decode was not kernel-traced (rocprofv3 `--attach` crashes the engine under graph-launched decode). Its
  evidence is the layer microbenchmark plus the served runs above.

## 7. Draft-only int2 lm_head (`RADIANCE_MOE_DRAFT_HEAD`)

Single-stream decode is no longer bounded by the experts. rocprofv3 `--attach` on single-stream decode of the
served model, compiled without CUDA graphs so the kernels can be attributed (per decode step):

| | per step | share |
|---|--:|--:|
| kernel busy (31 steps) | 20.7 ms | |
| **lm_head** (verify + 3.32 draft passes) | **6.89 ms** | **33.3%** |
| bf16 dense GEMMs (including `shared_expert_gate`) | 6.87 ms | 33.2% |
| target expert GEMMs (a16w4) | 3.44 ms | 16.7% |
| MTP experts / MoE routing + reduce / GDN | | 1.8% / 2.6% / 3.3% |
| attention / sampling / norms + elementwise | | 0.9% / 0.3% / 7.9% |

An eager trace puts lm_head at 27.1%, but eager splits the inductor fusions and inflates "other", so the compiled
trace is the representative one. Each MTP draft pass computes the full-vocab logits `[batch, 248320]` against the
bf16 `lm_head` (1.017 GB, untied) and takes the argmax. The image's dynamic draft runs 1 to 4 passes per step
(3.3 on average), and the verify pass reads the head once more. At 634 GB/s one read is 1.6 ms.

The drafter only needs the top-1 token (`draft_sample_method` is greedy, so drafts are one-hot in rejection
sampling). A worse draft costs acceptance and nothing else, because the verify pass scores with the target's own
`LogitsProcessor` and its bf16 head. So the drafter's head can be approximate.

**`patch_moe_drafthead.py`** (with `moe-drafthead/radiance_moe_drafthead.py`, copied into site-packages) ends
`Qwen3_5MTP.load_weights` with `radiance_moe_drafthead.arm(self)`. The drafter's own `LogitsProcessor` gets a
compressed copy of its populated lm_head; the target's `LogitsProcessor` is not touched. This happens before vLLM
sizes the KV cache, so the copy's memory shows up in "Available KV cache memory". Modes:
- **int2** (default): the image's `radiance_drafthead`, an int2 g128 asymmetric coarse pass plus an exact bf16 rerank
  of the top 32 candidates. 0.133 GiB.
- **int4**: g128 symmetric, on vLLM's RDNA W4A16 kernels (HIP `wvSplitK_int4_g` up to M = 5, Triton above). 0.244 GiB.
- **fp8**: per-row e4m3 weights with bf16 activations, a Triton weight-only GEMV. 0.475 GiB.
- Every mode checks a 512-row sample of the compressed copy against the head, at load and again on the first real
  call, and raises on a mismatch. A test that hands it the wrong head raised in all three modes.

The head alone, `[248320, 2048]`, per call (GB/s = weight bytes / time):

| | M = 1 | M = 5 | M = 16 |
|---|--:|--:|--:|
| bf16 (`wvSplitK` / `F.linear`, the stock head) | 1605 µs (634 GB/s) | 1628 | 1667 |
| fp8 per-row, Triton weight-only | 835 | 850 | 878 |
| int4 g128 (`wvSplitK_int4_g` up to M = 5, Triton above) | 413 | 527 | 1098 |
| int2 g128 + exact rerank | 487 | 481 | 460 |

End to end, same stack as section 6 (exact scan, W4A8), on a production-style launcher set up with the defaults that
`serve-moe-mxfp4.sh` has now.
The greedy-equality set is 20 prompts x 256 tokens at temperature 0, and tok/s on those identical outputs is the
controlled single-stream A/B:

| arm | identical to off | acceptance (tokens/step) | tok/s, identical outputs (2 runs) | localeval 1k decode | 8 / 12 streams | gsm8k 200 | KV GiB |
|---|--:|--:|--:|--:|--:|--:|--:|
| off | 20/20 (off vs off) | 3.305 | 116.7 / 117.0 | 103.3 / 99.9 | 412 / 530; 411 / 554 | 0.355 / 0.385 | 5.69 |
| fp8 | 20/20 | 3.293 | 129.7 / 131.8 (+12%) | 109.7 (noise) | 416 / 575 | 0.365 | 5.19 |
| int4 | 20/20 | 3.277 | 136.5 / 137.9 (+18%) | 117.0 (+13%) | 444 / 578 | 0.370 | 5.42 |
| **int2** | 20/20 | 3.304 | **138.1 / 139.3 (+19%)** | **121.3 (+17%)** | **449 / 605** | 0.395 | 5.49 |

- The outputs were identical in every arm, so no divergence analysis was needed. A divergence could only have been a
  near-tie flip caused by different verify positions.
- Acceptance and the drafted tokens per step (3.28) did not move with int2, so its coarse logits do not hurt the depth
  gate's softmax either.
- gsm8k is within noise for every arm (`localeval compare`, 2 SE about ±0.097; the off arm itself moved 0.030).
- The 4k localeval decode depends on the content (one rep in some runs decodes at about 200 tok/s), so it is not used
  for the A/B.
- 8 and 12 streams ran with no errors and no regression.
- In production (the tatooine service, localeval 1k decode): single stream **125.9 tok/s** (it was about 100-103
  before), 8 / 12 streams 459.4 / 598.1.
- The copy costs 0.133 GiB for int2 (KV -0.20 GiB), 0.244 GiB for int4 and 0.475 GiB for fp8. int2 is the fastest arm at
  1 and 12 streams and the cheapest in memory, which is why it is the default.

**Which `radiance_drafthead`.** The int2 mode reuses the image's module and relies on its internals:
`_quantize_head_now(lp, head)`, `RERANK`, and the `_radiance_wq` / `_radiance_scale` / `_radiance_zs` buffers it leaves
on the `LogitsProcessor`, plus its quarter-major packing (`_dequant_int2` rebuilds rows from it for the sample check).
The 0.9.3 image's copy (md5 3e957338) is older than this repo's `radiance_drafthead.py` (md5 265a8593, which adds
fp8 lm_head support, `RADIANCE_DRAFT_RERANK`, a packing cache and unconditional deferral of the quantization), and
the launcher serves with the image's copy: nothing in it puts the repo's on the import path. Both copies provide
everything the patch uses, with the same packing. A kernel-level check on the real lm_head (arm each mode on a stand-in
drafter, compare the logits with bf16, give the first-call guard a wrong head) printed identical error figures with
both copies: int2 relative error 0.408 at M = 1 and 0.514-0.560 at M = 2-16, and the guard raised in all three modes.
The int2 call took 0.44-0.47 ms with the image's copy and 0.46-0.49 ms with this repo's. Only the image's copy was served,
so an image rebuilt from this repo's file is covered by that check and not by a served run.

Limits:
- The first-use check against the shared head runs on the first request, after the server is up. A mismatch there
  would kill the engine on that request, not at boot. Send one request after starting.
- `RADIANCE_FAST_DRAFT=1` is the image's own hook for the same head. Do not set it together with this knob: its
  wrapper around `load_weights` runs outside the patch and would re-arm the head last. Setting it alone, without this
  patch, was not tested here and has no first-use check.
- The patch needs TP = 1 (a vocab-sharded head raises) and a bf16 / fp16 lm_head.
- Acceptance was measured on the 20-prompt equality set (forced 256-token answers). Acceptance on real long answers
  was not re-measured.

`RADIANCE_MOE_DRAFT_HEAD=off` runs the stock bf16 draft head and imports nothing. The knob is also dropped when
`SPEC=0`, since there is no draft pass then.

## 8. FP8 dense layers (`RADIANCE_MOE_DENSE_FP8`)

The bf16 dense layers are the next block after the lm_head: Quark leaves the GDN and attention projections and the
shared expert in bf16, and they take 33.2% of the step in the section 7 trace. `patch_moe_densefp8.py` (with
`moe-densefp8/radiance_moe_densefp8.py` and the HIP kernel `moe-densefp8/radiance_fp8w.hip`) hooks the end of vLLM's
`process_weights_after_loading`, before KV sizing:
- It skips the MTP drafter (by class name).
- The target's `in_proj_qkvz`, `out_proj`, `qkv_proj`, `o_proj` and the shared expert's `gate_up_proj` / `down_proj`
  (160 linears, 2.62 GiB bf16) are quantized to e4m3 with one fp32 scale per output row into one contiguous arena
  (1.31 GiB). The bf16 parameters are dropped.
- Each layer's quant method becomes an fp8 apply that calls one opaque custom op,
  `torch.ops.vllm.radiance_fp8w_linear`, which picks the kernel by (N, K, M) from a measured table:
  - M <= 16: the better of the HIP skinny W8A16 kernel and Triton W8A16;
  - 17 to 80: Triton W8A16 tiles;
  - above 80: `torch._scaled_mm` rowwise W8A8 with vLLM's per-token fp8 activation quant. This is the only place
    activations are quantized, and it only runs in prefill chunks. At M = 4096 it is 1.3-1.4x faster than the bf16
    hipBLASLt GEMM; every weight-only path is 15-50% slower there.
- A load-time self-check runs every shape across every dispatch band (19 M values) against `x @ dequant(W)^T`. It also
  compiles every Triton variant before graph capture. Worst relative error: HIP / Triton 0.0018, W8A8 0.027.

Kept in bf16, on purpose:
- `in_proj_ba`: its outputs feed the GDN beta (a sigmoid) and decay (`exp(-exp(A_log) * softplus(a + dt_bias))`), so an
  error compounds through the recurrence across every later token. 4 MB in all and latency-bound.
- `mlp.gate`, the router: a top-8-of-256 choice flips on small logit changes, and vLLM builds it with
  `quant_config=None`.
- `shared_expert_gate`: a scalar gate (section 9).
- `lm_head` (the verify head stays exact) and the whole MTP drafter.

`moe-densefp8/radiance_fp8w.hip` is the HIP skinny kernel, built at container start with the image's hipcc by
`moe-densefp8/build.sh` (a few seconds). A block is WV output columns x SK K-splits waves; each wave owns one output
column over one contiguous K slice and keeps M accumulators, and the SK partials are reduced in LDS in a fixed order, so
it is deterministic and needs no second kernel. A 128-bit load is 16 fp8 weights; they are expanded with the gfx12
hardware converter `v_cvt_f32_fp8` (exact), packed to bf16 pairs with `v_perm` (exact, fp8 has at most 4 significant
bits) and multiplied with `v_dot2_f32_bf16`; the row scale is applied once in the epilogue. M <= 16.
On gfx1201 the packed `__builtin_amdgcn_cvt_pk_f32_fp8` returned the selected byte in both halves (a one-hot test
shows it); the scalar `__builtin_amdgcn_cvt_f32_fp8` with a constant byte select is exact, so the kernel uses that.

Per call at M = 5 (single-stream verify), inside a captured CUDA graph with the weights rotated over at least 256 MB
so the 64 MB last-level cache cannot hold them, µs / GB/s:

| shape (calls per step) | bf16 (stock) | HIP W8A16 | Triton W8A16 | `_scaled_mm` W8A8 + act quant |
|---|--:|--:|--:|--:|
| `in_proj_qkvz` 12288x2048 (x30) | 83.0 / 606 | **46.6 / 541** | 57.7 / 437 | 72.8 / 346 |
| `out_proj` 2048x4096 (x30) | 30.8 / 544 | **18.8 / 446** | 28.9 / 290 | 44.9 / 187 |
| `qkv_proj` 9216x2048 (x10) | 62.7 / 602 | **36.5 / 518** | 45.0 / 420 | 58.5 / 323 |
| `o_proj` 2048x4096 (x10) | 30.9 / 543 | **18.9 / 444** | 28.1 / 298 | 44.9 / 187 |
| shared expert `gate_up` 1024x2048 (x40) | 9.9 / 423 | **8.3 / 254** | 15.7 / 134 | 17.5 / 120 |
| shared expert `down` 2048x512 (x40) | 6.6 / 318 | 6.4 / 166 | **5.6 / 190** | 24.5 / 43 |

vLLM rounds CUDA-graph sizes up to multiples of 1 + `SPEC` (5, 10, 20, ... at `SPEC=4`), so one stream always verifies
at M = 5, 8 streams at M = 40 and 12 streams at M = 60.

End to end, same session, same launcher, knobs only (the int2 head is on in every arm). The fixed-prompt set is the
20 x 256 greedy set of section 7; acceptance comes from the server's Prometheus spec-decode counters:

| arm | KV GiB | fixed prompts tok/s (acceptance) | localeval 1k decode | 8 / 12 streams (warm) | prefill TTFT 4k / 16k / 32k | gsm8k, 2 runs | mmlu |
|---|--:|--:|--:|--:|--:|--:|--:|
| off | 5.49 | 135.7 (3.304) | 115.2 / 117.5 | 463 / 607 (2 runs) | 0.30 / 1.39 / 3.32 s | 0.375 / 0.390 | 0.839 |
| gate only (section 9) | 5.79 | 140.9 (3.303), **+3.8%** | 128.5 | 477 / 614 | 0.29 / 1.39 / 3.33 s | 0.365 (1 run) | not run |
| fp8 only | 6.55 | 147.1 (3.272), **+8.4%** | 131.3 / 118.4 | 477 / 604 | 0.28 / 1.34 / 3.22 s | 0.375 / 0.395 | 0.839 |
| **both** | **6.55** | **154.4 (3.257), +13.8%** | 139.2 / 140.9 | **491 / 622** | **0.28 / 1.33 / 3.19 s** | 0.395 / 0.365 | 0.839 |

- "off" is the mean of three fixed-prompt runs (134.0 / 135.8 / 137.3), "both" the mean of two (153.6 / 155.2). A third
  "both" run, 145.0 tok/s, was the first traffic after a fresh compile: the stock MoE Triton kernels were still
  JIT-compiling on their first shapes. Cold, the gain is about +8%; warm, +14%.
- The step shrinks from 24.3 ms (off) to 22.2 (fp8) and 21.1 ms (both). Acceptance falls about 1% with fp8 (3.304 to
  3.272) and 1.4% with both, so tok/s rises less than the step time falls (+8.4% / +13.8% against -9% / -13%).
- localeval's 1k decode moves with the content, because the outputs differ per arm and so does acceptance.
- Prefill is +3-6% at 4k-32k (the W8A8 GEMMs at M > 80). localeval calls that noise, but the sign is the same at all
  three sizes. No regression.
- Quality: every `localeval compare` verdict against off is "noise" (gsm8k 2 SE about ±0.097, mmlu ±0.062). The gate-only
  arm has one gsm8k run and no mmlu.
- 8 / 12 streams are warm runs, one per fp8 arm (two for off). A cold cache gave 433 / 618 (fp8) and 450 / 573 (both).
  Single runs vary by about ±5%, so the 8-stream gain (+6%, one warm pair) is weak evidence and 12 streams is noise.
- In production (the tatooine service, 1k decode): single stream 141-155 tok/s, 8 streams 484.6 (warm), 12 streams
  628-644, prefill 14,702 / 11,673 / 10,623 tok/s at 4.1k / 16.7k / 33.9k, gsm8k 0.395.

Numerics and outputs:
- The fp8 per-row weight error on real layer-0 / layer-3 weights is 2.6% (relative Frobenius). Layer output error is
  2.4-2.8% against fp32 (bf16 itself is 0.17%); W8A8 gives 3.3-3.9%.
- Greedy outputs diverge from off on 18 of 20 fixed prompts, with fp8 and with both. Every first divergence is a near-tie:
  the top-2 logprob margin is at most 0.5 for fp8, and the arm's token is off's second choice in 18 of 18 (both: 17
  of 18, largest margin 0.63).

KV memory. Of the 1.31 GiB freed, 1.06 GiB reaches the KV pool; about 0.35 GiB stays reserved by torch's allocator in
20 MB segments that still hold live tensors.

Limits:
- Needs its own compile cache (`-dfp8`). vLLM's AOT compile-cache key does not see the swapped linear apply: a copied bf16
  cache replays the bf16 graph on the uint8 weights and the profile run dies with `expected mat1 and mat2 to have the
  same dtype`. A cold cache costs about a minute of compile, plus JIT of the stock MoE Triton kernels on the first new
  prefill shapes: TTFT is 1-1.6 s higher on the first few requests after the first start, then normal.
- A cold-cache start sizes the KV pool smaller (5.62 GiB against 6.55 GiB on the warm restart, in production), as in
  section 10. Restart once.
- W8A8 is only used for prefill chunks (M > 80), so prefill and decode numerics differ slightly. The W4A8 experts make
  the same split.
- M = 81 to 511 goes to `_scaled_mm` but was not measured. At M = 80 it is slower than bf16 on `out_proj` / `o_proj`
  (52 against 47 µs); at M = 512 it is faster on every big shape. Small prefill chunks and mixed batches in that range may
  lose a few µs per call.
- M = 65 to 80 (13 or more streams) uses the 128-row Triton tiles, about 10% slower than bf16 on the 2048x4096 shapes
  (at the kernel level, not end to end). M = 60, the 12-stream size, uses the 64-row tile and wins.
- Tested with the int2 draft head on in every arm, and only on Qwen3.5-35B-A3B-MXFP4 (the layer list and the
  M table are measured on its shapes).

`RADIANCE_MOE_DENSE_FP8=0` keeps the bf16 layers and builds no HIP module.

## 9. Fused shared-expert gate (`RADIANCE_MOE_GATE_FIX`)

Every MoE layer computes `sigmoid(shared_expert_gate(x)) * out` as three kernels. The 1x2048 gate falls to a hipBLASLt
GEMV that alone takes about 20 µs per call, 40 times per step. `patch_moe_densefp8.py` (the same patch as section 8)
makes `Qwen2MoeMLP.forward` (target and drafter) call one Triton kernel, `torch.ops.vllm.radiance_gate_mul`, which takes
about 3 µs per call and emulates the bf16 roundings of the three ops it replaces. A stride / dtype guard falls back to the
original path. Only the dot product's summation order differs from hipBLASLt: it is bit-equal on the bench checks.

- Fixed-prompt decode: +3.8% (140.9 against 135.7 tok/s), the step 24.3 to 23.4 ms (predicted 0.72 ms, measured
  0.9 ms). In the serve 15 of 20 fixed prompts are identical over 256 tokens; the other 5 flip at exact near-ties (top-2
  margin at most 0.125, and the arm's token is off's second choice). Acceptance is unchanged (3.303 against 3.304).
- KV +0.3 GiB on its own (5.49 to 5.79) from less non-torch memory (24.24 to 23.94 GiB consumed, weights equal). The
  cause is not isolated; it does not add on top of section 8 (both = 6.55 GiB).
- The gate-only arm has one gsm8k run (0.365) and no mmlu.

`RADIANCE_MOE_GATE_FIX=0` keeps the three stock ops. It is applied through a flag in `qwen2_moe.py` that the patch
installs either way, so it gets its own `-gate` compile-cache suffix.

## 10. This launcher, end to end

Two runs of `serve-moe-mxfp4.sh` on `amd/Qwen3.5-35B-A3B-MXFP4`, docker, a copy of this tree (no `.git`), libr4d
v0.5.0 already checked out, localeval as in the sections above. The first ran with the defaults of sections 4-6,
before sections 7-9 existed; the second has all nine on and is the launcher as it stands.

### All nine on

`SNAP` pointed at the Hugging Face cache snapshot, `RUNTIME=docker`, `DETACH=1`, the default knobs, and a compile cache that
did not exist yet; the server was stopped and started again for the warm start.

- **Boot.** Every patch prints OK, including the three new ones (`moe draft head: arm in Qwen3_5MTP.load_weights`, `moe dense
  fp8: convert hook`, `moe gate fix`, two patches) and both builds (`radiance_fp8w`, `radiance_moe_w4a8_hip`). The log shows
  `exact chunk scan ON` from the API server and the engine core, `W4A8 expert GEMMs ON`, `MoE dense fp8 ON: 160 linears` (2.62
  to 1.31 GiB), `MoE gate fix ON` for the target (40 gates) and the MTP drafter (1), each with a self-check difference of 0, and
  `MoE draft head int2 ON`. After the first request: `int2 draft head verified against the shared lm_head at first use (sample
  rel. error 0.5254, tol 0.8)`. No traceback in either start.
- **Start time and KV cache.** The cold start took 242 s and sized the KV pool at **5.62 GiB** (331,692 tokens). The warm
  restart took 115 s and got **6.55 GiB** (385,191 tokens, 5.88x at 65k), the pool of the production launcher. The previous run
  had 5.69 GiB, so fp8 layers and gate (+1.06 GiB) net of the draft head (-0.20 GiB) are +0.86 GiB. The smaller pool on the first
  start was seen in both runs (sections 4-6 defaults: 4.76 against 5.69 GiB); restart once.
- **Decode** (1k prompts, 512 forced tokens, 3 reps, warm; the sweep ran twice and this is the second):
  1 stream **140.6 tok/s** (138.6-141.4), 8 streams **490.4** aggregate, 12 streams **632.1**. The first run, right after the
  start, gave 139.1 / 353.9 / 569.1: the 8-stream cell had a 2.9 s TTFT, probably the first new prompt shapes JIT-compiling the
  stock MoE kernels, as seen on the production launcher. Before sections 7-9 this launcher measured 100.2 tok/s single-stream and 394 / 518 at 8 / 12 streams (three runs);
  the production launcher with all nine measured 141-155, 484.6 and 628-644.
- **Prefill** (speed sweep, 128 forced tokens, 5 reps, median):

  | Prompt | TTFT | Prefill tok/s | Before sections 7-9 | change |
  |--:|--:|--:|--:|--:|
  | 4,089 | 0.28 s | 14,873 | 13,605 | +9% |
  | 16,633 | 1.32 s | 12,627 | 11,808 | +7% |
  | 33,858 | 3.20 s | 10,589 | 10,011 | +6% |

  The 16k cell had one slow rep (7,703 tok/s); the median does not include it. The previous column is the earlier run
  below (a different session).
- **gsm8k** (200, `--nonce`, thinking off): 0.395. The standard error at n = 200 is about 0.034, and the earlier runs of this
  launcher scored 0.345 and 0.385.
- **Knob-off paths.** `RADIANCE_MOE_DRAFT_HEAD=off`, `RADIANCE_MOE_DENSE_FP8=0`, `RADIANCE_MOE_GATE_FIX=0`, `SPEC=0` and
  `MOE_FIXES=0` each print the expected command, and bad values (`RADIANCE_MOE_DRAFT_HEAD=int8`, `RADIANCE_MOE_DENSE_FP8=yes`,
  `RADIANCE_MOE_GATE_FIX=2`) stop with an error. With all three new knobs off the printed command, the patch chain, the
  environment and the cache path are byte-identical to the previous revision's. The container's patch and build chain also ran
  in the image without a GPU, twice per case (the second pass reports NOOP for every patch; the builds and the module copies
  rerun):
  - `RADIANCE_MOE_DRAFT_HEAD=off` patches nothing in `qwen3_5_mtp.py` and installs no `radiance_moe_drafthead`;
  - `SPEC=0` does the same (the head drops with the speculation) and passes no `RADIANCE_MOE_DRAFT_HEAD`;
  - `RADIANCE_MOE_DENSE_FP8=0` builds no `radiance_fp8w`, drops `-dfp8` from the cache path, and still applies the gate fix;
  - `RADIANCE_MOE_GATE_FIX=0` still applies both patches (the gate is a flag in `qwen2_moe.py`) with the flag off, and the cache
    path has no `-gate`;
  - both off: `patch_moe_densefp8.py` is not run, `radiance_moe_densefp8` is not installed;
  - `MOE_FIXES=0` drops dense fp8, the gate fix and W4A8, and keeps the draft head and the GDN fix;
  - `MOE_FIXES=0 RADIANCE_GDN_SCAN_FIX=0 RADIANCE_MOE_DRAFT_HEAD=off` leaves an empty chain.

### Sections 4-6 defaults only (earlier run)

- **Boot.** Every patch prints OK (`aiter a16w4`, `quark moe`, `unified_attention`, `radiance_r4d_attn`,
  `gdn scan fix bind`, `moe w4a8: knob + import` and `dispatch`). The log shows `exact chunk scan ON` from the
  API server and the engine core, and `W4A8 expert GEMMs ON (fp8 WMMA, calls >= 1025 tokens ...)`.
- **KV cache.** 5.69 GiB, 334,367 tokens (5.10x at 65k) on the second start, which reused the compile cache.
  **The one start on an empty compile cache sized 4.76 GiB (279,531 tokens)**: the profiling pass saw 1.77 GiB
  of peak activation instead of 1.17 GiB, and 24.38 GiB of weights plus non-torch memory instead of 24.05.
  Same flags, so if the first start comes up with the smaller pool, restart it. This was seen once and the
  cause was not isolated.
- **Prefill** (speed sweep, 128 forced tokens, 5 reps, median):

  | Prompt | TTFT | Prefill tok/s | Section 3 (r4d, no W4A8, stock scan) | change |
  |--:|--:|--:|--:|--|
  | 4,088 | 0.30 s | 13,605 | 10,760 | +26% |
  | 16,655 | 1.41 s | 11,808 | 9,128 | +29% |
  | 33,879 | 3.38 s | 10,011 | 7,833 | +28% |

  The section 3 column is a different session on this launcher before these changes. The controlled A/B is
  section 6. One rep each at 4k (3,656 tok/s) and 16k (7,108) was slow; the medians do not include them.
- **Decode, 1 stream** (9 reps x 256 tokens): 100.2 tok/s at 1k, 98.9 at 4k.
- **gsm8k** (200, `--nonce`, thinking off): 0.345 and 0.385 in two runs, against 0.365-0.395 on the validated
  stack. The standard error at n = 200 is about 0.034.
- **Concurrency** (1k prompts, 512 tokens, 3 runs): 8 streams 368 / 383 / 430 tok/s aggregate, 12 streams
  535 / 525 / 495. No errors, no traceback in the server log, at most 12 requests running and 0 waiting.
- **Knob-off paths.** `SPEC=8`, `RADIANCE_GDN_SCAN_FIX=0`, `RADIANCE_MOE_W4A8=0`, `MOE_PREFILL_ATTN=off` and
  `MOE_FIXES=0` each print the expected command. `SPEC=8` only changes the speculative config. For the other
  four the container's patch and build chain also ran in the image without a GPU (twice per case: the second
  pass reports NOOP for every patch):
  - `MOE_PREFILL_ATTN=off` still mounts libr4d and builds the GDN module, and builds no `r4d_moe`;
  - `RADIANCE_GDN_SCAN_FIX=0` builds and patches no GDN module, and leaves `radiance_gdn.py` untouched;
  - `RADIANCE_MOE_W4A8=0` builds and patches no W4A8 module;
  - `MOE_FIXES=0` keeps the GDN fix and drops W4A8 (it rides on the a16w4 lane).
- The production launcher that sections 4-6 were measured on runs decode attention on AITER's unified kernel
  (+13% / +21% decode at 16k / 32k). That patch is not in this repo, so decode at long context is not expected
  to match it here.
  Prefill and KV size matched production. The 8- and 12-stream aggregates averaged 394 and 518 tok/s over
  three runs, 4% and 3% under the production launcher's single-run 410.5 and 535.8, inside the spread of the
  three runs.

## Running

```bash
# MoE (one card): fixes on. Defaults: MTP-4, 16 sequences, 0.97, exact GDN scan, W4A8 prefill, int2 draft head,
# fp8 dense layers, fused expert gate
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 ./serve-moe-mxfp4.sh
# the previous defaults (MTP-8, 8 sequences, 0.95), or switch the newer pieces off one at a time
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 SPEC=8 MAXSEQS=8 GPU_UTIL=0.95 ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_GDN_SCAN_FIX=0 ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_MOE_W4A8=0 ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_MOE_DRAFT_HEAD=off ./serve-moe-mxfp4.sh   # or fp8 / int4
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_MOE_DENSE_FP8=0 ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_MOE_GATE_FIX=0 ./serve-moe-mxfp4.sh
# W4A8 only from 2,049 tokens up
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RADIANCE_MOE_W4A8_MIN_TOKENS=2049 ./serve-moe-mxfp4.sh
# stock vLLM for comparison
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 MOE_FIXES=0 ./serve-moe-mxfp4.sh
# TRITON_ATTN prefill instead of R4D, or an existing libr4d v0.5.0 checkout. The libr4d clone is still made
# for the GDN scan fix; with MOE_PREFILL_ATTN=off RADIANCE_GDN_SCAN_FIX=0 nothing needs it.
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 MOE_PREFILL_ATTN=off ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 R4D_SRC=~/src/libr4d ./serve-moe-mxfp4.sh

# kernel test for the split-KV patch (inside the image, after the patch)
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups -v "$PWD":/w \
  --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 \
  -c 'cd /w && python3 patch_attn_3d_multiq.py && python3 moe-tests/test_attn_3d.py /tmp/attn3d.json'
```

## Not tested

- The MoE lane at TP > 1.
- R4D prefill under concurrent mixed prefill/decode batches (only single-stream runs and the 1k-prompt
  concurrency runs were measured), and on the `triton` mode end to end.
- podman for `serve-moe-mxfp4.sh`. Its flags mirror `serve-mxfp4.sh`, but it has only been run with docker.
- Sections 1-3 were measured at MTP-8 / 8 sequences / 0.95 and were not repeated at the new defaults, apart
  from the end-to-end check in section 10.
- The new capacity defaults: `GPU_UTIL=0.97` leaves about 1 GiB of the card free, and it was not tried with
  another process on the same GPU (a desktop session, for instance). `MAXSEQS=14` is untested. The
  depth-4 numbers use forced-length outputs; acceptance on real long answers was not re-measured.
- W4A8 (`RADIANCE_MOE_W4A8`): no perplexity or long-generation quality run. The gsm8k check ran a test-only
  mode, because its prompts never reach the 1,025-token threshold. The judged long-context set (n = 12) is
  too small to show a difference, and its judge could not take the 32k rows. Per-layer numbers come from one
  layer. A checkpoint other than Qwen3.5-35B-A3B-MXFP4 may have block-exponent gaps above the 14 the kernel
  represents exactly.
- Exact GDN scan (`RADIANCE_GDN_SCAN_FIX`): the served per-head error check used prompts up to about 4.1k
  tokens, so a carried-over initial state from an earlier prefill chunk is covered by the synthetic kernel
  tests (random initial state, multi-sequence batches) and by the 16.7k / 33.9k speed runs, not by a served
  per-head measurement. The dense launcher `serve-mxfp4.sh` does not build the module. Its 48/16 head layout
  is in the kernel tests and the patch applies to its `radiance_gdn.py`, but it was never served with it.
- Draft head (`RADIANCE_MOE_DRAFT_HEAD`): acceptance was measured on a 20-prompt set of forced 256-token answers, not on
  real long answers. The first-use guard runs on the first request, so a server that booted can still fail on it;
  `RADIANCE_FAST_DRAFT=1` alone (without this patch) was not tested, and neither were both knobs together. The fp8 and
  int4 modes ran only in the production-style bench, not through `serve-moe-mxfp4.sh` (the launcher's dry runs print the
  right command for them). TP > 1 is rejected by the patch.
- FP8 dense layers and fused gate: prefill chunks of 81 to 511 tokens (they go to `_scaled_mm`) were not measured; at
  13 or more streams two layer shapes run about 10% slower at the kernel level (not measured end to end);
  mixed prefill/decode batches were not measured; only Qwen3.5-35B-A3B-MXFP4 was served, with the int2 draft head on in
  every arm, and no run compares all three of sections 7-9 on against all three off in one session (the chain is
  117 / 139 / 154 tok/s across two sessions on the same fixed prompts). Greedy outputs differ from the bf16 layers at
  near-ties on 18 of 20 prompts, and there is no perplexity or long-generation quality run beyond gsm8k (200, twice per
  arm) and mmlu (localeval `--limit 5`, one run per arm; the gate-only arm has one gsm8k run and no mmlu). The first
  prompt shapes after a cold start JIT the stock MoE kernels for 1-2 s.
- Sections 7-9 were measured on a production-style launcher with AITER decode attention. Section 10 has this launcher's
  own numbers, which are lower at long context for that reason.
- Qwen3.6-35B-A3B compressed-tensors W4A16 checkpoints (e.g. `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`).
  On the 0.9.3 image vLLM picks a MoE backend that fails with `'_C' has no gptq_marlin_repack`, and
  its dense MXFP4 layers fall back to emulation. That needs a separate patch.
- The a16w4 tile table was swept only on the Qwen3.5-35B-A3B expert shapes (gate_up N=1024 K=2048,
  down N=2048 K=512). The "K ≥ 1024 means gate_up" split is a heuristic.
