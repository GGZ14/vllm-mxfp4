# MXFP4 MoE on gfx1201, split-KV verify attention, host-RAM KV tier, MoE prefill attention

Four independent additions, each opt-in or scoped so the default Qwen3.8-27B serve is unchanged.
All numbers are from one Radeon AI PRO R9700 (gfx1201, 32 GB, TP=1) on the published image
`stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1). Each is a single run unless noted.

| Addition | Files | Default |
|---|---|---|
| MXFP4 MoE experts on gfx1201 | `patch_quark_moe_w4a16.py`, `patch_gfx12_aiter_a16w4.py`, `serve-moe-mxfp4.sh` | on in `serve-moe-mxfp4.sh` (`MOE_FIXES=1`) |
| Split-KV attention for spec-decode verify | `patch_attn_3d_multiq.py`, `moe-tests/test_attn_3d.py` | applied by `serve-moe-mxfp4.sh` only |
| Host-RAM KV tier | `patch_offload_mamba_eagle.py`, `RAM_TIER_BYTES` in both launchers | off (`RAM_TIER_BYTES=0`) |
| R4D prefill attention on the MoE lane | `patch_moe_prefill_attn.py`, `moe-r4d/` | on in `serve-moe-mxfp4.sh` (`MOE_PREFILL_ATTN=r4d`) |

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

## 3. Host-RAM KV tier (`RAM_TIER_BYTES`)

vLLM's `OffloadingConnector` (CPU tier) is in the image, but on a GDN hybrid under speculative decoding
it is **write-only**. In one run it stored 12.95 GB, loaded 0 bytes and recorded 0 external prefix hits.
Nothing errors.

When speculation is on and no KV group is tagged as a draft group (the DFlash2 backport tags none),
the offload scheduler falls back to marking every group volatile:

```python
if use_eagle and not eagle_groups:
    eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
```

A Mamba/GDN group holds a single state chunk, so dropping its "volatile trailing chunk" leaves no
GDN state to restore. The startup log shows it: `EAGLE/MTP draft attention groups [0, ..., 8] detected`.

**`patch_offload_mamba_eagle.py`** makes the fallback mark only groups whose layers sit past the
target's `num_hidden_layers` (the drafter's), and falls back to every non-Mamba group. It is inert
unless a connector is configured. Upstream tracks this as vllm-project/vllm#57032. The fix in flight
there, #54165, stops treating DFlash as an EAGLE-style drafter. Its hunk gave identical results on
this box.

Qwen3.8-27B MXFP4 + DFlash2, TP=1, 24 GiB tier. Workload: 8 conversations over distinct ~24k-token
prompts, served round-robin, so the working set overflows the GPU prefix cache:

| | GPU cache only | Tier, unpatched | Tier, patched |
|---|--:|--:|--:|
| Repeat-turn TTFT p50 | 7.24 s | 7.25 s | **0.52 s** |
| Tokens restored per hit | 0 | 0 | 22,880 |

- Answers were byte-identical to the GPU-only run: 32/32, plus 24/24 on a long-answer variant.
- The drafter's acceptance held on restored turns: 0.70 of drafted tokens accepted (1,482 / 2,121) in
  restored-only intervals, vs 0.69 on cold turns in the same run (841 / 1,218) and 0.69 GPU-only
  (2,430 / 3,500). Read from vLLM's 10 s `SpecDecoding metrics` log intervals on the long-answer
  variant, not per request.
- A 400-token decode probe on a fresh prompt after each run held at 87.5 tok/s with and without the tier.
- Restores ran at 16.9 GB/s, about 50 ms per hit. The rest of the TTFT is recomputing the tail after
  the last align-mode GDN checkpoint.
- GPU-only prefix caching held only 2 such conversations; at 4 it got zero hits.

The tier is one mmap'd file in `/dev/shm`, pinned for DMA. Both launchers run with `--ipc=host`, so
it comes out of the host's `/dev/shm`. They refuse to start if it doesn't fit there. It stored about
67 KB per token, so 24 GiB holds roughly 380k tokens.

## 4. Prefill attention on the MoE lane (`MOE_PREFILL_ATTN`)

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

## 5. MTP draft depth and sequence capacity (`SPEC`, `MAXSEQS`, `GPU_UTIL`)

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
  repair post-pass), not on `serve-moe-mxfp4.sh`. Section 8 has this launcher's own numbers.

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

## 6. Exact GDN chunk scan (`RADIANCE_GDN_SCAN_FIX`)

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

## 7. W4A8 expert GEMMs for prefill (`RADIANCE_MOE_W4A8`)

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

## Running

```bash
# MoE (one card): fixes on, tier off
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 ./serve-moe-mxfp4.sh
# the same with a 16 GiB host-RAM tier, or stock vLLM for comparison
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 RAM_TIER_BYTES=17179869184 ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 MOE_FIXES=0 ./serve-moe-mxfp4.sh
# TRITON_ATTN prefill instead of R4D (no libr4d clone), or an existing libr4d v0.5.0 checkout
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 MOE_PREFILL_ATTN=off ./serve-moe-mxfp4.sh
SNAP=~/models/Qwen3.5-35B-A3B-MXFP4 R4D_SRC=~/src/libr4d ./serve-moe-mxfp4.sh

# the 27B with a 24 GiB host-RAM tier
RAM_TIER_BYTES=25769803776 ./serve-mxfp4.sh

# kernel test for the split-KV patch (inside the image, after the patch)
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups -v "$PWD":/w \
  --entrypoint bash stilldeadcode/vllm-radiance:0.9.3 \
  -c 'cd /w && python3 patch_attn_3d_multiq.py && python3 moe-tests/test_attn_3d.py /tmp/attn3d.json'
```

## Not tested

- The MoE lane at TP > 1.
- R4D prefill under concurrent mixed prefill/decode batches (only single-stream runs were measured),
  and on the `triton` mode end to end.
- The RAM tier together with the MoE lane, and the RAM tier at TP = 2.
- Qwen3.6-35B-A3B compressed-tensors W4A16 checkpoints (e.g. `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`).
  On the 0.9.3 image vLLM picks a MoE backend that fails with `'_C' has no gptq_marlin_repack`, and
  its dense MXFP4 layers fall back to emulation. That needs a separate patch.
- The a16w4 tile table was swept only on the Qwen3.5-35B-A3B expert shapes (gate_up N=1024 K=2048,
  down N=2048 K=512). The "K ≥ 1024 means gate_up" split is a heuristic.
