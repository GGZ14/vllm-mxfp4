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
