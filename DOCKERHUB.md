# vllm-radiance

vLLM inference server for the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4). Bundles a working ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on this card, so you don't have to build the stack yourself.

## Tested so far

These setups have been run and measured:

| | |
|---|---|
| Models | **Qwen3.8-27B-FP8** and **Qwen3.6-27B-FP8** (gated-delta-net hybrids; architecturally identical, so they take the same tuned paths), **Qwen3.6-35B-A3B-FP8** (fine-grained MoE: 256 experts, top-8), **Gemma-4-31B-it-FP8** (dense, sliding + global attention, vision) |
| KV cache | FP8 (bf16 / `auto` also supported) |
| GPUs | 2x R9700, tensor parallel (TP=2) |

Untested (may or may not work): any other model, non-FP8 weights, single GPU, more than two GPUs, non-R9700 hardware. Treat the defaults below as a starting point for these three setups, not a general recommendation.

**Qwen3.8-27B-FP8 (and Qwen3.6-27B-FP8).** The model this image is tuned around, and what every default below assumes. It is a **gated-delta-net hybrid**: 64 layers, of which **48 are linear attention** (GDN) and **16 are full attention**, hidden size 5120, attention `head_dim` 256 with 24 query heads over 4 KV heads (**6 queries per KV head**), GDN key/value head dim 128 with a width-4 causal convolution, 248320-token vocabulary and a 262144-token position limit. Weights are fp8-e4m3 with 128x128 blocks. The two versions are byte-for-byte the same shape, so a kernel compiled for one serves the other.

That geometry is exactly what the hand-written kernels are compiled for, so on these models the whole R4D library engages: paged attention at head 256 / GQA 6 (`--attention-backend R4D`), the entire gated-delta-net layer, and the TP=2 all-reduce. The startup log prints which kernel each part of the model resolved to.

**The MTP head is inside the checkpoint** (`mtp.fc` plus one decoder layer), so speculative decoding needs no separate drafter model -- just `--speculative-config '{"method":"mtp",...}'`. It is also a vision-language checkpoint with a head-72 vision tower; pass `--language-model-only` to skip loading the vision half when serving text, which is what the compose does.

**Fine-grained MoE (Qwen3.6-35B-A3B-FP8).** Supported and tuned. Two MoE paths are baked in and activate automatically for it: RDNA4-tuned fused-MoE Triton configs (removes the stock config's `M>=96` cliff, lower prefill TTFT, lossless) and a custom bf16 MoE-gate GEMM for the skinny `n` in `[6,16]` band that rocBLAS serves poorly. One serving requirement: with `--mamba-cache-mode=align` this model's attention block size is 2240, and align asserts `block_size <= max_num_batched_tokens`, so pass **`--max-num-batched-tokens >= 2240`** (2560 is a clean default; the compose ships 2048, which is fine for the 27B but must be raised for the 35B).

**Gemma-4-31B-it-FP8.** Supported (0.4.0), e.g. `RedHatAI/gemma-4-31B-it-FP8-block`. Quantization is auto-detected from `config.json` (compressed-tensors, 128x128 blocks) and the RDNA4-tuned block-FP8 GEMM configs for its shapes load automatically (measured -5% TTFT at 8K prompt, decode unchanged). Text and vision both work. It is not a GDN hybrid, so drop `--mamba-cache-mode`; it also uses its own chat template and parsers rather than the Qwen ones. It carries a lot of KV (60 layers: 50 sliding-window + 10 global head-512), so at a given `--gpu-memory-utilization` it wants a smaller `--max-model-len` than the Qwen models. Long-context prefill is tuned for its head-512 global-attention layers (a head-size-keyed 2D attention config, measured up to -38% TTFT at 64K, -46% at 120K vs the untuned kernel; inert on other head sizes).

**Gemma-4-31B MTP speculative decoding.** For a large decode speedup, pair the target with Google's official drafter `google/gemma-4-31B-it-assistant` (vLLM loads its `gemma4_assistant` checkpoint as an MTP model). Lossless (the target verifies every drafted token), and the `RADIANCE_DYNAMIC_DRAFT` controller applies. The drafter has a head-512 layer, so its speculative `attention_backend` must be `ROCM_AITER_UNIFIED_ATTN` (`flash_attn` caps at head 256). Example: `--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' --no-async-scheduling --trust-remote-code`.

**bf16 / `auto` KV cache is supported** (0.1.5). Earlier builds crashed at startup with a Triton shared-memory (`OutOfResources`) error on any 2-byte KV cache (including the `--kv-cache-dtype auto` default), because the attention decode kernel's tile didn't fit the R9700's 64 KiB LDS at head_size 256. As of 0.4.0 that fit is enforced generally, for any head size and KV dtype, so fp8, bf16, and `auto` all work, including models with a head_size of 512, where AITER's own pick overflows LDS the same way. fp8 packs more KV per GB of VRAM; bf16 / `auto` keeps full KV precision.

## Why this exists

vLLM's ROCm builds target datacenter cards (MI300 / CDNA). RDNA4 workstation cards like the R9700 (gfx1201) don't work out of the box: AITER isn't enabled for the arch, GPU enumeration fails, several kernels need patching, and the vendor attention and GEMM paths aren't tuned for RDNA4. This image pins a working combination, builds AITER from source for gfx1201, applies the fixes, and adds tuned kernels.

## Stack

Everything below is compiled from source for `gfx1201` in the image build (0.5.0 onward); nothing is
pulled from a prebuilt wheel index.

| Component | Version |
|---|---|
| vLLM | 0.27.1 |
| PyTorch | 2.11.0 |
| Triton | 3.6.0 |
| torchvision | 0.24.1 |
| AITER | 0.1.17 |
| transformers | 5.14.1 (pinned) |
| ROCm userspace | 7.14, bundled |
| Base | `rocm/dev-ubuntu-24.04:7.14.0-full` (Ubuntu 24.04, Python 3.12) |

The PyTorch / Triton / torchvision versions are the ones upstream builds vLLM against **on ROCm**, not a
newer combination chosen for this image. Read the ROCm numbers, not `pyproject.toml`: 0.27.1's build-system
asks for `torch == 2.13.0`, which is the CUDA build, while upstream's own ROCm image builds torch
release/2.11 with torchvision 0.24.1 and pins no torch in `requirements/rocm.txt`. That distinction is
deliberate: see the tensor-parallel hang note below.

transformers is pinned because vLLM does not pin it -- `requirements/common.txt` asks only for
`transformers >= 5.5.3`, so an unpinned rebuild picks up whatever is newest and the stack moves underneath
the build. transformers 5.15.0 made Gemma-4's `head_dim` a per-layer attribute and turned the global read
into an exception that no released vLLM handles, so a Gemma-4 checkpoint fails during argument parsing,
before a model or an attention backend exists. 5.14.1 is the last release before that change and loads
every architecture this image serves. If you build your own image, keep the pin.

There is no flash-attention package: the vendor flash kernels have no gfx1201 device code. Attention
runs on the AITER unified path, and the vision tower on the image's own Triton flash kernel.

## What it patches (to make vLLM run on gfx1201)

- GPU enumeration (amdsmi init order). Without it the platform is undetected and device count reads 0.
- AITER enablement for gfx12x (upstream gates it to MI3xx).
- Triton driver activation for the GPU-less model-inspection subprocess.
- Native sampler fallback (AITER's top-k/top-p kernel doesn't build on RDNA4).
- Tool-parser streaming vs non-streaming consistency.
- `from_json` Jinja filter for tool-calling chat templates.
- MTP drafter unpadding, so `--speculative-config`'s `disable_padded_drafter_batch:true` works (the single-stream MTP speed path).
- MTP drafter multimodal mask alignment, so speculative decoding works with image inputs (otherwise the vision-placeholder mask outlives the compacted draft batch and the engine crashes).
- `torch.compile` telemetry JSON encoding, which otherwise raises `TypeError: Object of type function is not JSON serializable` at startup on this torch version (harmless but alarming: the serve came up anyway).

## Custom kernels and tuning (on by default, env-gated)

The hand-written kernels are a separate library, [libr4d](https://codeberg.org/StillDeadcode/libr4d),
written for gfx1201 rather than for any one model: paged attention, the gated-delta-net chunked scan,
a two-rank P2P all-reduce and a skinny bf16 GEMM. Each entry point is named for the geometry it is
compiled for and refuses anything else, so `import r4d; r4d.kernels()` inside the image lists exactly
what it covers. The image build clones a pinned tag and compiles it with its own `hipcc`. Everything
below is a switch on top of that.

| Env var | Default | What it does |
|---|---|---|
| `RADIANCE_USE_R4D` | `1` | master switch for the hand-written gfx1201 kernel library. Everything it covers is on with it and gone without it: the paged attention kernels behind `--attention-backend R4D`; the whole gated-delta-net layer for hybrid linear-attention models (conv + gating + cumsum, the K-gram and its triangular inverse, the chunked scan, and the decode conv and recurrent state update), measured **2.80x on the fused prefill scan** in isolation and **+1.8 to +2.2% prefill end to end**; the native head_dim-72 vision-encoder kernel; the TP=2 all-reduce; and the skinny bf16 GEMM that takes projections too small for rocBLAS to fill the machine with (see `RADIANCE_SKINNY_GEMM`). Each kernel is compiled for a specific geometry and declines per call for anything else, so all of it is inert on a model it does not fit. Once the model is loaded the log prints which R4D kernel each part of it resolved to, and for anything that resolved to none, the geometry constraint that ruled it out. Set `0` and every path reverts to stock (Triton/AITER attention, the FLA Triton scan, torch SDPA, RCCL, rocBLAS) without rebuilding the image, which is the quickest way to tell whether a problem is ours or upstream's. `--attention-backend R4D` then refuses to start rather than quietly serving something slower than what was asked for. |
| `RADIANCE_PRESHUFFLE` | `1` | preshuffled AITER FP8 blockscale GEMM |
| `RADIANCE_SKINNY_GEMM` | `1` | route bf16 projections whose weight is small enough that rocBLAS lays it out as a handful of workgroups to the R4D split-K kernel instead, for `M` in `[6,64]`. `1` covers the shapes where the kernel is a clear win on its own (the MoE gate on fine-grained MoE models: 9.6 us -> 3.2 us). `all` adds shapes that are faster but differ from rocBLAS at a bf16 ULP on a few elements in ten thousand -- most notably the gated-delta-net `in_proj_ba`, which is 480 KiB of bf16 run **48 times per step** at 28.5 us against 3.6 us for the R4D kernel. Under speculative decoding a ULP-level change can move drafting acceptance, which is why those are not in the default set; measured on a DFlash2 drafter over four paired compiles, `all` was worth **-3.9% on the decode step** with no acceptance cost. `0` takes the dispatch out entirely without disabling the rest of the library. |
| `RADIANCE_GDN_META` | `1` | build the gated-delta-net attention metadata with numpy on the host instead of the stock tensor path. Byte-identical output, and it removes host work from every step of a hybrid linear-attention model. |
| `RADIANCE_FAST_DRAFT` | `0` | **The tuned drafter stack, as one switch.** Off by default, in which case the drafter runs exactly what vLLM would run. On, it turns on every drafting change this image has measured a win from; there are no sub-knobs, because each of those was a sweep and the answer is baked in. Each half engages only where it applies, so this is safe to leave on across models. **The draft head goes to 2 bits with an exact rerank** (any drafter): stored as int2 with an asymmetric per-(row, group-of-128) scale, 0.167 GiB/rank instead of 1.18, the coarse pass emits the best 8 candidates of each 64-wide block for free and the top 32 are rescored exactly against the bf16 weight -- **+16.6% tokens/s single-stream and +12.5% at 8 concurrent** on the BetterBench prompt corpus, with acceptance unchanged, and *exact*: on 8192 real draft-head inputs the reranked token matches the bf16 argmax on every row. **A `dflash` drafter's weights go to 4 bits** (inert under `mtp`, whose head lives in the target checkpoint): signed symmetric int4, one f16 scale per group of 128 input channels, no zero point -- 4.25 bits per weight against fp8's 8 -- on two purpose-built gfx1201 kernels, an f16 matrix-core one at or below 16 rows where the GEMM is already at the memory roofline, and an int8 matrix-core one above it, because gfx1201's f16 matrix instruction is half the rate of its int8 one. One packed weight feeds both. The codes are derived at load from the weight, so this needs no calibration data, no offline step and nothing on disk. On Qwen3.8-27B with the DFlash2 drafter the draft pass falls **9.1%** at a drafter batch of 64, and paired with `RADIANCE_SKINNY_GEMM=all` the decode step falls **5.1%** for **+3.5% tokens/s** (four independent compiles per arm, 0.11% spread within each). None of it can change what the model emits: a drafter only chooses what is *proposed*, and the target verifies every token with its own untouched weights. Pair it with `RADIANCE_DRAFT_TAU=0.28` under `mtp` -- a cheaper draft step lowers the marginal acceptance a draft position has to clear. |
| (always on) | | **Shard-local draft confidence.** The draft controller needs two numbers per row, the drafted token id and its top-1 softmax probability. Both are recovered from per-rank partial reductions plus a cross-rank logsumexp, exchanging three floats per row instead of all-gathering the full vocabulary logit row on every draft slot. Exact, not an approximation. Tensor-parallel only. |
| `RADIANCE_USE_R4D_AR` | `1` | custom PCIe peer-to-peer all-reduce for TP=2, byte-identical to RCCL, falls back to RCCL if P2P is unavailable |
| `RADIANCE_USE_R4D_AR_QUANT` | `1` | compress the all-reduce payload for large messages: each group of 64 is rotated by a Walsh-Hadamard, scaled by its own amplitude and stored in 6 uniform bits, so a message costs 6.25/16 of its bf16 size. Speeds up prefill, leaves decode untouched. NOT bit-identical to RCCL (it is quantized), though the two TP ranks stay bit-identical to each other. On by default; set `0` for the exact bf16 all-reduce. |
| `RADIANCE_FUSE_RMS_QUANT` | `1` | folds group-FP8 quant into the RMSNorm epilogue |
| `RADIANCE_DYNAMIC_DRAFT` | `1` | **dynamic** MTP draft depth: per request, a per-slot confidence gate decides how deep to draft (up to `num_speculative_tokens`) and whether to take a verbatim n-gram continuation (deep on high-acceptance content like code and JSON, shallow on prose; see "Speculative decoding" below). Lossless. Needs `--speculative-config method=mtp`. **`mtp` only, and not merely by convention**: the controller works by stopping a serial loop of draft forwards early, and it hooks the proposer class that `mtp` uses. A `dflash` drafter emits every speculative position in ONE forward pass at a depth fixed when its CUDA graph is captured, so there is no loop to shorten and this knob does nothing there, set either way. |
| `RADIANCE_DRAFT_SCHEDULE` | `1:8,2:7,4:6,8:5,16:4` | `bs:max_depth` pairs (carry-forward): caps how many serial MTP forwards run at each batch size, so drafting stays deep single-stream and shallower at concurrency. The free n-gram tail is unaffected. |
| `RADIANCE_DRAFT_TAU` | `0.35` | confidence-product stop threshold: the drafter keeps drafting while the running product of its top-1 confidences stays `>= TAU`. Lower = draft deeper, higher = shallower. The baked `0.35` is tuned for the default bf16 draft head. A cheaper draft step lowers the marginal acceptance a draft position has to clear, so if you turn on `RADIANCE_FAST_DRAFT=1` set `RADIANCE_DRAFT_TAU=0.28` with it -- that pair measured +5.3% over keeping 0.35. |
| `--attention-backend R4D` | off (opt-in CLI flag, not an env var) | **R4D attention: purpose-built gfx1201 attention kernels, in place of the tuned AITER unified attention.** Prefill and decode are hand-written HIP built around a transposed score matrix, `S^T = K.Q^T`, so a wave32 matrix-core fragment gives each lane exactly one query row and the softmax stays inside the lane. In the serve the prefill kernel is 1.65x the AITER one: **+14.6% prefill throughput at 64K context** (attention is 34% of prefill GPU time there), +4.1% at 16K (11.8%), decode unchanged within noise. More accurate, not less: 1.69e-03 relative to an fp32 oracle against 2.28e-03. Requires head_dim 256, paged block 16, 6 query heads per KV head, causal decoder attention and a bf16 or fp8_e4m3 KV cache; any other shape is refused at startup with the reason. Give the drafter the same backend with `"attention_backend": "R4D"` inside `--speculative-config`. |
| `RADIANCE_RUN_BWTEST` | `1` | run the GPU topology + bandwidth sweep at startup (`rocm-bandwidth-test`, compiled into the image): device list, P2P access matrix, NUMA distances, and peak uni/bidirectional copy bandwidth per agent pair. Backgrounded and takes about a second, so it never delays the serve; the report lands in the log a few seconds in. Set `0` to skip it. |
| `RADIANCE_NUMA_BIND` | unset (off) | NUMA pinning for multi-node hosts; see below. Same as `--numa-bind`, which wins if both are given |
| `RADIANCE_BANNER_PLAIN` | `0` | set `1` for a startup banner without ANSI colour (log scrapers, CI). `NO_COLOR` does the same |

> **New in 0.6.0 - R4D attention (opt-in).** `--attention-backend R4D` swaps the tuned AITER
unified attention for kernels written here for this GPU. The core is a transposed score matrix,
`S^T = K.Q^T`: a wave32 matrix-core fragment splits a 16x16 tile column-wise, so with the score
matrix transposed each lane owns exactly ONE query row, and the running max, the sum and the
rescale all become lane-private registers instead of a cross-lane reduction. What that buys is the
freedom to make the softmax lazy - the accumulator is rescaled only when a row's max exceeds the
reference by more than the 16-bit P format can absorb, which at long context is almost never,
against every m-tile for the usual online form. The rest is layout: f16 rather than bf16 operands
(gfx1201 has no bf16 convert instruction, so an f32->bf16 costs ~6 VALU where `v_cvt_pkrtz_f16_f32`
does two conversions in one), contiguous-k fragments so each WMMA operand is one aligned
`ds_read_b128`, the block table hoisted into SGPRs a tile ahead, and an LDS-scoped barrier so
`__syncthreads()` stops invalidating the vector cache twice per tile.

**Measured in the serve, same image and flags, against `ROCM_AITER_UNIFIED_ATTN`: +14.6% prefill
throughput at 64K context** (65.6K-token prompt: 21.25 s -> 18.54 s to first token) and +4.1% at
16K. The gain scales with context because attention's share of prefill does: 34% of prefill GPU
time at 64K, 11.8% at 16K. **Decode is unchanged within noise** - attention is only ~7% of a
speculative decode step, the rest being the MoE GEMMs of the draft loop - so this is a
long-context prefill and TTFT feature, not a tokens/s feature. It is also more accurate than what
it replaces: 1.69e-03 relative error against an fp32 oracle, against 2.28e-03.

Shape support is narrow on purpose: head_dim 256, paged block size 16, 6 query heads per KV head,
causal decoder attention, bf16 query, bf16 or fp8_e4m3 KV cache. Anything else is refused at
startup with the reason and the backends that would work instead, so it cannot silently run
something it was not built for. With speculative decoding, give the drafter the same backend:
`--speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D"}'`.

> **New in 0.5.11 - all-reduce payload.** The compressed all-reduce (`RADIANCE_USE_R4D_AR_QUANT=1`) now
sends a rotated 6-bit payload instead of block-scaled fp8. Each group of 64 elements is rotated by a
Walsh-Hadamard, scaled by its own amplitude and stored in 6 uniform bits. The rotation removes the
outlier channel, which is what makes 6 uniform bits enough - and once the range problem is gone, a
float format is spending exponent bits on range it no longer needs. Net: 24% fewer PCIe bytes at
slightly better accuracy, **+7.2% prefill throughput at 16K context and +3.5% at 32K** against the
fp8 payload it replaces, with decode unchanged. The two tensor-parallel ranks remain bit-identical
to each other. Set `RADIANCE_USE_R4D_AR_QUANT=0` for the exact bf16 all-reduce, which is unchanged.

> **Also new in 0.5.10 - prefill chunking and draft depth.** `--max-num-batched-tokens` moves
2560 -> 4096 (+2.2% prefill at 16K context, +3.8% at 64K). The custom all-reduce is sized to hold
that chunk: a chunk's all-reduce is `max-num-batched-tokens * hidden * 2` bytes, and anything over
the kernel's cap falls back to RCCL, which is 2.3x slower here. The dynamic drafter's per-batch depth
caps were re-tuned on real text, `1:8,2:7,4:6,8:5` -> `1:6,2:6,4:5,8:4`: **+3.4% tokens/s
single-stream and +1.6% at eight concurrent**, with lower time per engine step at both. The draft
positions this removes were running at roughly 5% marginal acceptance against a ~7% break-even.

> **New in 0.5.10 - the draft head.** `RADIANCE_FAST_DRAFT=1` replaces the drafter's bf16 head with
a 2-bit one behind an exact rerank: **+16.6% tokens/s single-stream and +12.5% at 8 concurrent** on
the BetterBench prompt corpus, drafting acceptance unchanged, and the reranked draft token matches
the bf16 argmax on all 8192 real inputs tested. Fewer bits is not what makes it pay: a first version
at group 64 measured *slower* than a 4-bit head, because halving the group doubles the per-group
accumulator work, which is the dominant non-memory term. Group 128, quarter-split packing, and
building the bf16 value with one shift and mask instead of an integer conversion took the kernel
781 -> 350 us. Candidate width matters in a less obvious way: candidates are selected per block, so
with one candidate per block a winner sharing a block with a stronger token can never be rescored *at
any rerank depth* -- 2 bits needs 8 per block where 4 bits was fine with 1. The draft policy is tuned
for it (tau 0.28 with `1:8,2:7,4:6,8:5`), since a cheaper draft step lowers the marginal acceptance a
draft position must clear and moves the optimum deeper.

> **New in 0.5.10 - decode path.** Four changes to the speculative decode path, measured against 0.5.8 on Qwen3.8-27B-FP8 with identical flags and matched seeds: **+21% output tokens/s and -20% time per engine step at 8 concurrent streams** (with drafting acceptance matched), and **-21% step time single-stream at 64K context**. Prefill is unchanged. Three are the tuning entries above; the fourth was a bug. AITER exposes its `unified_attention` module under two names and executes it once per name, so there are two module objects with independent globals. Patching only one left roughly one attention call per engine step running AITER's stock configuration at 5596us instead of 277us, about 8.9% of all GPU time in a decode step. Every alias is now patched. The startup log reports how many it found: it must say `attn tuned-config override installed on 2 module aliases`.

> **Fixed in 0.5.7 -- tensor-parallel GPU hang under sustained load (multi-GPU only).** Builds 0.5.0 through 0.5.5-pre could hang a GPU during long agentic sessions: both cards pegged at 100% utilisation while drawing a fraction of their power cap, the driver then reporting `HW Exception ... GPU Hang`, the engine dying on an RPC timeout and the container restarting. **The cause was a dependency mismatch, not a kernel bug.** vLLM 0.26.0 pins `torch == 2.11.0`, and this image's build strips torch/torchvision pins (via vLLM's own `use_existing_torch.py`, which exists so pip does not refetch them) -- earlier 0.5.x builds then compiled against torch 2.13 / triton 3.7.1 / torchvision 0.28, a combination upstream never tests. The pinned trio (torch 2.11.0, triton 3.6.0, torchvision 0.24.1) is restored, and the hang is gone under the workload that reproduced it. Single-GPU serves were never affected, and nothing is disabled: speculative drafting and the compressed all-reduce both remain on by default. If you build your own image, take the versions upstream pins -- they are not free choices on this architecture.

All of these are baked ON in the image. Set `RADIANCE_DYNAMIC_DRAFT=0` to turn draft control off (`RADIANCE_DRAFT_SCHEDULE` and `RADIANCE_DRAFT_TAU` are values, not toggles). `RADIANCE_DYNAMIC_DRAFT` only does anything when speculative decoding is enabled; it is lossless (it changes only *how many* tokens are drafted and whether they come from MTP or a verbatim copy of earlier text, never what the model verifies).

**NUMA pinning (`RADIANCE_NUMA_BIND` / `--numa-bind`, opt-in, off by default).** On a multi-socket or multi-NUMA-node host, pin the server and its TP workers to the NUMA node(s) local to the GPUs so memory stays off the cross-node link. Set `RADIANCE_NUMA_BIND=auto` (detect from the visible GPUs) or pass `--numa-bind[=SPEC]` in the command; the flag wins. `SPEC` = `auto` \| explicit nodes (`0`, `0,1`) \| `bind=<nodes>` \| `interleave[=<nodes>]` \| `preferred=<node>` \| `none`. It is a no-op on single-node hosts and requires `--cap-add SYS_NICE` under Docker's default seccomp (already covered if you run `--security-opt seccomp=unconfined`).

## Requirements

- AMD Radeon AI PRO R9700 (gfx1201). Compiled for gfx1201 only, won't run on other GPUs. Two GPUs (TP=2) is the only configuration tested so far.
- Linux host with the amdgpu kernel driver and `/dev/kfd` + `/dev/dri`. ROCm userspace is inside the image.
- Docker with device passthrough.

## Run

On start the image prints a RADIANCE banner and runs a quick preamble (GPU count, gfx1201 check, P2P, enabled optimizations, component versions), then hands off to `vllm serve`. (It also runs a GPU topology + bandwidth sweep -- device list, P2P access matrix, NUMA distances, and peak uni/bidirectional copy bandwidth for every agent pair. `rocm-bandwidth-test` is compiled into the image and the sweep is **on by default**: it is backgrounded and takes about a second, so it never delays the serve, and its report appears in the log a few seconds in. Set `RADIANCE_RUN_BWTEST=0` to skip it.) First argument is the model path, the rest are `vllm serve` flags. The `RADIANCE_*` vars below are the custom optimizations (see the table above). They are already baked ON in the image; they are listed here so they are visible and easy to flip off.

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro \
  -v "$PWD/vllm-cache:/cache" \
  -p 8000:8000 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_PRESHUFFLE=1 \
  -e RADIANCE_USE_R4D_AR=1 \
  -e RADIANCE_USE_R4D_AR_QUANT=1 \
  -e RADIANCE_FUSE_RMS_QUANT=1 \
  -e RADIANCE_DYNAMIC_DRAFT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  stilldeadcode/vllm-radiance:0.7.4 \
    /models/YourOrg/Your-Model-FP8 \
    --served-model-name my-model \
    --quantization fp8 --kv-cache-dtype fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --enable-prefix-caching --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' \
    --no-async-scheduling \
    --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello!"}]}'
```

### compose

```yaml
services:
  vllm:
    image: stilldeadcode/vllm-radiance:0.7.4
    restart: unless-stopped
    command:
      - /models/YourOrg/Your-Model-FP8
      - --served-model-name=my-model
      - --quantization=fp8
      - --kv-cache-dtype=fp8
      - --tensor-parallel-size=2
      - --gpu-memory-utilization=0.92
      - --attention-backend=ROCM_AITER_UNIFIED_ATTN
      - --enable-prefix-caching
      - --mamba-cache-mode=align
      - '--speculative-config={"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'
      - --no-async-scheduling
      - --host=0.0.0.0
      - --port=8000
    environment:
      HIP_VISIBLE_DEVICES: "0,1"
      VLLM_ROCM_USE_AITER: "1"
      VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: "1"
      VLLM_ROCM_USE_AITER_MHA: "0"
      VLLM_ROCM_USE_AITER_MLA: "0"
      VLLM_ROCM_USE_AITER_MOE: "0"
      VLLM_ROCM_USE_AITER_LINEAR: "0"
      VLLM_ROCM_USE_AITER_FP8BMM: "0"
      VLLM_ROCM_USE_AITER_FP4BMM: "0"
      VLLM_ROCM_USE_AITER_RMSNORM: "0"
      NCCL_PROTO: Simple
      RADIANCE_PRESHUFFLE: "1"
      RADIANCE_USE_R4D_AR: "1"
      RADIANCE_USE_R4D_AR_QUANT: "1"
      RADIANCE_FUSE_RMS_QUANT: "1"
      RADIANCE_DYNAMIC_DRAFT: "1"
      # point the torch.compile / Triton / AITER caches at the mounted /cache so they persist
      # across restarts (without these the ./vllm-cache mount does nothing and every start re-autotunes)
      VLLM_CACHE_ROOT: /cache/vllm
      TORCHINDUCTOR_CACHE_DIR: /cache/inductor
      TRITON_CACHE_DIR: /cache/triton
      AITER_ROOT_DIR: /cache/aiter
      TRITON_CACHE_AUTOTUNING: "1"
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - "RENDER_GID"   # getent group render | cut -d: -f3
      - "VIDEO_GID"    # getent group video  | cut -d: -f3
    shm_size: "4gb"        # vLLM's TP workers share tensors via /dev/shm; the 64 MB default is too small
    cap_add: [SYS_PTRACE]
    security_opt: ["seccomp=unconfined"]
    ports: ["8000:8000"]
    volumes:
      - /path/to/models:/models:ro
      - ./vllm-cache:/cache   # persists the caches pointed at /cache above; first start is slow, restarts fast
```

## First run is slower

With an empty cache the first start spends a few extra minutes compiling Triton / inductor kernels before the engine comes up; it looks idle but it is compiling. (Older builds spent 15 to 20 minutes here, dominated by the gated-delta-net fp32 autotune; that path is gone on this stack.) Mount a persistent cache so restarts stay fast:

```bash
  -v /path/to/vllm-cache:/cache \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1
```

## Flags

| Flag | Suggested | Notes |
|---|---|---|
| `--tensor-parallel-size` | `2` | one rank per R9700 |
| `--quantization` | `fp8` | tuned for FP8 weights |
| `--kv-cache-dtype` | `fp8`, `bf16`, or `auto` | fp8 = 1 byte/elem (most KV capacity); bf16 / `auto` keep full precision |
| `--attention-backend` | `ROCM_AITER_UNIFIED_ATTN` | required for the tuned attention path |
| `--max-model-len` | model dependent | context length per request |
| `--max-num-seqs` | workload dependent | max concurrent sequences |
| `--gpu-memory-utilization` | `0.90` to `0.97` | VRAM fraction for weights + KV |
| `--enable-prefix-caching` | on for shared prefixes | enables automatic prefix caching; **required**: hybrid (GDN/mamba) models leave it off by default even though the engine default looks on |
| `--mamba-cache-mode` | `align` (hybrid models) | makes the linear-attention (GDN) layers prefix-cacheable; pair with `--enable-prefix-caching` on this hybrid. `none` disables mamba-layer caching; `all` is unsupported by this model |
| `--numa-bind` | omit (off) | multi-NUMA-node hosts only: pin the fleet to the GPU-local node(s). `auto` / `<nodes>` / `interleave` / `preferred=<n>` / `none`. Same as `RADIANCE_NUMA_BIND`; needs `--cap-add SYS_NICE`. See NUMA pinning above. |

Speculative decoding (MTP). Two forms depending on where the MTP head lives:

```
# Qwen3.8-27B / Qwen3.6-27B / 35B: the MTP head is in the target checkpoint, so no separate drafter
--speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'

# ...and on the 27B hybrids, give the drafter the same R4D backend if the target uses it
--attention-backend R4D --speculative-config '{"method":"mtp","num_speculative_tokens":8,"attention_backend":"R4D","disable_padded_drafter_batch":true}'

# Gemma-4-31B: the drafter is a separate model, so add "model" (and --trust-remote-code --no-async-scheduling)
--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'
```

**What `num_speculative_tokens` means here.** In stock vLLM it is a *fixed* draft length: every decode step drafts exactly that many tokens and verifies them. With `RADIANCE_DYNAMIC_DRAFT=1` (baked on) it becomes a **ceiling, not a fixed cost**: per request the controller drafts *up to* that many tokens, stops early on low-acceptance content, and may take a verbatim n-gram continuation when it matches the drafter's own guess, but the total draft is always clamped to `num_speculative_tokens`. So a larger value like **8** is the recommended default: it gives the dynamic drafter more room to run deep on high-acceptance content (code, JSON, boilerplate) without adding fixed overhead on prose. There is no separate depth-ceiling knob to keep in sync; the ceiling is `num_speculative_tokens` itself. (Set `RADIANCE_DYNAMIC_DRAFT=0` to get the classic fixed-length behavior, in which case a smaller value such as 3 is more typical.)

**DFlash drafters.** `dflash` is a different speculative shape from `mtp`: instead of running one draft
forward per speculative position, it emits all of them in a single pass, so the draft is one CUDA-graph
replay whose depth is fixed at capture. That has three consequences worth knowing.

```
--speculative-config '{"method":"dflash","model":"/models/<drafter>","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}'
```

* `RADIANCE_DYNAMIC_DRAFT` does nothing here (see the table above) -- `num_speculative_tokens` is a fixed
  width again, so it is worth tuning: on Qwen3.8-27B the per-position acceptance rate falls to ~0.10 by the
  seventh, and each extra position widens both the draft pass and the target's verify.
* The drafter's attention backend must be one that supports full CUDA graphs, or vLLM logs that it is
  "running the draft eagerly" and you lose the graph. `TRITON_ATTN` does; check the startup log for
  `Capturing dflash CUDA graphs (FULL)`.
* Prefix caching with `--mamba-cache-mode align` has only been verified against `mtp` on hybrid models; it
  is a different speculative shape and has not been re-verified here.

`disable_padded_drafter_batch:true` is the key single-stream lever (~+50% on the 27B hybrids): it drops the drafter's batch padding, and the image bakes the vLLM unpad patch this relies on. Leave it on. Note it is incompatible with async scheduling: pass `--no-async-scheduling` to disable it explicitly (otherwise vLLM auto-enables async scheduling and then disables it with a runtime warning; `--async-scheduling` would hard-error).

Prefix caching (shared system prompts, RAG, agentic context):

```
--enable-prefix-caching --mamba-cache-mode align
```

Automatic prefix caching reuses a shared prompt prefix across requests so only the new suffix is prefilled, a large time-to-first-token drop when many requests share a system prompt or document. On this **GDN hybrid you must pass both flags**: hybrid models default their prefix-caching support flag off ("experimental"), so vLLM **silently disables** prefix caching unless `--enable-prefix-caching` is given, and `--mamba-cache-mode align` is what makes the linear-attention (GDN) layers cacheable by snapshotting and restoring their conv + recurrent state at block boundaries. That restore is **verified bit-identical to a full recompute** (including under MTP), so outputs are unchanged; the win is purely latency (measured ~3.6x faster TTFT on shared prefixes). Trade-offs: align reconciles the mamba and attention page sizes, which raises the attention block size to 1664 tokens and adds one state block per linear-attention layer (slightly lower max concurrency at full context), and prefix hits land on 1664-token boundaries. Do **not** use `--mamba-cache-mode all` (unsupported by this model, raises at startup) and do **not** set `VLLM_SSM_CONV_STATE_LAYOUT=DS` (asserts under MTP + align).

Tool-calling and reasoning:

```
--enable-auto-tool-choice --tool-call-parser <parser> --reasoning-parser <parser>
```

Pass a template with `--chat-template file.jinja` if the model needs one. The image ships the `from_json` filter those templates often rely on.
