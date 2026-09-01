# vllm-radiance

vLLM inference server for the AMD Radeon AI PRO R9700 (gfx1201 / RDNA4). Bundles a working ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on this card, so you don't have to build the stack yourself.

## Tested so far

| | |
|---|---|
| Models | **Qwen3.8-27B-FP8** / **Qwen3.6-27B-FP8** (gated-delta-net hybrids, architecturally identical), **Qwen3.6-35B-A3B-FP8** (fine-grained MoE, 256 experts top-8), **Gemma-4-31B-it-FP8** (dense, sliding + global attention, vision), **Qwen3.8-27B-Quark-AWQ-MXFP4** (4-bit, see [MXFP4](#mxfp4-4-bit-checkpoints)) |
| KV cache | fp8, bf16 or `auto` |
| GPUs | 2x R9700, tensor parallel (TP=2) |

Untested: any other model, non-FP8 weights, single GPU, more than two GPUs, non-R9700 hardware. The
defaults below are a starting point for these setups, not a general recommendation.

**Qwen3.8-27B-FP8 / Qwen3.6-27B-FP8** are what everything here is tuned around: 64 layers, 48 of
them linear attention (GDN) and 16 full attention, hidden 5120, `head_dim` 256 with 6 query heads
per KV head. That is exactly the geometry the hand-written kernels are compiled for, so the whole
R4D library engages on them. The MTP head is inside the checkpoint, so speculative decoding needs
no separate drafter. They are vision-language checkpoints -- pass `--language-model-only` to skip
the vision tower when serving text.

**Qwen3.6-35B-A3B-FP8.** RDNA4-tuned fused-MoE configs and the skinny MoE-gate GEMM activate
automatically. One requirement: with `--mamba-cache-mode=align` its attention block size is 2240 and
align asserts `block_size <= max_num_batched_tokens`, so pass **`--max-num-batched-tokens >= 2240`**.

**Gemma-4-31B-it-FP8** (e.g. `RedHatAI/gemma-4-31B-it-FP8-block`). Quantization is auto-detected;
tuned block-FP8 GEMM configs load for its shapes. Text and vision both work. Not a GDN hybrid, so
drop `--mamba-cache-mode`, and it uses its own chat template and parsers. Long-context prefill is
tuned for its head-512 global layers (up to -38% TTFT at 64K, -46% at 120K; inert on other head
sizes). It carries a lot of KV (60 layers), so at a given `--gpu-memory-utilization` it wants a
smaller `--max-model-len` than the Qwen models. For a large decode speedup pair it with Google's
drafter `google/gemma-4-31B-it-assistant`, whose head-512 layer needs
`"attention_backend":"ROCM_AITER_UNIFIED_ATTN"` in the speculative config.

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
- Reasoning-parser/chat-template agreement about whether thinking is on. `Qwen3Parser` decides its
  start state from `chat_template_kwargs["enable_thinking"]` alone, but templates in the wild also
  disable thinking — i.e. pre-close `<think></think>` in the *prompt* — for
  `reasoning_effort` in `{none, off}` and for `auto_disable_thinking_with_tools` with tools present.
  The parser only ever sees the *output*, so a pre-closed block leaves no `</think>` to find and the
  whole response is filed as reasoning: **`content` comes back `null` and the answer hides in
  `reasoning`**. Measured on Qwen3.8-27B with froggeric v22.3: 50/50 requests at
  `reasoning_effort: "off"` returned empty content. The patch mirrors the template's own decision
  from the same kwargs, before `qwen3_config()` consumes it, so the streaming path (same
  `initial_state`) is fixed too. Not covered: the inline `<|think_off|>` message tag, which lives in
  the message list `__init__` never receives — pass `enable_thinking: false` or
  `reasoning_effort: "off"` instead.

## Custom kernels and tuning (on by default, env-gated)

The hand-written kernels are a separate library, [libr4d](https://codeberg.org/StillDeadcode/libr4d),
written for gfx1201 rather than for any one model. Each entry point is named for the geometry it is
compiled for and refuses anything else, so `import r4d; r4d.kernels()` inside the image lists exactly
what it covers. The build clones a pinned tag and compiles it with its own `hipcc`.

| Env var | Default | What it does |
|---|---|---|
| `RADIANCE_USE_R4D` | `1` | Master switch for the hand-written gfx1201 kernels: the paged attention behind `--attention-backend R4D`; the whole gated-delta-net layer for hybrid linear-attention models (**2.80x** on the fused prefill scan in isolation, **+1.8-2.2%** prefill end to end); the head_dim-72 vision-encoder kernel; the TP=2 all-reduce; and the skinny bf16 GEMM. Each is compiled for a specific geometry and declines anything else, so all of it is inert on a model it does not fit. The startup log prints which kernel each part of the model resolved to. Set `0` and every path reverts to stock -- the quickest way to tell whether a problem is ours or upstream's. |
| `RADIANCE_PRESHUFFLE` | `1` | preshuffled AITER FP8 blockscale GEMM |
| `RADIANCE_SKINNY_GEMM` | `1` | Route bf16 projections too small for rocBLAS to fill the machine to the R4D split-K kernel, for `M` in `[6,64]`. `1` covers the shapes that are a clear win alone (the MoE gate on fine-grained MoE models: 9.6 -> 3.2 us). `all` adds shapes that differ from rocBLAS at a bf16 ULP on a few elements in ten thousand -- notably the gated-delta-net `in_proj_ba`, 480 KiB run **48 times per step**, 28.5 us against 3.6 us. Under speculative decoding a ULP-level change can move drafting acceptance, which is why those are not in the default set; measured on a DFlash2 drafter over four paired compiles, `all` was worth **-3.9% on the decode step** with no acceptance cost. |
| `RADIANCE_GDN_META` | `1` | build the gated-delta-net attention metadata with numpy on the host instead of the stock tensor path. Byte-identical output, and it removes host work from every step of a hybrid linear-attention model. |
| `RADIANCE_FAST_DRAFT` | `0` | **The tuned drafter stack, as one switch** -- no sub-knobs; each was a sweep and the answer is baked in. Each half engages only where it applies, so it is safe to leave on across models. *The draft head goes to 2 bits with an exact rerank* (any drafter): 0.167 GiB/rank instead of 1.18, the coarse pass emits the best 8 candidates of each 64-wide block for free and the top `RADIANCE_DRAFT_RERANK` are rescored exactly against the bf16 weight -- **+16.6% tokens/s single-stream, +12.5% at 8 concurrent**, acceptance unchanged, and exact (it matches the bf16 argmax on all 8192 real inputs tested). *That guarantee is an ARGMAX guarantee.* A top-k caller -- DFlash2 asks the head for `selector_top_k`=16 candidates per position -- draws its whole pool from the reranked set, so the rerank width becomes a recall ceiling rather than a rescoring budget: at 32 it costs 5.3% of acceptance, and 64 restores it exactly. See `RADIANCE_DRAFT_RERANK`. *A `dflash` drafter's weights go to 4 bits* (inert under `mtp`): signed symmetric int4, one f16 scale per 128 input channels, no zero point, 4.25 bits/weight, on two gfx1201 kernels -- f16 matrix-core at or below 16 rows, int8 above, because this chip's f16 matrix instruction is half the rate of its int8 one. Codes are derived at load from the weight: no calibration data, no offline step, nothing on disk. On Qwen3.8-27B + DFlash2 the draft pass falls **9.1%** at a drafter batch of 64; with `RADIANCE_SKINNY_GEMM=all` the decode step falls **5.1%** for **+3.5% tokens/s**. It cannot change what the model emits -- a drafter only chooses what is *proposed*, and the target verifies every token with its own weights. Pair with `RADIANCE_DRAFT_TAU=0.28` under `mtp`. |
| (always on) | | **Shard-local draft confidence.** The draft controller reads the drafted token id and its top-1 probability from each rank's own vocab shard rather than gathering the full logits, which is exact and cuts the per-step all-reduce by 42%. |
| `RADIANCE_USE_R4D_AR` | `1` | custom PCIe peer-to-peer all-reduce for TP=2, byte-identical to RCCL, falls back to RCCL if P2P is unavailable |
| `RADIANCE_USE_R4D_AR_QUANT` | `1` | Compress the all-reduce payload: each group of 64 is rotated by a Walsh-Hadamard, scaled by its own amplitude and stored in 6 uniform bits, so a message costs 6.25/16 of its bf16 size. **+7.2% prefill at 16K context, +3.5% at 32K**; decode untouched. NOT bit-identical to RCCL, though the two TP ranks stay bit-identical to each other. Set `0` for the exact bf16 all-reduce. |
| `RADIANCE_AR_MAX_KB` | `49152` | size gate for the P2P all-reduce, in KB. Upstream hardcodes this at 48 MB, sized for a 4096-token prefill chunk; this fork restores it as a knob. **Check it against your chunk size.** The gate compares the raw bf16 byte count, and a chunked-prefill all-reduce is `--max-num-batched-tokens x hidden x 2`: at 8192 tokens and hidden 5120 that is 80 MiB, above the default, so *every prefill reduction* silently falls back to RCCL while the P2P kernel only ever sees the small decode messages. Measured on 2x R9700 (TP2, Qwen3.8-27B): all-reduce was 18.8% of prefill GPU time on RCCL at 3.145 ms per call; sizing the cap to fit moved all of it to the P2P kernel at 1.317 ms (2.18x) and gained **+0.9-7.3% prefill on fp8 and +3.1-12.8% on MXFP4**, with no change to KV cache size (the extra `2 x max_bytes` IPC scratch comes out of non-KV budget). Verify with a torch profile: `ncclDevKernel` should be absent and the R4D all-reduce call count should match `vllm::all_reduce`. |
| `RADIANCE_MXFP4` | `0` | native MXFP4 linear GEMM for Quark OCP micro-scaling checkpoints (`quant_method: quark`, mxfp4 weights *and* activations), e.g. `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Stock vLLM gates native MX compute to CDNA4 and falls back to emulation, which materialises every weight tensor in bf16 on each forward. Triton 3.6 does lower `tl.dot_scaled` on gfx12x (upconvert + bf16 WMMA), so aiter's `gemm_afp4wfp4` runs here. Output is **bit-identical** to the emulated path -- the activation quantization is the same either way -- so this is a speed change only. Measured on gate_up (17408x5120), speedup vs emulation: **6.1x at M=16, 4.7x at M=32, 2.5x at M=64**, 1.9x at M=128; 65% of the memory-bandwidth roofline at M=16. Ships tuned tiles for the two dominant Qwen3.8-27B TP2 shapes plus a generic per-band table. No effect on any other quantization scheme. |
| `RADIANCE_MXFP4_W4A8` | `0` | routes large-M (prefill) MXFP4 linears to a hand-written fp8-WMMA HIP kernel. Triton will not emit gfx1201's fp8 matrix instruction -- measured register-resident, fp8 WMMA runs **325 TFLOP/s vs f16's 160**, while Triton's own fp8 `tl.dot` manages only 43 because it upconverts to 16-bit and pays conversion on top. Against the tuned aiter path it replaces this measures **1.47-2.26x faster and 4.2x more accurate** (relative error 0.0265 vs 0.1119 against exact arithmetic), because fp8 activations beat the mxfp4 ones aiter quantizes to. Off by default because it makes the layer W4A8 rather than the checkpoint's declared W4A4: more precise, but no longer bit-identical to emulation. Requires `RADIANCE_MXFP4=1`. |
| `RADIANCE_MXFP4_W4A8_MIN_M` | `0` | batch size above which `RADIANCE_MXFP4_W4A8` takes over from aiter. **0 means never fall back**, which is both a correctness and a speed choice. Correctness: aiter's W4A4 path returns a wrong result for N=5120 K=3072 (`o_proj`) -- replayed against an fp32 reference it lands at rel=1.066 with ~1/35th of the correct magnitude, against 0.0017 for ours -- because that shape has no tuned table in `mxfp4-configs/` and falls into aiter's generic bands. At the old default of 16 this was invisible in prefill (M=17, our kernel) and silently poisoned decode (M=9, aiter). Speed: 0 used to cost ~55% of decode because the only kernel below M=16 was the prefill-tiled one; `RADIANCE_MXFP4_DECODE_MAX_M` fixes that. Note the comparison is `M > MIN_M`, so 1 would still route M=1 to aiter -- use 0. |
| `RADIANCE_MXFP4_TN4_MIN_M` | `2048` | batch size above which the W4A8 kernel switches from its TN=2 tile to the wider TN=4 one (BNF 64 -> 128). A-tile staging is 24% of the kernel and the wider tile amortises it, but only once there is enough work to fill it: measured **+10.0% at M=8192, +8.5% at 4096, +1.3% at 2048, -8.8% at 512**. Identical numerics either way. |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `0` (off; the source repo's `serve-mxfp4.sh` sets **64**, or 128 above 8 concurrent sequences) | **decode-shaped MXFP4 GEMM for small M.** The W4A8 kernel above is tiled BM=256 for prefill; at decode M is `batch x (num_speculative_tokens+1)`, so at batch 1 with SPEC=4 it is 5 -- where that tile issues 4352 WMMA per wave against 5 real rows, **51x more matrix MACs than useful**. This routes `M` up to that bound to a second kernel with TM=`ceil(M/16)` (no wasted M-fragments), split-K to fill the CUs, and BK=128 -- which reverses the prefill tuning, where BK=128 measured -34%, because that loss was purely an LDS occupancy cliff a 16-row A tile never reaches. Needs `RADIANCE_MXFP4_W4A8_MIN_M=0` to be reachable at all; at 16 the M=5 call never enters our launcher. Measured on 2x R9700 against the same build with it off: single-stream step time **35.06 -> 32.16 ms (-8.3%)**, aggregate throughput **+28.5% at 4 concurrent** and **+19.7% at 8**, prefill unchanged within 1.2%. Lossless in practice: GSM8K 500q greedy scored 97.80% both ways, 3/3 discordant, exact sign test p=1.00, at 14% less wall clock. The bound must cover `max_num_seqs x (num_speculative_tokens + 1)` or the widest verify batches fall back onto the prefill tile. Set `0` to send every M to the prefill tile. |
| `RADIANCE_MXFP4_MAX_M` | *(retired)* | read by builds up to 0.5.8, ignored since. It handed batches past M~256 back to vLLM's emulated path, where a single amortised bf16 dequant beat the fp4 kernel. That crossover only ever mattered against the aiter W4A4 path -- with `RADIANCE_MXFP4_W4A8=1` large M belongs to the fp8-WMMA kernel, which beats both -- and the fallback could not run here regardless: it reached quark's TileLang backend, which dies with `HIP runtime library (libamdhip64.so) not found` inside the vLLM worker, and the branch was specialised into the torch.compile graph during the M=8192 profile run, killing startup rather than one request. Setting it now has no effect. |
| `RADIANCE_TOPK_TRITON_MIN_ROWS` | `1` | Row count at or above which `apply_top_k_top_p` uses the Triton kernel instead of a vocabulary-wide `logits.sort()`. Upstream vLLM hardcodes 8, on the assumption that sorting a few rows is cheap; on gfx1201 at vocab 248320 the sort is slower at *every* row count -- flat ~210 us for Triton to 20 rows against 235-1904 us for the sort, and the sort is not even monotonic in rows because torch switches algorithm around 8 (which is why batch 8 profiles faster than batch 4 upstream). Speculative decode sits under the gate: the rejection sampler calls it once per step on `batch x (SPEC+1)` rows -- 5 single-stream at SPEC=4 -- and the bonus-token sampler again on 1. Measured on 2x R9700 toggling only this: **32.38 -> 31.47 ms/step (-2.8%)**, decode 78.6 -> 80.8 tok/s, acceptance identical at 1.544. The two paths are **bit-identical** (same `-inf` mask, 0.0 max difference on kept entries), so this is scheduling only. Set `8` for upstream behaviour. |
| `RADIANCE_FUSE_RMS_QUANT` | `1` | folds group-FP8 quant into the RMSNorm epilogue |
| `RADIANCE_DYNAMIC_DRAFT` | `1` | **Dynamic MTP draft depth**: per request, a per-slot confidence gate decides how deep to draft (up to `num_speculative_tokens`) and whether to take a verbatim n-gram continuation -- deep on high-acceptance content like code and JSON, shallow on prose. Lossless. **`mtp` only, by mechanism**: it works by stopping a serial loop of draft forwards early, and a `dflash` drafter emits every position in one forward pass at a depth fixed when its CUDA graph is captured, so this does nothing there. |
| `RADIANCE_DRAFT_SCHEDULE` | `1:8,2:7,4:6,8:5,16:4` | `bs:max_depth` pairs (carry-forward): caps how many serial MTP forwards run at each batch size, so drafting stays deep single-stream and shallower at concurrency. The free n-gram tail is unaffected. |
| `RADIANCE_DRAFT_RERANK` | `32` | How many of the coarse pass's candidates the int2 draft head rescores exactly against the bf16 weight. Under `mtp` the head is asked for an argmax and 32 is ample. Under `dflash` it is asked for `selector_top_k` candidates and `_radiance_topk_only` blanks everything the rerank did not touch, so this is the size of the pool a top-16 request draws from: **use 64 under `dflash`** (measured on Qwen3.8-27B + DFlash2-FP8 at ctx 0, acc/draft 1.804 at 32 against 1.904 for the bf16 head, and 1.904 at 64 for +0.23 ms/step; 128 and 256 are identical). Scale it with `selector_top_k`, not with taste. |
| `RADIANCE_DRAFT_TAU` | `0.35` | Confidence-product stop threshold: keep drafting while the running product of the drafter's top-1 confidences stays `>= TAU`. Lower = deeper. The baked `0.35` suits the default bf16 head; with `RADIANCE_FAST_DRAFT=1` use `0.28` (+5.3% over keeping 0.35), since a cheaper draft step lowers the acceptance a position must clear. |
| `--attention-backend R4D` | off (opt-in CLI flag, not an env var) | **R4D attention: purpose-built gfx1201 attention kernels, in place of the tuned AITER unified attention.** Prefill and decode are hand-written HIP built around a transposed score matrix, `S^T = K.Q^T`, so a wave32 matrix-core fragment gives each lane exactly one query row and the softmax stays inside the lane. In the serve the prefill kernel is 1.65x the AITER one: **+14.6% prefill throughput at 64K context** (attention is 34% of prefill GPU time there), +4.1% at 16K (11.8%), decode unchanged within noise. More accurate, not less: 1.69e-03 relative to an fp32 oracle against 2.28e-03. Requires head_dim 256, paged block 16, 6 query heads per KV head, causal decoder attention and a bf16 or fp8_e4m3 KV cache; any other shape is refused at startup with the reason. Give the drafter the same backend with `"attention_backend": "R4D"` inside `--speculative-config`. |
| `RADIANCE_RUN_BWTEST` | `1` | run the GPU topology + bandwidth sweep at startup (`rocm-bandwidth-test`, compiled into the image): device list, P2P access matrix, NUMA distances, and peak uni/bidirectional copy bandwidth per agent pair. Backgrounded and takes about a second, so it never delays the serve; the report lands in the log a few seconds in. Set `0` to skip it. |
| `RADIANCE_NUMA_BIND` | unset (off) | NUMA pinning for multi-node hosts; see below. Same as `--numa-bind`, which wins if both are given |
| `RADIANCE_BANNER_PLAIN` | `0` | set `1` for a startup banner without ANSI colour (log scrapers, CI). `NO_COLOR` does the same |

**R4D attention (`--attention-backend R4D`, opt-in).** Purpose-built attention kernels for this GPU
instead of the tuned AITER path. The core is a transposed score matrix, `S^T = K.Q^T`: a wave32
matrix-core fragment splits a 16x16 tile column-wise, so with the score matrix transposed each lane
owns exactly one query row and the softmax never leaves the lane. Measured in the serve against
`ROCM_AITER_UNIFIED_ATTN`, same image and flags: **+14.6% prefill throughput at 64K context**
(65.6K-token prompt, 21.25 s -> 18.54 s to first token), +4.1% at 16K, decode unchanged within
noise. The gain scales with context because attention's share of prefill does -- 34% of prefill GPU
time at 64K, 11.8% at 16K -- so this is a TTFT feature, not a tokens/s one. It is also more accurate
than what it replaces: 1.69e-03 relative error against an fp32 oracle, against 2.28e-03.

Shape support is narrow on purpose: head_dim 256, paged block 16, 6 query heads per KV head, causal
decoder attention, bf16 query, bf16 or fp8_e4m3 KV. Anything else is refused at startup with the
reason and the backends that would work instead. With speculative decoding, give the drafter the
same backend: `"attention_backend":"R4D"` in the speculative config.

All of these are baked ON in the image. Set `RADIANCE_DYNAMIC_DRAFT=0` to turn draft control off (`RADIANCE_DRAFT_SCHEDULE` and `RADIANCE_DRAFT_TAU` are values, not toggles). `RADIANCE_DYNAMIC_DRAFT` only does anything when speculative decoding is enabled; it is lossless (it changes only *how many* tokens are drafted and whether they come from MTP or a verbatim copy of earlier text, never what the model verifies).

**NUMA pinning (`RADIANCE_NUMA_BIND` / `--numa-bind`, off by default).** On a multi-NUMA-node host,
pin the server and its TP workers to the node(s) local to the GPUs. `auto` detects from the visible
GPUs; `SPEC` may also be explicit nodes, `bind=`, `interleave`, `preferred=` or `none`. No-op on
single-node hosts; needs `--cap-add SYS_NICE` under Docker's default seccomp.

## Requirements

- AMD Radeon AI PRO R9700 (gfx1201). Compiled for gfx1201 only, won't run on other GPUs. Two GPUs (TP=2) is the only configuration tested so far.
- Linux host with the amdgpu kernel driver and `/dev/kfd` + `/dev/dri`. ROCm userspace is inside the image.
- podman or docker, with device passthrough. podman is what this is developed against; the source repo's launcher auto-detects either.

## Run

On start the image prints a banner and a short preamble (GPU count, gfx1201 check, P2P, enabled
optimizations, component versions), then hands off to `vllm serve`. First argument is the model
path, the rest are `vllm serve` flags. The `RADIANCE_*` vars below are already baked ON in the
image; they are spelled out here only so they are visible and easy to flip off.

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
  stilldeadcode/vllm-radiance:0.9.3 \
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

A ready-to-edit `docker-compose.yml` with the same flags, every device / group / volume already
wired, and each knob commented lives in the source repo:
<https://codeberg.org/StillDeadcode/vllm-radiance>.

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

**What `num_speculative_tokens` means here.** Under `mtp` with `RADIANCE_DYNAMIC_DRAFT=1` (baked on)
it is a **ceiling, not a fixed cost**: the controller drafts up to that many, stops early on
low-acceptance content, and may take a verbatim n-gram continuation. So **8** is a good default --
it gives the drafter room on code and JSON without adding fixed overhead on prose. With
`RADIANCE_DYNAMIC_DRAFT=0`, or under `dflash`, it is a fixed width again and a smaller value is
typical.

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

Automatic prefix caching reuses a shared prompt prefix across requests so only the new suffix is
prefilled -- a large TTFT drop when requests share a system prompt or document. On a **GDN hybrid
you must pass both flags**: hybrid models default their prefix-caching support flag off, so vLLM
silently disables it without `--enable-prefix-caching`, and `align` is what makes the linear-attention
layers cacheable by snapshotting their conv + recurrent state at block boundaries. That restore is
**verified bit-identical to a full recompute**, so outputs are unchanged and the win is purely
latency (~3.6x faster TTFT on shared prefixes). Trade-offs: align raises the attention block size to
1664 tokens and adds one state block per linear-attention layer, so prefix hits land on 1664-token
boundaries and max concurrency at full context drops slightly. Do **not** use `--mamba-cache-mode
all` (unsupported, raises at startup) or set `VLLM_SSM_CONV_STATE_LAYOUT=DS` (asserts under MTP +
align).

Tool-calling and reasoning:

```
--enable-auto-tool-choice --tool-call-parser <parser> --reasoning-parser <parser>
```

Pass a template with `--chat-template file.jinja` if the model needs one. The image ships the `from_json` filter those templates often rely on.

## MXFP4 (4-bit) checkpoints

Quark OCP micro-scaling checkpoints (`quantization_config.quant_method: quark`, mxfp4 weights *and*
activations, group 32, e8m0 scales) run **natively** on gfx1201 with `RADIANCE_MXFP4=1` -- e.g.
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Drop `--quantization`; the runtime reads the method from
`config.json`. The native path is bit-identical to vLLM's emulation and multiples faster (6.1x at
M=16 on gate_up 17408x5120), and with `RADIANCE_MXFP4_W4A8=1` the linears go to a hand-written
fp8-WMMA HIP kernel instead. On the 27B that is 9.4 GiB of weights per GPU against ~12.6 for FP8,
and the headroom goes straight into KV.

This does **not** work by pointing `docker run` at AMD's checkpoint, for two reasons that both bite
at load: AMD's release leaves its bf16 `mtp.*` layers out of both `exclude` and `layer_quant_config`
so vLLM asserts on a half-width parameter, and the libr4d pinned in this image predates the
gated-delta-net overflow fix, which NaNs this model's output. Both are handled by two scripts in the
source repo:

```bash
git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 && cd radiance-vllm-mxfp4
./setup-mxfp4.sh      # host check, image pull, checkpoints, kernels
./serve-mxfp4.sh      # serve on http://localhost:8080/v1
```

`setup-mxfp4.sh` is idempotent and needs no Python, ROCm or HF CLI on the host -- it runs everything
that needs them inside this image. `./serve-mxfp4.sh --help` lists the knobs; the full write-up,
including the measurements behind each default, is in that repo's README.

## ParoQuant (0.10.0)

`paroquant/` carries a W4A8 serving stack for ParoQuant checkpoints (int4 group-128 asymmetric +
learned pairwise Givens rotations + channel scaling; `quant_method: "paroquant"`), e.g.
z-lab/Qwen3.8-27B-PARO. Not baked into the image: `paroquant/run_paroquant.sh` builds the kernel
module into site-packages at container start and registers the quant method via the stdlib
sitecustomize (the site-packages one is shadowed on Ubuntu). Kernels are gfx1201 hand-written HIP:
fused rotation+quantization prologue, zero-point-as-row-sum-correction GEMM with partition select
for merged layers, per-token activation scales on the prefill band (`RADIANCE_PQ_PTOK=0` reverts
to per-group). Gated by `paroquant/run.sh` (harness) and `paroquant/test_module.sh` (integration
vs the real checkpoint); numbers and design notes in `paroquant/RESULTS.md`.
