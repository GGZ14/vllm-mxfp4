# vllm-radiance

A vLLM inference server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It bundles a working
ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on
this card, plus RDNA4-tuned GEMM / attention / all-reduce paths and a dynamic MTP draft controller, so you
don't have to build the stack yourself.

> **Status: super early dev (v0.7.4). Experimental.** Everything here was built and measured on a few exact
> setups: **Qwen3.8-27B-FP8** and **Qwen3.6-27B-FP8** (gated-delta-net hybrids, architecturally identical),
> **Qwen3.6-35B-A3B-FP8** (fine-grained MoE, 256 experts / top-8),
> **Gemma-4-31B-it-FP8** (block-fp8, sliding + global attention, vision), and
> **Qwen3.8-27B-Quark-AWQ-MXFP4** (4-bit OCP micro-scaling, see [MXFP4](#mxfp4-4-bit-checkpoints)), all with
> fp8 (or bf16/`auto`) KV cache on two R9700 GPUs (tensor parallel). Other models, other weight formats,
> single or 3+ GPUs, and non-R9700 hardware are untested. Expect rough edges and breaking changes. Not
> production hardened. Use at your own risk.

This repository is the **source** for the image published as `stilldeadcode/vllm-radiance` on Docker Hub.
See **[DOCKERHUB.md](DOCKERHUB.md)** for the full description, the complete environment-variable / knob
reference, tested configuration, and stack versions.

## Build

Everything the build needs is in this directory (a flat Docker build context). The version string lives in
one place, the `VERSION` file, which the tag and the build-arg both read:

```bash
docker build -t vllm-radiance:$(cat VERSION) --build-arg RADIANCE_VERSION=$(cat VERSION) .
```

That single command builds everything from source, in four stages. **builder** compiles PyTorch, Triton,
torchvision, AITER, and vLLM for `PYTORCH_ROCM_ARCH=gfx1201` against the official `rocm/dev-ubuntu-24.04`
base (digest-pinned) and leaves the wheels in `/wheels` (it also builds `rocm-bandwidth-test` for the startup
sweep). **rocmprune** (`prune_rocm.sh`) cuts the 19 GB ROCm tree down to this one GPU architecture.
**assemble** installs the wheels, applies the RDNA4 correctness patches, and clones and compiles
[libr4d](https://codeberg.org/StillDeadcode/libr4d) -- the hand-written gfx1201 kernel library (paged
attention, the fused gated-delta-net prefill scan, the P2P all-reduce, the MoE router GEMM) -- with the
image's own `hipcc`. **final** is the release image: a clean `ubuntu:24.04` that receives only the pruned ROCm tree, the
venv, and the entrypoint, so neither the build toolchain nor the wheels ever reach the published image. No
prebuilt component wheels, no rotating wheel indexes, and no checked-in binaries go into the image. It is a
long build (a full PyTorch compile); expect it to run for hours on a many-core box.

That structure is what keeps the download reasonable: the stock ROCm base is 7.4 GiB compressed on its own,
most of it device code for GPUs this image cannot run on. Pruning it to gfx1201 and shipping an allowlist
takes the image from 9.35 GiB compressed to **3.66 GiB**. Note the prune must happen in a stage the release
stage copies *from* -- deleting files in a layer stacked on the base reclaims nothing.

The release stage still ships a working **compiler** (hipcc, g++, and the C++/Python headers). That is
not slack to trim: AITER JIT-compiles its kernels on first use, inside the running container, so an
image without those headers boots and then dies on the first AITER module build. The build asserts it
by compiling and importing a pybind11 HIP module.

The component pins are the `ARG`s at the top of the `Dockerfile` (`TORCH_VERSION`, `TRITON_VERSION`,
`TORCHVISION_VERSION`, `AITER_VERSION`, `VLLM_VERSION`). Each one is both the git tag that gets compiled and
the version the resulting wheel reports, and the build asserts the two agree, so `pip show` and the startup
banner can be trusted. If you just want a known-good image without building, pull the published one:
`docker pull stilldeadcode/vllm-radiance`.

**Do not bump torch / triton / torchvision on their own.** They are not independent choices: vLLM pins the
torch version it is tested against, torch pins its triton, and torchvision ships a matching release. The
build runs vLLM's own `use_existing_torch.py`, which *strips* those pins -- but that exists so pip does not
re-download torch, not as licence to install a newer one. Builds 0.5.0 through 0.5.4 compiled against a
newer trio and hung a GPU under sustained tensor-parallel load; restoring the pinned versions fixed it with
no code change. If you override these with `--build-arg`, move them together and soak-test under real load.

## Run

`docker-compose.yml` is the canonical way to serve. Point it at your model directory and your GPU group GIDs,
then:

```bash
# put your model at ./models/Qwen/Qwen3.8-27B-FP8  (or set MODELS=/your/model/dir)
docker compose up -d          # start; follow with: docker compose logs -f
docker compose down           # stop
```

The compose defaults target **Qwen3.8-27B-FP8**, the model the image is tuned around: a gated-delta-net
hybrid of 64 layers (48 linear attention + 16 full attention), hidden 5120, attention `head_dim` 256 with 6
query heads per KV head, GDN key/value head dim 128 and a width-4 causal conv. **Qwen3.6-27B-FP8 has exactly
the same shape**, so it takes the same tuned paths and the same flags; only the checkpoint path changes. Both
carry their MTP head in the checkpoint, so speculative decoding needs no separate drafter, and both are
vision-language checkpoints served text-only here via `--language-model-only`.

To serve the fine-grained-MoE **Qwen3.6-35B-A3B-FP8**, point it
at that model and raise the batch-token budget: `--max-num-batched-tokens` must be **≥ 2240** (align mode
reconciles the GDN state to attention block size 2240). Its tuned
MoE config and the R4D gate GEMM are baked in and turn on automatically.

To serve **Gemma-4-31B-it-FP8** (block-fp8, e.g. `RedHatAI/gemma-4-31B-it-FP8-block`), just point the compose
at it: the quantization is auto-detected from `config.json` (compressed-tensors, 128x128 blocks), the tuned
GEMM configs load by shape, and the long-context prefill attention path is tuned for its head-512 global
layers. Drop the Qwen-specific `--mamba-cache-mode` and chat template / tool-reasoning parsers (it is not a
GDN hybrid and uses its own template). Its vision tower works as-is. Note it is a *big-KV* model (60 layers,
50 sliding + 10 global), so give it a smaller `--max-model-len` than the Qwen models at the same
`--gpu-memory-utilization`.

Gemma-4-31B also supports **MTP speculative decoding** for a large decode speedup, using Google's official
drafter `google/gemma-4-31B-it-assistant` (vLLM loads it as an MTP model). It is lossless (the target
verifies every drafted token) and the dynamic draft controller applies to it. Its one requirement on this
card: the drafter has a head-512 layer, so pass `"attention_backend":"ROCM_AITER_UNIFIED_ATTN"` in the
speculative config (the usual `flash_attn` caps at head 256). For example:
`--speculative-config '{"method":"mtp","model":"/models/google/gemma-4-31B-it-assistant","num_speculative_tokens":8,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}' --no-async-scheduling`.

### MXFP4 (4-bit) checkpoints

Quark OCP micro-scaling checkpoints (`quantization_config.quant_method: quark`, mxfp4 weights *and*
activations, group 32, e8m0 scales) run **natively** with `RADIANCE_MXFP4=1` -- e.g.
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Drop `--quantization`: the runtime reads the method from `config.json`
and routes it itself. `run_mxfp4_minm.sh` at the repo root is a complete worked launch, annotated with every
deliberate difference from the FP8 setup and why.

Without this, vLLM falls back to emulated MXFP4, which materialises every weight tensor in bf16 on each
forward. Nothing in the way was a compiler limitation -- Triton 3.6 does lower `tl.dot_scaled` on gfx1201 --
just three soft gates, all handled in `patch_quark_mxfp4.py`: an `is_fp4_avail()` allowlist that omits
gfx1201, an aiter module path that moved in 0.1.17, and gfx1250 tiles that ask for
`matrix_instr_nonkdim=32` when this card's WMMA is 16x16x16 only (`mxfp4-configs/` pins 16 across every
band). The native path is **bit-identical to emulation** -- the activation quantization is the same either
way -- so it is a speed change with no quality dimension: measured on gate_up 17408x5120, **6.1x at M=16,
4.7x at M=32, 2.5x at M=64**.

### Known: one gated-delta-net head emits NaN on 0.7.4

On this checkpoint, the input to the 64 `N=5120, K=3072` linears (gated-delta-net `out_proj`,
attention `o_proj`) contains NaN on **one TP rank**, on an otherwise healthy and coherent serve.
The count is exactly `M x 128` -- 512 at M=4, 640 at M=5, 2176 at M=17 -- i.e. one whole head of
the gated-delta-net value path (`head_v` is 128; attention's `head_dim` is 256, so this is not
attention). It is consistent across every M measured.

This did not happen on 0.5.8, which served the same layers with the same kernel.

It matters because the two MXFP4 activation paths react to it differently. aiter quantizes
activations to mxfp4, where NaN squashes to a finite code and the damage is contained. The W4A8
path quantizes to **per-token fp8**, where one NaN makes the row's amax NaN, hence the row scale
NaN, hence the entire row NaN -- which then propagates through the residual stream and destroys
the model. `RADIANCE_MXFP4_SANITIZE` (default `1`) zeroes non-finite activations before
quantizing, which is what makes the W4A8 path usable at all here.

Cost, WikiText-2, 300 chunks x 3000 chars, 208,539 tokens:

| build | PPL | top-1 |
|---|---|---|
| 0.5.8 MXFP4 W4A8 (no NaN present) | 8.3335 | 54.18% |
| 0.7.4, all 304 layers on our kernel + sanitize | **8.4004** | 54.02% |
| 0.7.4, those 64 layers on aiter instead | 8.4977 | 53.80% |

So sanitizing is the best available handling, not the cause: it is 1.14% better than letting aiter
serve those layers. The residual +0.80% against 0.5.8 is the NaN itself.

Attribution is complete by elimination. **Not** the W4A8 kernel (verified against exact fp32 across every M, both tile
paths, N on and off a 64 multiple, exponent spreads to d=60, no out-of-bounds writes, every output
element written, bit-identical replay of live operands -- and 37x more accurate than aiter at that
shape, 0.0014 vs 0.053). **Not** R4D attention (8.4048 with AITER attention, unchanged), so the
+47.7% long-context prefill costs nothing in quality. **Not** the rotated 6-bit all-reduce
(8.3975 with the exact bf16 payload, a 0.03% difference that matches the historical fp8-vs-exact
all-reduce spread). That leaves the R4D gated-delta-net path, which cannot be A/B'd here: `RADIANCE_USE_R4D=0` sends the prefill scan back
to FLA Triton, which tries to allocate 128 GiB regardless of `--max-num-batched-tokens` or
`--max-model-len` -- precisely the failure R4D exists to avoid.

**`RADIANCE_MXFP4_MAX_M` is retired** (it was read up to 0.5.8). It handed big batches back to emulation as
a throughput win on paper; in practice quark's TileLang backend cannot initialise inside a vLLM worker, and
the branch was specialised into the compile graph during the `max-num-batched-tokens` profile run -- so it
killed startup rather than one request. That also means the stock emulated path cannot serve these
checkpoints here at all, which makes the native kernel the only way to run them on this card, not merely the
faster one. With `RADIANCE_MXFP4_W4A8=1` the crossover is moot anyway: large M goes to the fp8-WMMA kernel,
which beats both the aiter path and emulation.

`RADIANCE_MXFP4_W4A8=1` additionally routes large-M (prefill) linears to a hand-written fp8-WMMA HIP kernel
(`radiance_mxfp4_fp8.hip`). Triton lowers `tl.dot_scaled` by upconverting e2m1 to bf16 and using the 16-bit
WMMA; register-resident on this card, **fp8 WMMA runs 325 TFLOP/s against f16's 160**, while Triton's own
fp8 `tl.dot` manages 43 because it upconverts and pays conversion on top. Against the tuned aiter path it
measures **1.47-2.26x faster and 4.2x more accurate** (0.0265 vs 0.1119 relative error), since fp8
activations beat the mxfp4 ones aiter quantizes to. It is **off by default because it changes numerics**:
the layer becomes W4A8 rather than the checkpoint's declared W4A4 -- more precise than what the model was
calibrated against, but no longer bit-identical. The N tile is M-keyed (`RADIANCE_MXFP4_TN4_MIN_M`, default
2048): the wide tile amortises A-tile staging for +10% at M=8192 but cannot fill below ~2048 rows.
`RADIANCE_MXFP4_W4A8_MIN_M` (default 256) is where it takes over from aiter; below that the W4A4 path wins,
since these tiles are sized for prefill.

Two practical notes. **4-bit weights leave far more room for KV**: on 2x R9700 the 27B MXFP4 body occupies
9.24 GiB/GPU against roughly 12.6 for the same model in FP8, and that headroom goes straight into context.
And **do not quantize the MTP drafter to MXFP4**: at n=8 the drafter is 34% of decode weight traffic so it
looks like an obvious target, but data-free RTN (~11.6% relative error) drops mean acceptance from 2.5 to
2.21 and AWQ calibration does not rescue it -- for a drafter, accuracy *is* throughput. An fp8 e4m3
per-channel drafter (~2-3% error) holds acceptance at 2.60-2.80 and is what the worked script serves.

All tunables are `${VAR:-default}` in the compose file; override via the shell or a `.env` file without
editing it. The full knob list (kernel toggles, draft controller, AITER routing, …) is in
[DOCKERHUB.md](DOCKERHUB.md).

## What's inside

Everything below is baked into the image; the tuned paths are env-gated and on by default. See
**[DOCKERHUB.md](DOCKERHUB.md)** for the per-knob reference: every flag, its default, and what it does.

- **gfx1201 correctness patches** (always on): GPU enumeration, AITER enablement, native sampler fallback,
  MTP drafter unpad + multimodal draft-mask alignment, tool-parser + `from_json` chat-template filter, and
  an attention LDS fit that shrinks the staged K/V tile into the R9700's 64 KiB shared memory for any head
  size and KV dtype (AITER sizes it for a larger LDS; without this, 2-byte KV at head 256 and fp8 KV at
  head 512 both abort at CUDA-graph capture).
- **RDNA4-tuned kernels**: preshuffled FP8 blockscale GEMM, unified-attention tiling (fp8 + bf16/`auto` KV,
  plus a head-size-keyed long-context prefill config for models with large attention heads such as Gemma's
  head-512 global layers), fused RMSNorm+quant, an fp16 matrix-core (WMMA) gated-delta-net path, a
  widened channel block for the gated-delta-net prefill convolution (a 16-byte-per-lane access instead
  of 4, which also defuses a power-of-two row pitch the layer's `split()` view creates: 2.2x on that
  kernel, bit-identical), a TP=2 P2P
  one-shot all-reduce (optional compressed payload), and a native head_dim-72 ViT flash kernel for multimodal
  vision encoders.
- **R4D attention** (opt-in, `--attention-backend R4D`): purpose-built attention kernels for this GPU, written
  in HIP rather than tuned out of a vendor library. They compute the score matrix transposed, `S^T = K.Q^T`,
  so a wave32 matrix-core fragment hands each lane exactly one query row and the softmax never leaves the
  lane, at a *smaller* error against an fp32 reference than the kernel it replaces. Measured in the serve
  against the tuned AITER unified attention on the same image: **+14.6% prefill throughput at 64K context**
  (the attention kernel itself is 1.65x, and it is 34% of prefill GPU time there), +4.1% at 16K, and decode
  unchanged within noise -- attention is only ~7% of a speculative decode step.
  Needs head_dim 256, paged block 16, 6 query heads per KV head, causal attention and a bf16 or fp8 KV
  cache; any other shape is refused at startup with the reason, and nothing changes unless you ask for it.
- **Fine-grained MoE support** (e.g. Qwen3.6-35B-A3B): RDNA4-tuned fused-MoE Triton configs (always on;
  removes the stock config's `M>=96` cliff for a lower prefill TTFT, lossless), plus a custom bf16 MoE-gate
  GEMM for the `n` in `[6,16]` band that rocBLAS serves poorly. Both inert on
  models they do not apply to.
- **Native MXFP4** for Quark OCP micro-scaling checkpoints (`RADIANCE_MXFP4`), bit-identical to vLLM's
  emulation and multiples faster, plus an optional hand-written fp8-WMMA W4A8 prefill GEMM
  (`RADIANCE_MXFP4_W4A8`) that reaches the fp8 matrix instruction Triton will not emit. See
  [MXFP4](#mxfp4-4-bit-checkpoints).
- **Lossless dynamic MTP drafting**: a per-request confidence gate plus verbatim n-gram tail that varies
  draft depth without changing what the model verifies.
- **Prefix caching that works on the GDN hybrid** (enabled in the compose): hybrid models leave automatic
  prefix caching off by default, so it is turned on explicitly with `--enable-prefix-caching
  --mamba-cache-mode=align`. Align mode snapshots and restores the linear-attention (GDN) recurrent state at
  block boundaries (verified bit-identical to full recompute, including under MTP), giving a large TTFT drop
  on shared prefixes (system prompts, RAG, agentic context).
- **Startup topology + bandwidth sweep** (`RADIANCE_RUN_BWTEST`, on by default): device list, P2P access
  matrix, NUMA distances, and peak uni/bidirectional copy bandwidth per agent pair, from a
  `rocm-bandwidth-test` compiled into the image. Backgrounded and about a second, so it never delays the
  serve. Set `0` to skip.
- **Optional NUMA pinning** (`--numa-bind`, off by default) for multi-NUMA-node hosts.

## Layout

Flat build context: the runtime Python modules (`radiance_*.py`), the `patch_*.py` fixes, the `fp8-configs/`
`moe-configs/` and `mxfp4-configs/` GEMM configs, the chat template, `Dockerfile`, and `docker-compose.yml` all live at the repo
root so `docker build .` works directly. `prune_rocm.sh` is the ROCm slimming step (it self-checks: the
arch's own kernels must survive and hipcc must still link a HIP shared object, since AITER JITs at runtime).

The HIP kernels are no longer in this repo. They live in
[libr4d](https://codeberg.org/StillDeadcode/libr4d), a library of kernels for gfx1201 rather than for
any one model, and are pinned by tag (`R4D_VERSION` in the `Dockerfile`), which the build asserts
against the version the compiled library reports. `make r4d` clones and builds that pinned tag here
for development, so a locally built `r4d.so` matches the one in the image.

R4D entry points are named for the geometry they are compiled for -- `attn_decode_h256_gqa6_fp8kv`,
`gdn_chunk_scan_k128_v128_c64_bf16`, `ar_oneshot_2rank_exact` -- and reject a mismatch rather than
running, so which kernels an engine binds is visible in the startup log and in `r4d.kernels()`.

One HIP kernel does stay here: `radiance_mxfp4_fp8.hip`, the MXFP4 W4A8 fp8-WMMA GEMM. It is specific to
this fork rather than general to gfx1201, so it is compiled by the image build directly and `make
radiance_mxfp4_fp8.so` rebuilds it against the image toolchain during development. `run_mxfp4_minm.sh` is a
worked MXFP4 launch kept alongside it (a podman invocation from the box it was measured on, not part of the
build).
