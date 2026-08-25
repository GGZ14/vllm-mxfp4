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

### Running this build

First build the checkpoint, once. `fp8_mtp.py` needs torch; if the host has none, run it inside the
image, which does:

```bash
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4

docker run --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/models:/models" -v "$PWD:/repo" \
  --entrypoint bash stilldeadcode/vllm-radiance:$(cat VERSION) -lc '
    python3 /repo/fp8_mtp.py \
      "$(ls -d /root/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/*/ | head -1)" \
      /models/Qwen3.8-27B-MXFP4-mtpfp8'
```

On a host that already has torch, `./fp8_mtp.py <src-snapshot> $HOME/models/Qwen3.8-27B-MXFP4-mtpfp8`
does the same thing without the container.

This is **not** an optimization step you can skip -- AMD's release does not load with MTP enabled.
It ships the MTP head bf16 but names `mtp.*` in neither `exclude` nor `layer_quant_config`, so
vLLM's quark config falls through to `global_quant_config` (mxfp4), builds a packed uint8 weight of
half the input width, and dies loading the full-width bf16 tensor into it: `AssertionError:
Attempted to load weight (torch.Size([5120, 10240])) into parameter (torch.Size([5120, 5120]))`.
`fp8_mtp.py` requantizes the eight MTP projections to fp8 e4m3 per-channel and writes the matching
`layer_quant_config`, which is what makes the head loadable. Why fp8 and not mxfp4 is
[below](#the-drafter-is-fp8). It reads and rewrites one safetensors file, needs only torch, and
takes about fifteen minutes.

Then:

```bash
MODELS=$HOME/models ./run_mxfp4_074.sh
```

That is the whole thing. On the first launch it clones libr4d at a pinned commit and compiles it
inside the same image it will be loaded from, which takes a few minutes; the result is cached in
`~/.cache/radiance-libr4d` and every later launch reuses it. This repo's vLLM patches and the W4A8
kernel are applied in the container prelude, so no image rebuild is involved either.

The build step exists because the fix this build depends on is upstream but unreleased. The GDN
overflow guards are merged (StillDeadcode/libr4d PR #1), but the only tag is still `v0.4.0` and the
0.7.4 image pins that tag -- so the `r4d.so` **inside the image predates the fix** and NaNs the
gated-delta-net output on this model: WikiText-2 perplexity 653586 against 8.3706. Once deadcode
cuts a tag and ships an image pinning it, the whole step disappears.

Knobs, if you want them:

| | |
|---|---|
| `R4D_SO=/path/to/libr4d` | use your own checkout instead; nothing is rebuilt behind your back |
| `R4D_PIN=<sha>` | build a different libr4d commit (each is cached separately) |
| `AUTO_R4D=0` | skip the build and run the image stock kernel -- see below |

The pin is deliberate. `main` is a moving branch, and nothing in radiance version-checks the library
it loads: all six `radiance_*.py` modules just `import r4d` and call it. If a later commit renames an
entry point or changes a compiled-in geometry constant, the registry and the constants disagree,
`radiance_gdn.py` sets `ENABLED = False`, and the build **falls back to the Triton path with one line
on stderr** -- you lose the performance rather than getting an error. `R4D_PIN=main` builds the tip instead, but note the cache
is keyed by that string, so it is fetched once and then reused -- `rm -rf ~/.cache/radiance-libr4d/main`
to pick up newer commits. Either way, read the R4D selections table printed at startup
(`RADIANCE_R4D_REPORT=1`, on by default) and confirm the GDN and attention kernels actually bound.

Verified reproducible: the automatic build produces an `r4d.so` byte-identical (sha256
`3026297b...`) to the one every number below was measured with.

`AUTO_R4D=0` leaves you on the stock kernel, where the W4A8 path is unusable. If you must run
stock, set `RADIANCE_MXFP4_SANITIZE=1`, which zeroes non-finite activations and gets you to
perplexity 8.4004 -- worse than the fix, but serviceable.

#### The drafter is FP8

Checkpoint is `Qwen3.8-27B-MXFP4-mtpfp8`: AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` body with the MTP
drafter requantized to fp8 by [`fp8_mtp.py`](fp8_mtp.py). The drafter must NOT be MXFP4 -- 4-bit
costs more acceptance than it saves in bandwidth, and AWQ does not rescue it. Data-free RTN
measured ~11.6% relative error and cost acceptance 2.5 -> 2.21; AWQ calibration improved that by
0-5% (the alpha search chose 0.1-0.2, and 0.0 for `mtp.fc`, because MXFP4's per-32 E8M0 block
exponent already does most of what per-channel scaling would). FP8 e4m3 per-channel is ~2-3%
relative error and holds acceptance at 2.60-2.80, removing ~17% of decode weight traffic instead
of 25% -- the smaller win that actually holds.

### Measured

Against the 0.5.8 MXFP4 build, same box (2x R9700, TP2), same harness, `SPEC=4`:

| | 0.5.8 | 0.7.4 | |
|---|---|---|---|
| prefill 7.8k | 3873 | **4387** | +13.3% |
| prefill 26k | 3445 | **4138** | +20.1% |
| prefill 104k | 2310 | **3143** | +36.1% |
| prefill 182k | 1736 | **2511** | +44.6% |
| prefill 260k | 1393 | **2089** | +49.9% |
| decode short / medium | 63.0 / 67.4 | **67.1 / 67.5** | +6.5% / +0.1% |
| WikiText-2 PPL | 8.3335 | 8.3719 | +0.46% |

KV cache 857,399 tokens at `GPU_UTIL=0.98`. All 304 linear layers run the W4A8 fp8-WMMA kernel;
`aiter` is not used at all now that `RADIANCE_MXFP4_W4A8_MIN_M` defaults to 0. The prefill gain
scales with context because it is mostly R4D's paged attention, whose share of prefill grows with
sequence length.

**The decode GEMM (`RADIANCE_MXFP4_DECODE_MAX_M`, default 48), measured against the same 0.7.4 with
it off.** Report step time, not tokens/s: tokens/s swings ~14% on draft-acceptance luck alone at
fixed config, and `ms/step = 1000 x (accepted/draft + 1) / tok_s` divides that out.

| | off | on | |
|---|---|---|---|
| single stream, ms/step | 35.06 | **32.16** | -8.3% |
| ms/step at 32k context | 36.53 | **33.47** | -8.4% |
| aggregate tok/s, 4 concurrent | 170.1 | **218.5** | +28.5% |
| aggregate tok/s, 8 concurrent | 295.0 | **353.1** | +19.7% |
| prefill, all five lengths | — | — | unchanged (-0.3 to -1.2%) |
| GSM8K 500q, greedy | 97.80% | 97.80% | 3/3 discordant, sign test p=1.00 |

Batched gains most because at M=20-40 aiter's tuned band uses `NUM_KSPLIT=1`, which leaves the grid
underfilled, while this kernel keeps split-K. GSM8K also ran **14% faster wall** (375.5s -> 322.7s)
on slightly *more* generated tokens.

`run_mxfp4_074.sh --help`-style knobs worth knowing: `R4D_ATTN` (default 1), `FAST_DRAFT`
(default 1, the int2 draft head, +6.5% decode), `MIN_M` (0), `RADIANCE_MXFP4_DECODE_MAX_M` (48),
`SPEC` (4 -- measurably better than 6 and 8 here, re-confirmed on this build), `CHUNK` (8192),
`GPU_UTIL` (0.98).

### The gated-delta-net NaN (fixed upstream)

libr4d v0.4.0 produces NaN in the gated-delta-net output on this model -- WikiText-2 PPL **653586**
with the W4A8 path and no mitigation. Three exponent overflows, all the same shape: an unguarded
`__expf` on an inactive lane or a split-form half, giving `0 * INF = NaN`.

1. **`kkt_solve`, padding rows.** `gi` is forced to 0 for `i >= rows` while `gb[j]` keeps its real
   negative cumsum, so `d = -gb[j]` is large POSITIVE -- the opposite of the "never positive"
   invariant the code asserts, which holds only for live rows. The NaNs land in padding rows of the
   64x64 tile and the blocked inverse merges the whole tile with WMMA, so they reach live rows.
2. **`chunk_scan`, split-form halves.** `e^{g_i-c}.e^{c-g_j}` with `cref` at the chunk midpoint
   gives each half +/-(gate span)/2; a span past ~176 sends one to +INF and the other to 0.
3. **`chunk_scan`, `V'` staging.** The dominant one, and only visible across chunks. `V' = V.gv[t]`
   is staged in **bf16**, so `gv` must leave room for `V` under bf16's 3.4e38 ceiling. Clamping at
   `e^88` still NaNs; `e^80` leaves margin.

Fixed in **StillDeadcode/libr4d PR #1** (merged 2026-08-24). Not in a tag yet, hence the build-from-
main step above.

Clamp value, measured over 208,539 WikiText-2 tokens with no other mitigation:
70 -> 8.3841, **80 -> 8.3706**, 83 -> 8.3728, reference 8.3335, stock 653586. (Those were measured
before the fold was widened; on the current build the same configuration reads 8.3719, against
8.3736 with the original fold table.)

**The clamp bounds the damage; it does not remove the cause** -- and upstream sharpened this point
when merging. The original note here claimed the clamped product "evaluates to 0, which is the
correct answer". That is wrong: what leaves range is the distance from `cref`, not `g_i-g_j`, so on
a span-200 chunk the last token's own diagonal -- and its `e^{gl-g_t} ~ 1` weight into the state,
which the next chunk reads -- are *attenuated* by `e^{80-(cref-g_t)}` rather than correctly
vanishing. The real fix is to stop splitting a weight that is provably <= 1 into a huge x tiny
pair: stage `V'` in fp32, or apply `e^{gl-g_t}` directly on the state path.

Fixing this also removed a second symptom: `RADIANCE_FAST_DRAFT` used to hang a worker at chunk
8192 because the draft head was being fed NaN like everything else downstream of the GDN core.

**`RADIANCE_MXFP4_MAX_M` is retired** (it was read up to 0.5.8). It handed big batches back to emulation as
a throughput win on paper; in practice quark's TileLang backend cannot initialise inside a vLLM worker, and
the branch was specialised into the compile graph during the `max-num-batched-tokens` profile run -- so it
killed startup rather than one request. That also means the stock emulated path cannot serve these
checkpoints here at all, which makes the native kernel the only way to run them on this card, not merely the
faster one. With `RADIANCE_MXFP4_W4A8=1` the crossover is moot anyway: large M goes to the fp8-WMMA kernel,
which beats both the aiter path and emulation.

`RADIANCE_MXFP4_W4A8=1` additionally routes linears to a hand-written fp8-WMMA HIP kernel
(`radiance_mxfp4_fp8.hip`). Triton lowers `tl.dot_scaled` by upconverting e2m1 to bf16 and using the 16-bit
WMMA; register-resident on this card, **fp8 WMMA runs 325 TFLOP/s against f16's 160**, while Triton's own
fp8 `tl.dot` manages 43 because it upconverts and pays conversion on top. Against the tuned aiter path it
measures **1.47-2.26x faster and 4.2x more accurate** (0.0265 vs 0.1119 relative error), since fp8
activations beat the mxfp4 ones aiter quantizes to. It is **off by default because it changes numerics**:
the layer becomes W4A8 rather than the checkpoint's declared W4A4 -- more precise than what the model was
calibrated against, but no longer bit-identical. The N tile is M-keyed (`RADIANCE_MXFP4_TN4_MIN_M`, default
2048): the wide tile amortises A-tile staging for +10% at M=8192 but cannot fill below ~2048 rows.
`RADIANCE_MXFP4_W4A8_MIN_M` is where it takes over from aiter, and it now defaults to **0** -- our
kernel serves every M.

**There are two tilings, because prefill and decode are different problems.** The tile above
(BM=256 via TM=4) is sized for prefill. At decode M is 5 (batch 1 x `num_speculative_tokens`+1), where
it issues 51x more matrix MACs than useful -- 4352 WMMA per wave against 5 real rows. So M<=48 goes
to a second kernel with TM=`ceil(M/16)`, no wasted M-fragments, and split-K to fill the CUs, gated by
`RADIANCE_MXFP4_DECODE_MAX_M` (default 48). The split-K reduction is **fused**: the KS blocks covering
one output range race on an atomic counter and the last arrival reduces in place, so there is no
second launch — worth a further -3.9% of step time on top, at bit-identical output. It reverses one of the prefill answers: **BK=128 wins at
decode** (1.87x on gate_up) where it measured -34% at prefill, because that loss was purely the LDS
occupancy cliff and a 16-row A tile never reaches it.

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
