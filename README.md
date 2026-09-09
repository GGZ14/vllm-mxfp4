# vllm-radiance (MXFP4)

A vLLM inference server for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**, packaged as a
container image. It bundles a working ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4
patches and custom kernels needed to run vLLM on this card, plus RDNA4-tuned GEMM / attention /
all-reduce paths and a speculative draft controller.

Two commands after the clone get you a server. You do not build an image, and you do not edit
anything for a different card count.

```bash
git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 && cd radiance-vllm-mxfp4
./setup-mxfp4.sh      # host check, image pull, checkpoints, kernels (~40 GiB, mostly download)
./serve-mxfp4.sh      # serve on http://localhost:8080/v1
```

## Contents

- [Status](#status)
- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Setup](#setup)
- [Checkpoints](#checkpoints)
- [Running the server](#running-the-server)
- [ParoQuant (int4 W4A8)](#paroquant-int4-w4a8)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [How the MXFP4 path works](#how-the-mxfp4-path-works)
- [What's inside the image](#whats-inside-the-image)
- [Building the image from source](#building-the-image-from-source)
- [Repository layout](#repository-layout)
- [Further reading](#further-reading)

## Status

> **Early dev, experimental. Not production hardened. Use at your own risk.**

Repo version `0.12.0`; pinned image `stilldeadcode/vllm-radiance:0.9.3`.

| | |
|---|---|
| **Tested hardware** | 2 x Radeon AI PRO R9700 (gfx1201), tensor parallel |
| **Tested models** | Qwen3.8-27B-FP8, Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8, Gemma-4-31B-it-FP8, Qwen3.8-27B-Quark-AWQ-MXFP4, Qwen3.8-27B-PARO |
| **Tested KV dtypes** | fp8, bf16, `auto` |
| **Detected, not assumed** | GPU count, tensor-parallel size, KV cache size |
| **Untested** | other models, other weight formats, 1 or 4+ GPUs, non-R9700 hardware, TP=3 on a real three-card box |

This repository carries the MXFP4 work on top of
[vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), the source for the
`stilldeadcode/vllm-radiance` image on Docker Hub. The launcher pulls that published image and
applies this repo's patches and kernels at container start, so **running the MXFP4 stack never
requires building an image**.

## Quickstart

Run the commands at the top of this file. `setup-mxfp4.sh` is idempotent: re-run it any time
and it skips whatever is already done. Then test the server:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'
```

You get **Qwen3.8-27B in native 4-bit MXFP4** with an FP8 speculative drafter. On two R9700:

| | |
|---|--:|
| Weights | 9.4 GiB per GPU |
| KV cache | 943,581 tokens |
| Context length | 262,144 |
| Decode step | 22.7 ms |
| Aggregate throughput at 8 concurrent | 573 t/s |
| WikiText-2 perplexity | 8.3708 |
| GSM8K (500q, greedy) | 97.8% |

Full numbers in [Performance](#performance).

Useful next commands:

```bash
./serve-mxfp4.sh --help     # every knob, short form
./gpu-detect.sh             # what was detected and what it will do with it
DRY_RUN=1 ./serve-mxfp4.sh  # print the container command without running it
```

Any argument `serve-mxfp4.sh` does not recognise is passed straight through to `vllm serve`.

## Requirements

| | |
|---|---|
| **GPU** | An AMD RDNA4 (gfx1201) card. The image is compiled for gfx1201 only. One card works; four work. See [GPU and TP detection](#gpu-and-tp-detection) |
| **Host OS** | Linux with the amdgpu kernel driver, exposing `/dev/kfd` and `/dev/dri`. ROCm userspace lives inside the image |
| **Runtime** | podman (preferred and best exercised; the launcher uses `--replace` and `keep-groups`) or docker. Auto-detected |
| **Disk** | ~60 GiB for a full setup: 19 source + 19 built checkpoint + 2 drafter + ~10 image. The source download is deletable afterwards, and setup prints the command |
| **Host Python / ROCm / HF CLI** | Not needed. Setup runs everything that needs them inside the image |

## Setup

### What `setup-mxfp4.sh` does

| Step | What it does |
|---|---|
| 1. Host check | `/dev/kfd` + `/dev/dri`, at least one AMD GPU with 8 GiB+ VRAM, a container runtime, free disk |
| 2. Image | Pulls `stilldeadcode/vllm-radiance:0.9.3` |
| 3. Download | [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (~19 GiB), fetched with the image's own `huggingface_hub` |
| 4. Checkpoint rewrite | Rewrites it with an **fp8 MTP head** (`fp8_mtp.py`, ~15 min). AMD's release does not load as-is; see [Why AMD's checkpoint needs a rewrite](#why-amds-checkpoint-needs-a-rewrite). A checkpoint that already ships an fp8 MTP head skips this step |
| 5. Drafter | Downloads [`tcclaviger/Qwen3.8-27B-DFlash2-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) (2 GiB), the block-diffusion drafter. `--no-drafter` skips it; then serve with `SPEC_METHOD=mtp` |
| 6. Kernels | Builds the pinned **libr4d** inside the image (a few minutes, cached in `~/.cache/radiance-libr4d`). Load-bearing: the kernel shipped in the image predates the gated-delta-net overflow fix and NaNs this model's output. See [the GDN NaN](PERFORMANCE.md#the-gated-delta-net-nan-fixed-upstream) |

### First start is slow

The first `serve-mxfp4.sh` after setup spends several extra minutes compiling Triton and inductor
kernels before the engine comes up. It looks idle; it is compiling. The result is cached in
`~/.radiance-cache-w4a8-093-gdnm` and later starts skip it.

## Checkpoints

`setup-mxfp4.sh` downloads the defaults on its own. Every repo id is a `${VAR:-default}` override,
so pointing at a different one needs no edit.

| Checkpoint | Role | Notes |
|---|---|---|
| [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (~19 GiB) | Default target | AMD's Quark OCP micro-scaling release, Apache-2.0. Needs the MTP-head rewrite; setup does it. Override with `SRC_REPO=` |
| [`just1moremodel/Qwen3.8-27B-Uncensored-MXFP4-awq`](https://huggingface.co/just1moremodel/Qwen3.8-27B-Uncensored-MXFP4-awq) | Alternative target | Same architecture, abliterated. Ships the MTP head at `fp8_e4m3` already, so no rewrite. Opt-in: refusal behaviour has been removed |
| [`tcclaviger/Qwen3.8-27B-DFlash2-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) (2 GiB) | Drafter | Block-diffusion drafter for `SPEC_METHOD=dflash`, the default. `--no-drafter` skips it; then serve with `SPEC_METHOD=mtp`. Override with `DRAFT_REPO=` |

To serve the uncensored variant, download it under `$MODELS` and point the launcher at it. There is
no rewrite step and nothing else changes:

```bash
SNAP=$HOME/models/Qwen3.8-27B-Uncensored-MXFP4-awq ./serve-mxfp4.sh
```

*That one is verified from its `config.json`, which declares all 8 `mtp.*` modules `fp8_e4m3`
per-channel in `layer_quant_config`, exactly what `fp8_mtp.py` produces. It has not been load-tested
here; every number in this README is the AMD checkpoint.*

### Why AMD's checkpoint needs a rewrite

Its `mtp.*` layers are bf16, and its `exclude` list does name them, but as **tensor** names
(`mtp.fc.weight`, all 15 `.weight`-suffixed) among 112 **module** names
(`model.visual.blocks.0.attn.qkv`). Quark matches modules, so `mtp.fc.weight` never matches the
module `mtp.fc`, the exclusion silently does nothing, vLLM applies the mxfp4 scheme to a bf16 head,
and the load dies with:

```
Attempted to load weight (torch.Size([5120, 10240])) into parameter (torch.Size([5120, 5120]))
```

So the check for any other MXFP4 checkpoint is not "is `mtp` mentioned" but "is it mentioned the
same way as everything else": module names in `exclude`, or an entry in `layer_quant_config`.
`fp8_mtp.py` fixes the AMD one by requantizing the head to fp8 and declaring it properly.

## Running the server

### GPU and TP detection

`serve-mxfp4.sh` works out three things for itself at startup, and all three are overridable.

```bash
./gpu-detect.sh
```
```
AMD GPUs usable:  2 x Radeon AI PRO R9700 (0x7551, 32624 MiB each)
HIP indices:      0,1
skipped (<8192 MiB): 2:0x13c0:2048MiB
tensor parallel:  2   (supported: 8 4 2 1)
hardware sig:     2x7551-32624
KV cache pin:     18563072000 bytes (17.29 GiB/GPU, measured)
```

**Which GPUs.** `gpu-detect.sh` reads `mem_info_vram_total` from sysfs for every `amdgpu` render
node, with no `rocm-smi` on the host and no container start just to count cards. Anything below
`MIN_GPU_MIB` (8192) is excluded, which is what keeps an integrated GPU out of the count: the
development box reports three AMD render nodes (two R9700 and a 2 GiB Granite Ridge iGPU), and a
naive count picks `--tensor-parallel-size 3`.

**Tensor-parallel size.** The largest of `8 4 2 1` the usable cards can fill. `3`, `6` and `12` are
not picked automatically even though they divide `num_attention_heads=24`, because TP must also
divide the GDN layers' `linear_num_key_heads=16` and `linear_num_value_heads=48`. A three-card host
therefore serves on two by default, leaves one idle, and says so.

**KV cache size.** See [KV cache calibration](#kv-cache-calibration).

### TP=3 (explicit)

Three cards are served through zero-weight dummy heads. `radiance_tp3pad.py` widens the config to
36 q / 6 kv / 18 GDN-key / 54 GDN-value heads, MLP 17472 and vocab 248448, and pads the checkpoint
tensors at load with heads whose weights are exactly zero, so contiguous sharding puts every dummy
on rank 2 and the per-rank geometry (12 q / 2 kv, GQA 6) keeps the R4D attention kernels. Nothing
is rewritten on disk. Details in [TP3_PADDING_PLAN.md](TP3_PADDING_PLAN.md).

```bash
TP=3 ./serve-mxfp4.sh     # sets RADIANCE_TP_PAD=3, uses its own cache dir (-tp3pad)
./tp3pad_selftest.py      # check every padding rule against the safetensors headers, no GPU needed
./tp3pad_selftest.py --torch   # also pad one real tensor per rule, inside the image
```

`TP=3` also switches off two production defaults the padded widths cannot serve
(`RADIANCE_MXFP4_WPERM`, `RADIANCE_FP8_STREAM`), costing ~3-6% decode until the GDN merge gate is
relaxed. All-reduces at TP=3 ride RCCL until the 3-rank R4D kernel (`ar_oneshot_3rank_exact`,
libr4d extras rx6) has been measured on three cards; it stays out of the auto-pick list until it
passes that hardware gate. Every `TP != 3` serve passes `RADIANCE_TP_PAD=0` and is byte-identical
to before.

### KV cache calibration

An explicit `--kv-cache-memory` beats letting vLLM profile, because profiling is deliberately
conservative: it subtracts the profile run's *transient* activation peak plus the cudagraph
estimate, neither of which the steady state needs alongside a full cache. On the R9700 pair that
conservatism is 0.93 GiB per rank, **5.7% of the cache**: 892,799 KV tokens against 943,581.

That margin depends on the activation peak at `CHUNK`, on the cudagraph capture set `MAXSEQS`
produces, and on how the allocator fragments on that particular card, so it is measured rather than
computed. Measured rows live in [`kv-profiles.tsv`](kv-profiles.tsv), keyed on hardware **and**
batch shape:

```
# sig            maxseqs chunk  maxlen  spec    bytes
2x7551-32624     8       8192   262144  dflash  18563072000
```

A signature that is not in the table falls back to vLLM's own profiling. That is always safe; it
just leaves the margin unclaimed, and the launcher prints a one-line note saying so.

To claim it on your own hardware:

```bash
./calibrate-kv.sh              # ~15 min, needs the GPUs to itself
./calibrate-kv.sh --dry-run    # show the plan, run nothing
./calibrate-kv.sh --quick      # pass 1 only: pin what vLLM profiles, no search
```

Pass 1 profiles. Pass 2 raises the pin in 2% steps until the server stops coming up, then backs off
one step. A step only counts as passing if the server answers `/health` **and** completes a
`CHUNK`-sized prefill plus a decode; reaching `GPU KV cache size` is not enough, because the cache
is allocated before cudagraph capture and capture is where an over-committed pin actually dies.

The result is written to `~/.cache/radiance-mxfp4/kv-profiles.local.tsv`, which is read after the
shipped table and wins over it. Re-run after changing `MAXSEQS` or `CHUNK`: a pin is only valid at
the shape it was measured at.

### Serving something other than MXFP4

`docker-compose.yml` serves **Qwen3.8-27B-FP8** and is the path for the FP8 and Gemma checkpoints.
It is a plain `vllm serve` with no patch prelude, so it is compose-shaped rather than script-shaped.

```bash
# put your model at ./models/Qwen/Qwen3.8-27B-FP8  (or set MODELS=/your/model/dir)
docker compose up -d          # start; follow with: docker compose logs -f
docker compose down           # stop
```

All of its tunables are `${VAR:-default}`, so override them from the shell or a `.env` file without
editing it. With podman, `podman compose` takes the same file. The per-model notes (35B-A3B's
`--max-num-batched-tokens >= 2240`, Gemma-4-31B's template and drafter) are in
[DOCKERHUB.md](DOCKERHUB.md#tested-so-far).

## ParoQuant (int4 W4A8)

MXFP4 is not the only int4 format this stack serves. **ParoQuant** checkpoints
(`z-lab/Qwen3.8-27B-PARO`: int4 group-128 asymmetric, plus learned pairwise Givens rotations and
channel scaling on the activations) run on a W4A8 path built for gfx1201 — hand-written HIP
rotation and GEMM kernels, registered as a real vLLM quantization method through the public plugin
hook.

The reference ParoQuant implementation is CUDA-only and W4A16. This one is independent: the
rotation is a kernel, not a PyTorch fallback, and the asymmetric zero point is folded into the GEMM
epilogue as a row-sum correction rather than dequantized into the matmul. Weight-side traffic works
out at 4.25 bits/weight, the same as MXFP4.

```bash
./setup-paroquant.sh                      # host check, image, checkpoint, drafter, kernels
MODE=prod SPEC=7 ./paroquant/run_paroquant.sh
```

It shares the image, the DFlash2-FP8 drafter and libr4d with the MXFP4 setup, so running both costs
one download of each. Both stacks want both cards and port 8080, so only one serves at a time
(`vllm-switch paro` where the systemd units are installed).

Measured against MXFP4 production on 2 x R9700: GSM8K 500q **97.4-98.0%** (MXFP4 97.8), combined
decode **226 t/s** (MXFP4 186), conc-8 512 t/s, 24.19 ms/step, in-serve numerics gate rel = 0.00000
on every gated shape on both TP ranks (<= 4e-5 at wider M).

A second ParoQuant format keeps the learned rotations on **MXFP4 weights** (e2m1 + e8m0/32),
which puts the GEMM on the zero-VALU fp8-WMMA loop AMD's MXFP4 runs on -- `quant_method:
paroquant_mxfp4`, built by `paroquant/build_hybrid.py` from the bf16 base and z-lab's rotations,
served by `paroquant/radiance_paroquant_mxfp4.py`. In-serve CHECKALL with real inputs holds rel 0.0012-0.0021 (bf16 rounding) on
every shape and partition at TP=2; served-path GSM8K 500q **97.60%** (int4 PARO 97.60, AMD MXFP4 97.8);
prod decode at parity with int4 PARO (24.53 vs 24.19 ms/step) once the prologue, the stream producers
and the merged-linear GEMM each became one launch. See
[PAROQUANT.md](PAROQUANT.md#mxfp4-weights-the-zero-valu-loop).

The format, the kernels, the knob reference and the rejected experiments are in
[PAROQUANT.md](PAROQUANT.md).

## Configuration

Every default below is the measured production configuration, and all of them are `${VAR:-default}`,
so override from the environment; nothing needs editing. `./serve-mxfp4.sh --help` is the short
version of these tables, and `DRY_RUN=1 ./serve-mxfp4.sh` prints the container command a given set
of overrides produces without running it.

### Paths and runtime

| Variable | Default | What it does |
|---|---|---|
| `MODELS` | `~/models` | Checkpoint directory, bind-mounted at `/models`. Both checkpoints must live under it: it is the only mount |
| `SNAP` | `$MODELS/Qwen3.8-27B-MXFP4-mtpfp8` | The target checkpoint |
| `DRAFTER` | `$MODELS/Qwen3.8-27B-DFlash2-FP8` | The `dflash` drafter |
| `PORT` | `8080` | Listen port |
| `NAME` | `vllmmxfp4074` | Container name (historical; `podman logs -f <name>` uses it) |
| `IMAGE` | `stilldeadcode/vllm-radiance:0.9.3` | Container image. **Moves with `CACHE`** |
| `CACHE` | `~/.radiance-cache-w4a8-093` + suffixes | Compile cache. Keyed on model, torch/Triton version **and** every knob that changes the traced graph. Never share one across configurations |
| `RUNTIME` | auto | `podman` (preferred) or `docker` |
| `CHAT_TEMPLATE` | `./qwen-fixed-v22.3.jinja` | Mounted by path, so it must exist on the host |
| `HF_CACHE` | `~/.cache/huggingface` | Mounted for tokenizer files |
| `DRY_RUN` / `PREPARE_ONLY` | off | Print the command instead of running / do the one-time work and stop |

### Serving shape

| Variable | Default | What it does |
|---|---|---|
| `SPEC_METHOD` | `dflash` | `dflash` (block-diffusion drafter, one graphed pass, needs the second checkpoint) or `mtp` (the head inside the target, no extra download) |
| `SPEC` | `7` dflash / `4` mtp | Speculative depth. Under dflash it is content-dependent: 7 wins on a weighted mix (code/JSON run 4.7-6.0 tok/update), 5 wins on prose-heavy or batch-throughput serving. `RADIANCE_DYNAMIC_WIDTH` mostly dissolves the trade-off |
| `MAXSEQS` | `8` | Max concurrent sequences. Above 8 the decode band widens to `RADIANCE_MXFP4_DECODE_MAX_M=128` automatically and the `KV_MEM` lookup misses, so re-run `./calibrate-kv.sh` if you standardise on another value |
| `MAXLEN` | `262144` | Context length. Only lower it for diagnostics: the FLA GDN fallback allocates against this, not against the chunk size |
| `CHUNK` | `8192` | Prefill chunk. `RADIANCE_AR_MAX_KB` is derived from it, so raising it cannot silently drop prefill onto RCCL |
| `GPU_UTIL` | `0.98` | The ceiling on this box. Use `0.75` for perplexity work: `prompt_logprobs` allocates a 1-1.7 GiB transient vLLM does not reserve for |
| `KV_MEM` | `auto` | KV cache size. `auto` looks up a measured pin and falls back to profiling; `<bytes>` pins explicitly; `0` forces profiling. Consulted only at `GPU_UTIL=0.98`. Worth 892,799 -> 943,581 tokens on the R9700 pair. See [KV cache calibration](#kv-cache-calibration) |
| `TP` / `GPUS` | auto | Tensor-parallel size and the HIP indices to serve on. `TP=3` is explicit; see [TP=3](#tp3-explicit) |
| `RADIANCE_TP_PAD` | `3` at TP=3, else `0` | The dummy-head padding itself. `3` at TP=1/2 runs the validation gates; `_INTERMEDIATE=17408` keeps the MLP stock, `_DRAFTER=0` leaves the DFlash2 drafter unpadded, `_STRICT=0` demotes a weight-coverage mismatch to a warning |
| `MIN_GPU_MIB` | `8192` | VRAM floor for "usable". Lower it to admit a small card, raise it to skip one |
| `ASYNC` | `0` | Async scheduling. vLLM refuses it together with `disable_padded_drafter_batch`, so the two are one switch; the unpad lever is ~+50% single-stream under mtp |
| `EXTRA` | empty | Extra `vllm serve` flags (or just pass them as arguments) |

### Kernels

| Variable | Default | What it does |
|---|---|---|
| `R4D_ATTN` | `1` | The R4D paged attention backend. +37.8% prefill at 260k against AITER unified attention; `0` falls back to it |
| `AUTO_R4D` | `1` | Build the pinned libr4d on first run. `0` uses the image's, which **NaNs this model** |
| `R4D_SO` | unset | Use your own libr4d checkout directory instead; nothing is rebuilt behind your back |
| `R4D_PIN` | `b9e42ab` | Which libr4d commit to build. Each is cached separately and keyed by the string, so `R4D_PIN=main` is fetched once and reused (`rm -rf ~/.cache/radiance-libr4d/main` to refresh) |
| `MIN_M` | `0` | M above which the hand-written W4A8 kernel takes over from aiter. `0` means always: the comparison is `>`, so `1` would still send M=1 to aiter |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `64` (`128` if `MAXSEQS>8`) | The small-M decode GEMM band. Must cover `MAXSEQS x (SPEC+1)` rows or the biggest verify batches fall onto the prefill tile |
| `FAST_DRAFT` | `1` | The int2 draft head with an exact rerank: +6.5% decode under mtp, +5.1% under dflash |
| `RADIANCE_DRAFT_RERANK` | `80` dflash / `32` mtp | The candidate pool a top-k caller can draw from, not just a rescoring budget. Under dflash, 32 costs 5.3% of acceptance; 80 covers 4x the drafter's `selector_top_k` and 4x the sampler's `top_k` |
| `RADIANCE_VERIFY_HEAD` | `1` dflash / `0` mtp | The int2 head applied to the target's verify `lm_head` (one 2.02 ms GEMM per step, 5.9% of wall). +2.9% combined decode, output-equivalent |
| `RADIANCE_DYNAMIC_WIDTH` | `1` | Scheduler-side per-request verify width from an acceptance EMA. Recovers static `SPEC=5`'s batch efficiency at `SPEC=7` without losing code depth. Lossless by construction |
| `RADIANCE_GDN_MERGE_INPROJ` | `1` | GDN `in_proj_qkvz` + `in_proj_ba` as one GEMM (-2.9% decode). **Changes the traced graph**, so it keys the cache directory |
| `RADIANCE_SKINNY_GEMM` | `1` | R4D split-K for skinny bf16 projections. `all` adds shapes that differ from rocBLAS at a bf16 ULP |
| `RADIANCE_MXFP4_SANITIZE` | `0` | Zero non-finite activations. Only useful with `AUTO_R4D=0`, where it gets perplexity to 8.4004 instead of 653586 |

### Diagnostics

`RADIANCE_MXFP4_CHECKALL`, `_SHADOW`, `_KERNEL_NK`, `_PERBLOCK_NK`, `_MHIST`,
`RADIANCE_GDN_NANTRACE` and `PROFILE_DIR` are unset by default and documented where they are read in
`serve-mxfp4.sh`. The full image-level knob reference is in [DOCKERHUB.md](DOCKERHUB.md).

## Troubleshooting

### First, check three lines in the log

```
Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM     the W4A8 kernel won the selection
[radiance] native MXFP4 enabled on gfx12x              the aiter fp4 gate was relaxed
R4D selections table (RADIANCE_R4D_REPORT=1, on)       which kernels bound, and why not
```

A kernel that fails to bind **falls back silently** and costs performance rather than raising, so
read that table rather than assuming.

### Symptoms

The launcher preflights the host and fails with the command that fixes it, so most problems surface
as a one-line error rather than a traceback.

| Symptom | Cause and fix |
|---|---|
| `port 8080 is already in use` | Another server holds the port and, more importantly, the GPUs. Stop the container `podman ps` shows, or its systemd unit (`systemctl --user stop qwen_vllm_38` on the dev box). Or `PORT=8081 ./serve-mxfp4.sh` |
| `no checkpoint at .../Qwen3.8-27B-MXFP4-mtpfp8` | Run `./setup-mxfp4.sh`. AMD's release cannot be served directly; see [why](#why-amds-checkpoint-needs-a-rewrite) |
| `no dflash drafter at ...` | `./setup-mxfp4.sh` fetches it, or serve without it: `SPEC_METHOD=mtp ./serve-mxfp4.sh` |
| `chat template not readable` | `CHAT_TEMPLATE=<path>`; unset uses this repo's `qwen-fixed-v22.3.jinja`. It is mounted by path, so it must exist **on the host** |
| `/dev/kfd is missing` | The amdgpu kernel driver is not loaded. The image ships ROCm userspace, not the driver |
| `AssertionError: Attempted to load weight (torch.Size([5120, 10240]))` | You pointed it at AMD's raw checkpoint instead of the one `fp8_mtp.py` builds |
| Fluent but wrong output; perplexity in the hundreds of thousands | The stock libr4d NaNs the gated-delta-net. Confirm the launcher printed `[radiance] libr4d <pin> -> ...`; if you ran with `AUTO_R4D=0`, set `RADIANCE_MXFP4_SANITIZE=1` as a stopgap |
| `IndexError` in `rocm_unquantized_gemm_impl` at load | The int2 draft head against a libr4d that ships `r4d_gemm_w4a16_nt_m64`. Set `FAST_DRAFT=0` |
| Engine dies at startup on an `N=0` GEMM | A cache directory reused across a config that changes the traced graph. `rm -rf ~/.radiance-cache-w4a8-093*` and start again; `CACHE` and `IMAGE` must always move together |
| `running the draft eagerly` in the log | The drafter lost its CUDA graph, which is the whole point of `dflash`. Check `DRAFT_ATTN` supports full graphs (`TRITON_ATTN` does) |
| `current platform does not support native MXFP4/MXFP6` | **False alarm.** It comes from a separate `supports_mx()` call. The line that matters is `[radiance] native MXFP4 enabled on gfx12x` |
| Startup is slow and looks hung | First run compiles Triton/inductor kernels. Later runs reuse `$CACHE` |
| OOM at startup after changing `MAXSEQS`, `CHUNK` or a graph-changing knob | The KV pin (`KV_MEM`) was derived at `MAXSEQS=8`. `KV_MEM=0` re-enables vLLM's own profiling |

## Performance

Measured with BetterBench 0.4.0, corpus v1.0, `2026-08-30`, on the reference box: 2x R9700, TP2,
`dflash` `SPEC=7`, `GPU_UTIL=0.98`, KV pinned, temp 0.7 / top_p 0.95 / top_k 20, cold prefix cache
per request.

Two notes on reading these numbers:

- **Report step time, not tokens/s, for anything decode-related.** Tokens/s swings ~14% on
  draft-acceptance luck alone at a fixed config; `ms/step = 1000 x (accepted/draft + 1) / tok_s`
  divides that out.
- **Speculative decoding packs several tokens into one stream update**, so there is no per-token
  latency to quote. `update p50` is the wall-clock gap between updates, and `tok/update` is how many
  tokens land in each.

### Single stream

One pass per category, so treat each cell as a single observation rather than a distribution. The
`update p50` column is the exception worth trusting: it is the median of hundreds of update gaps
inside one run, and it lands within 0.2-0.5 ms of the `22.66 ms/step` figure the decode work was
gated on.

| Category | TTFT p50 (ms) | PP t/s | update p50 (ms) | tok/update | decode t/s |
|---|--:|--:|--:|--:|--:|
| code | 54.5 | 1229 | 22.9 | 5.77 | 253.8 |
| json | 54.7 | 1335 | 22.8 | 5.90 | 266.7 |
| math | 53.8 | 1319 | 22.9 | 5.81 | 256.0 |
| summarization | 59.0 | 1899 | 22.8 | 5.45 | 244.6 |
| file_edit | 59.1 | 1794 | 22.9 | 5.38 | 236.1 |
| reasoning | 56.4 | 1648 | 23.1 | 5.00 | 218.5 |
| chat | 59.1 | 1912 | 22.8 | 2.78 | 122.9 |
| prose | 36.5 | 1534 | 23.2 | 2.54 | 109.4 |

**Combined** (weighted code 0.3, reasoning 0.2, prose 0.15, json 0.15, file_edit 0.1,
summarization 0.1): decode **224.3 t/s**, update p99 **23.4 ms**, TTFT p50 **53 ms**.

The spread across categories is almost entirely `tok/update`, not step time. Every category holds
22.8-23.2 ms per update, and `code` reaches 253.8 t/s against `prose`'s 109.4 purely because code
drafts accept 5.77 tokens per update where prose accepts 2.54. That is why the depth default is
content-dependent; see `SPEC` in [Serving shape](#serving-shape).

### Concurrency

48 requests per level, `MAXSEQS=8`.

| Concurrent | Aggregate t/s | TTFT p50 (ms) | Per-request decode t/s |
|--:|--:|--:|--:|
| 1 | 175.4 | 55.5 | 223.0 |
| 2 | 302.6 | 83.8 | 202.8 |
| 4 | 442.7 | 93.0 | 153.2 |
| 8 | **572.8** | 115.4 | 104.2 |
| 16 | 565.5 | 4430.2 | 101.2 |

Aggregate throughput peaks at 8 and does not improve at 16, because `MAXSEQS` is 8: the extra
requests queue, which is what the 4430 ms TTFT is. Serving 16 concurrent usefully means
`MAXSEQS=16`, which also widens the decode band to `RADIANCE_MXFP4_DECODE_MAX_M=128`; measured
there, conc-16 reaches 549-622 t/s at 75-79 ms steps. That path re-profiles KV rather than using the
pin, so run `./calibrate-kv.sh` if you standardise on it.

### Prefill

8 runs per depth, cold prefix cache.

| Target depth | Prompt tokens | TTFT p50 (ms) | PP t/s |
|--:|--:|--:|--:|
| 2k | 1,514 | 314.7 | 4810 |
| 8k | 5,918 | 1,239.5 | 4773 |
| 16k | 11,794 | 2,483.1 | 4749 |
| 32k | 23,543 | 5,134.7 | 4585 |
| 64k | 47,056 | 10,866.2 | 4330 |

This sweep stops at 64k. For deeper context the reference points are the fp8-attention gate
(3831 t/s at 106k) and the 0.5.8 -> 0.7.4 table, both in
[PERFORMANCE.md](PERFORMANCE.md); they run on a different harness and are not directly comparable to
this table. Prefill throughput keeps falling with depth in both.

### Quality and capacity

| | |
|---|--:|
| WikiText-2 perplexity (208,539 tokens) | 8.3708 |
| WikiText-2 top-1 | 54.13% |
| GSM8K 500q greedy | 97.8% (97.0-98.0 across the whole optimization stack) |
| Weights | 9.4 GiB/GPU |
| KV cache | 943,581 tokens |
| Concurrency at 262,144 tokens per request | 3.60x |

**The change-by-change ledger** (what each landed optimization was worth and how it was gated), the
0.5.8 -> 0.7.4 provenance A/B, and the gated-delta-net NaN write-up all live in
[PERFORMANCE.md](PERFORMANCE.md).

## How the MXFP4 path works

### Native MXFP4

Quark OCP micro-scaling checkpoints (`quantization_config.quant_method: quark`, mxfp4 weights *and*
activations, group 32, e8m0 scales) run natively with `RADIANCE_MXFP4=1`. Drop `--quantization`: the
runtime reads the method from `config.json` and routes it itself.

Without this, vLLM falls back to emulated MXFP4, which materialises every weight tensor in bf16 on
each forward. Nothing in the way was a compiler limitation, since Triton 3.6 does lower
`tl.dot_scaled` on gfx1201. It was three soft gates, all handled in `patch_quark_mxfp4.py`: an
`is_fp4_avail()` allowlist that omits gfx1201, an aiter module path that moved in 0.1.17, and
gfx1250 tiles that ask for `matrix_instr_nonkdim=32` when this card's WMMA is 16x16x16 only
(`mxfp4-configs/` pins 16 across every band).

The native path is **bit-identical to emulation** (the activation quantization is the same either
way), so it is a speed change with no quality dimension. Measured on gate_up 17408x5120: **6.1x at
M=16, 4.7x at M=32, 2.5x at M=64**.

The stock emulated path cannot serve these checkpoints on this card at all, which makes the native
kernel the only way to run them here, not merely the faster one. (Quark's TileLang backend cannot
initialise inside a vLLM worker, and the retired `RADIANCE_MXFP4_MAX_M` fallback was specialised
into the compile graph during the profile run, so it killed startup rather than one request.)

### The W4A8 fp8-WMMA kernel

`RADIANCE_MXFP4_W4A8=1` additionally routes linears to a hand-written fp8-WMMA HIP kernel
(`radiance_mxfp4_fp8.hip`). Triton lowers `tl.dot_scaled` by upconverting e2m1 to bf16 and using the
16-bit WMMA; register-resident on this card, **fp8 WMMA runs 325 TFLOP/s against f16's 160**, while
Triton's own fp8 `tl.dot` manages 43 because it upconverts and pays conversion on top. Against the
tuned aiter path it measures **1.47-2.26x faster and 4.2x more accurate** (0.0265 vs 0.1119 relative
error), since fp8 activations beat the mxfp4 ones aiter quantizes to.

It is **off in the image** because it changes numerics (the layer becomes W4A8 rather than the
checkpoint's declared W4A4: more precise than what the model was calibrated against, but no longer
bit-identical), and **on in `serve-mxfp4.sh`**, which is what every number above was measured with.

**There are two tilings, because prefill and decode are different problems.**

| | Prefill tile | Decode tile |
|---|---|---|
| Shape | BM=256 via TM=4 | TM=`ceil(M/16)`, no wasted M-fragments |
| Gate | default above `RADIANCE_MXFP4_DECODE_MAX_M` | `M <= RADIANCE_MXFP4_DECODE_MAX_M` (64, or 128 at `MAXSEQS>8`) |
| K block | BK=64 (BK=128 measured -34%) | BK=128 (1.87x on gate_up) |
| Split-K | no | yes, with a fused reduction |
| N tile | M-keyed at `RADIANCE_MXFP4_TN4_MIN_M` (2048): the wide tile amortises A-tile staging for +10% at M=8192 but cannot fill below ~2048 rows | fixed |

At decode M is 5 (batch 1 x `num_speculative_tokens`+1), where the prefill tile issues 51x more
matrix MACs than useful: 4352 WMMA per wave against 5 real rows. The decode split-K reduction is
fused, so the KS blocks covering one output range race on an atomic counter and the last arrival
reduces in place with no second launch, worth a further -3.9% of step time at bit-identical output.
BK=128 winning at decode reverses the prefill answer because that -34% was purely the LDS occupancy
cliff, which a 16-row A tile never reaches.

`RADIANCE_MXFP4_DECODE_MAX_M` must cover `MAXSEQS x (SPEC+1)` rows or the biggest verify batches
fall back onto the prefill tile. `RADIANCE_MXFP4_W4A8_MIN_M` defaults to **0**, so our kernel serves
every M.

**4-bit weights leave far more room for KV.** On 2x R9700 the 27B MXFP4 body occupies 9.24 GiB/GPU
against roughly 12.6 for the same model in FP8, and that headroom goes straight into context.

### Why the drafter is FP8, not MXFP4

The target checkpoint is `Qwen3.8-27B-MXFP4-mtpfp8`: AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` body with
the MTP drafter requantized to fp8 by [`fp8_mtp.py`](fp8_mtp.py).

Four-bit costs more acceptance than it saves in bandwidth, and AWQ does not rescue it. Data-free RTN
measured ~11.6% relative error and cost acceptance 2.5 -> 2.21; AWQ calibration improved that by
0-5% (the alpha search chose 0.1-0.2, and 0.0 for `mtp.fc`, because MXFP4's per-32 E8M0 block
exponent already does most of what per-channel scaling would). FP8 e4m3 per-channel is ~2-3%
relative error and holds acceptance at 2.60-2.80, removing ~17% of decode weight traffic instead of
25%: the smaller win that actually holds.

[MXFP4-NOTES.md](MXFP4-NOTES.md) collects the longer form of this reasoning, and
[`serve-mxfp4.sh`](serve-mxfp4.sh) is the worked launch, with the measurement that chose each
default.

## What's inside the image

Everything below is baked into the image, and the tuned paths are env-gated and on by default. See
[DOCKERHUB.md](DOCKERHUB.md) for the per-knob reference: every flag, its default, and what it does.

| | |
|---|---|
| **gfx1201 correctness patches** (always on) | GPU enumeration, AITER enablement, native sampler fallback, MTP drafter unpad + multimodal draft-mask alignment, tool-parser and `from_json` chat-template filter, and an attention LDS fit that shrinks the staged K/V tile into the R9700's 64 KiB shared memory for any head size and KV dtype |
| **RDNA4-tuned kernels** | Preshuffled FP8 blockscale GEMM, unified-attention tiling (fp8 + bf16/`auto` KV, plus a head-size-keyed long-context prefill config), fused RMSNorm+quant, an fp16 matrix-core (WMMA) gated-delta-net path, a widened channel block for the GDN prefill convolution (2.2x on that kernel, bit-identical), a TP=2 P2P one-shot all-reduce, and a native head_dim-72 ViT flash kernel |
| **R4D attention** (`--attention-backend R4D`) | Purpose-built HIP attention kernels. They compute the score matrix transposed (`S^T = K.Q^T`) so a wave32 matrix-core fragment hands each lane exactly one query row and the softmax never leaves the lane, at a *smaller* error against an fp32 reference than the kernel it replaces. Measured in the serve: **+14.6% prefill at 64K** (the kernel itself is 1.65x), +4.1% at 16K, decode unchanged within noise (attention is only ~7% of a speculative decode step). Needs head_dim 256, paged block 16, GQA 6, causal attention, bf16/fp8 KV; any other shape is refused at startup with the reason |
| **Fine-grained MoE support** | RDNA4-tuned fused-MoE Triton configs (always on; removes the stock config's `M>=96` cliff, lossless), plus the skinny bf16 GEMM on the MoE gate. Inert on models they do not apply to |
| **Skinny bf16 GEMM** (`RADIANCE_SKINNY_GEMM`) | Projections too small for rocBLAS to fill the machine go to the R4D split-K kernel for `M` in `[6,64]`. `all` adds shapes that differ from rocBLAS at a bf16 ULP, most importantly GDN `in_proj_ba`, 480 KiB run 48 times per step: 28.5us against 3.6us |
| **Native MXFP4** (`RADIANCE_MXFP4`) | Bit-identical to vLLM's emulation and multiples faster, plus the optional hand-written fp8-WMMA W4A8 GEMM (`RADIANCE_MXFP4_W4A8`). See [How the MXFP4 path works](#how-the-mxfp4-path-works) |
| **Lossless dynamic MTP drafting** | A per-request confidence gate plus verbatim n-gram tail that varies draft depth without changing what the model verifies. `mtp` only: it stops a serial loop of draft forwards early, and a `dflash` drafter has no such loop |
| **The tuned drafter stack** (`RADIANCE_FAST_DRAFT`) | The draft head at 2 bits with an exact rerank (any drafter), plus a `dflash` drafter's decoder projections packed to signed symmetric int4 (one f16 scale per 128 input channels, no zero point, 4.25 bits per weight) on two purpose-built gfx1201 kernels (f16 matrix-core below 16 rows, int8 above). Codes are derived at load, so there is no calibration data and nothing on disk. Draft pass -9.1% at a drafter batch of 64; decode step -5.1% for +3.5% tokens/s |
| **Prefix caching on the GDN hybrid** | Hybrid models leave APC off by default, so the compose turns it on with `--enable-prefix-caching --mamba-cache-mode=align`. Align mode snapshots and restores the GDN recurrent state at block boundaries (verified bit-identical to full recompute, including under MTP), giving a large TTFT drop on shared prefixes |
| **Startup topology + bandwidth sweep** (`RADIANCE_RUN_BWTEST`) | Device list, P2P access matrix, NUMA distances, peak copy bandwidth per agent pair. Backgrounded, about a second. Set `0` to skip |
| **Optional NUMA pinning** (`--numa-bind`) | Off by default; for multi-NUMA-node hosts |

## Building the image from source

You do not need any of this to serve. `setup-mxfp4.sh` pulls the published image, and the MXFP4
patches and kernels are applied at container start. Build from source only to change a pinned
component or a baked-in patch.

Everything the build needs is in this directory (a flat Docker build context). The version string
lives in one place, the `VERSION` file, which the tag and the build-arg both read (`podman build`
takes the same arguments):

```bash
docker build -t vllm-radiance:$(cat VERSION) --build-arg RADIANCE_VERSION=$(cat VERSION) .
```

That single command builds everything from source, in four stages:

| Stage | What it does |
|---|---|
| **builder** | Compiles PyTorch, Triton, torchvision, AITER and vLLM for `PYTORCH_ROCM_ARCH=gfx1201` against the official digest-pinned `rocm/dev-ubuntu-24.04` base, leaving wheels in `/wheels`. Also builds `rocm-bandwidth-test` for the startup sweep |
| **rocmprune** | `prune_rocm.sh` cuts the 19 GB ROCm tree down to this one GPU architecture |
| **assemble** | Installs the wheels, applies the RDNA4 correctness patches, and clones and compiles [libr4d](https://codeberg.org/StillDeadcode/libr4d) with the image's own `hipcc` |
| **final** | A clean `ubuntu:24.04` that receives only the pruned ROCm tree, the venv and the entrypoint, so neither the build toolchain nor the wheels reach the published image |

Expect it to run for hours on a many-core box; it is a full PyTorch compile.

**Why the prune matters.** The stock ROCm base is 7.4 GiB compressed on its own, most of it device
code for GPUs this image cannot run on. Pruning to gfx1201 and shipping an allowlist takes the image
from 9.35 GiB compressed to **3.66 GiB**. The prune must happen in a stage the release stage copies
*from*: deleting files in a layer stacked on the base reclaims nothing.

**The release stage still ships a working compiler** (hipcc, g++, and the C++/Python headers). That
is not slack to trim: AITER JIT-compiles its kernels on first use inside the running container, so
an image without those headers boots and then dies on the first AITER module build. The build
asserts it by compiling and importing a pybind11 HIP module.

**Component pins** are the `ARG`s at the top of the `Dockerfile` (`TORCH_VERSION`,
`TRITON_VERSION`, `TORCHVISION_VERSION`, `AITER_VERSION`, `VLLM_VERSION`). Each is both the git tag
that gets compiled and the version the resulting wheel reports, and the build asserts the two agree,
so `pip show` and the startup banner can be trusted.

> **Do not bump torch / triton / torchvision on their own.** They are not independent choices: vLLM
> pins the torch version it is tested against, torch pins its triton, and torchvision ships a
> matching release. The build runs vLLM's own `use_existing_torch.py`, which *strips* those pins,
> but that exists so pip does not re-download torch, not as licence to install a newer one. Builds
> 0.5.0 through 0.5.4 compiled against a newer trio and hung a GPU under sustained tensor-parallel
> load; restoring the pinned versions fixed it with no code change. If you override these with
> `--build-arg`, move them together and soak-test under real load.

If you just want a known-good image without building: `docker pull stilldeadcode/vllm-radiance`.

## Repository layout

Flat build context. The runtime Python modules (`radiance_*.py`), the `patch_*.py` fixes, the
`fp8-configs/` `moe-configs/` and `mxfp4-configs/` GEMM configs, the chat templates, `Dockerfile`
and `docker-compose.yml` all live at the repo root so `docker build .` works directly.
`prune_rocm.sh` is the ROCm slimming step, and it self-checks: the arch's own kernels must survive
and hipcc must still link a HIP shared object, since AITER JITs at runtime.

### MXFP4 entry points

| File | What it is |
|---|---|
| `setup-mxfp4.sh` | One-time setup: host check, image, checkpoints, kernels. Idempotent |
| `serve-mxfp4.sh` | The launcher. `--help` for the knobs, `DRY_RUN=1` to see the command it builds |
| `gpu-detect.sh` | GPU/TP/KV detection, sourced by the launcher. Run it directly to see what it finds |
| `calibrate-kv.sh` | Measures a `--kv-cache-memory` pin for your hardware and saves it |
| `kv-profiles.tsv` | Pins measured so far, keyed on hardware and batch shape |
| `fp8_mtp.py` | Builds the loadable checkpoint from AMD's release (setup drives it) |
| `tp3pad_selftest.py` | Checks the TP=3 padding rules against the safetensors headers, no GPU needed |
| `run_mxfp4_074.sh` | Compatibility shim: the launcher's old name, forwards to `serve-mxfp4.sh` |
| `run_mxfp4_minm.sh` | The 0.5.8 launch, frozen. The only way to reproduce the baseline the numbers here are measured against |

### ParoQuant entry points

| File | What it is |
|---|---|
| `setup-paroquant.sh` | One-time setup: host check, image, PARO checkpoint, drafter, kernels. Idempotent |
| `paroquant/run_paroquant.sh` | The launcher. `MODE=eval` gates numerics, `MODE=prod` serves |
| `paroquant/radiance_paroquant.py` | The `paroquant` quant method: checkpoint loading, layout, dispatch |
| `paroquant/radiance_paroquant.hip` | Kernel module. Compiled in-container at launch |
| `paroquant/par_kernels.h` | The rotation and W4A8 GEMM device code |
| `paroquant/par_harness.hip` | Standalone gates for the kernels, no server needed (`run.sh`, `run2.sh`) |
| `paroquant/radiance_paroquant_mxfp4.py` | The `paroquant_mxfp4` quant method: MXFP4 weights + rotations on the W4A8 MXFP4 GEMM, per-token stream producers |
| `paroquant/test_mxfp4_loader.py`, `bench_linear_tp2.py` | GPU unit test (fp32 reference, stream equivalence, single-launch equivalence) and the int4-vs-MXFP4 per-shape decode microbench |
| `paroquant/RESULTS.md` | The change-by-change engineering log |

### Where the HIP kernels live

The HIP kernels are not in this repo. They live in
[libr4d](https://codeberg.org/StillDeadcode/libr4d), a library of kernels for gfx1201 rather than
for any one model, pinned by tag (`R4D_VERSION` in the `Dockerfile`), which the build asserts
against the version the compiled library reports. `make r4d` clones and builds that pinned tag here
for development, so a locally built `r4d.so` matches the one in the image.

R4D entry points are named for the geometry they are compiled for
(`attn_decode_h256_gqa6_fp8kv`, `gdn_chunk_scan_k128_v128_c64_bf16`, `ar_oneshot_2rank_exact`) and
reject a mismatch rather than running, so which kernels an engine binds is visible in the startup
log and in `r4d.kernels()`.

One HIP kernel does stay here: `radiance_mxfp4_fp8.hip`, the MXFP4 W4A8 fp8-WMMA GEMM. It is
specific to this fork rather than general to gfx1201, so the image build compiles it directly and
`make radiance_mxfp4_fp8.so` rebuilds it against the image toolchain during development.

## Further reading

| Document | What's in it |
|---|---|
| [DOCKERHUB.md](DOCKERHUB.md) | The image description, the complete environment-variable / knob reference, and stack versions |
| [PAROQUANT.md](PAROQUANT.md) | The ParoQuant W4A8 path: format, rotation kernel, GEMM, knobs, and the experiments that were rejected |
| [PERFORMANCE.md](PERFORMANCE.md) | The change-by-change optimization ledger, the 0.5.8 -> 0.7.4 provenance A/B, and the gated-delta-net NaN write-up |
| [MXFP4-NOTES.md](MXFP4-NOTES.md) | Design notes, measurements and traps behind `serve-mxfp4.sh` |
| [TP3_PADDING_PLAN.md](TP3_PADDING_PLAN.md) | The TP=3 dummy-head padding design and its validation gates |
