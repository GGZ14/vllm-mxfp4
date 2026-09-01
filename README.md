# vllm-radiance

A vLLM inference server image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)**. It bundles a working
ROCm + PyTorch + Triton + AITER + vLLM stack with the RDNA4 patches and custom kernels needed to run vLLM on
this card, plus RDNA4-tuned GEMM / attention / all-reduce paths and a dynamic MTP draft controller, so you
don't have to build the stack yourself.

> **Status: early dev, experimental.** Repo version `0.10.0`; the pinned image is
> `stilldeadcode/vllm-radiance:0.9.3`. Everything here was built and measured on a few exact setups:
> **Qwen3.8-27B-FP8** and **Qwen3.6-27B-FP8** (gated-delta-net hybrids, architecturally identical),
> **Qwen3.6-35B-A3B-FP8** (fine-grained MoE, 256 experts / top-8),
> **Gemma-4-31B-it-FP8** (block-fp8, sliding + global attention, vision), and
> **Qwen3.8-27B-Quark-AWQ-MXFP4** (4-bit OCP micro-scaling), all with fp8 (or bf16/`auto`) KV cache on two
> R9700 GPUs (tensor parallel). Other models, other weight formats, single or 3+ GPUs, and non-R9700
> hardware are untested. Expect rough edges and breaking changes. Not production hardened. Use at your own
> risk.

This repository carries the MXFP4 work on top of
[vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), which is the source for the image
published as `stilldeadcode/vllm-radiance` on Docker Hub. The launcher pulls that published image and
applies this repo's patches and kernels at container start, so **running the MXFP4 stack never requires
building an image**. See **[DOCKERHUB.md](DOCKERHUB.md)** for the image description, the complete
environment-variable / knob reference, and stack versions.

---

## Quickstart

Two commands. `setup-mxfp4.sh` is idempotent -- re-run it any time; it skips whatever is already done.

```bash
git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 && cd radiance-vllm-mxfp4
./setup-mxfp4.sh      # host check, image pull, checkpoints, kernels (~40 GiB, mostly download)
./serve-mxfp4.sh      # serve on http://localhost:8080/v1
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'
```

That serves **Qwen3.8-27B in native 4-bit MXFP4** with an FP8 speculative drafter: 9.4 GiB of weights per
GPU, ~940k tokens of KV cache, 260K context. `./serve-mxfp4.sh --help` lists every knob, and any
argument it does not recognise is passed straight through to `vllm serve`.

**What setup does, and why each step exists.** Nothing here is an optimization you can skip:

| | |
|---|---|
| 1. host | `/dev/kfd` + `/dev/dri`, two AMD GPUs, a container runtime (podman or docker, auto-detected), disk |
| 2. image | pulls `stilldeadcode/vllm-radiance:0.9.3` |
| 3. download | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (~19 GiB), fetched with the image's own `huggingface_hub`, so the host needs no Python |
| 4. checkpoint | rewrites it with an **fp8 MTP head** (`fp8_mtp.py`, ~15 min). AMD's release does not load as-is: its `mtp.*` layers are bf16 but named in neither `exclude` nor `layer_quant_config`, so vLLM applies the mxfp4 scheme to them and dies with `Attempted to load weight (torch.Size([5120, 10240])) into parameter (torch.Size([5120, 5120]))` |
| 5. drafter | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (2 GiB), the block-diffusion drafter. `--no-drafter` skips it; then serve with `SPEC_METHOD=mtp` |
| 6. kernels | builds the pinned **libr4d** inside the image (a few minutes, cached in `~/.cache/radiance-libr4d`). The kernel *shipped in the image* predates the gated-delta-net overflow fix and NaNs this model's output -- WikiText-2 perplexity 653586 against 8.3706 -- so this build is load-bearing. See [the GDN NaN](#the-gated-delta-net-nan-fixed-upstream) |

The first `serve-mxfp4.sh` after setup spends several extra minutes compiling Triton and inductor kernels
before the engine comes up. It looks idle; it is compiling. That result is cached in
`~/.radiance-cache-w4a8-093-gdnm` and later starts skip it.

### Requirements

- **AMD Radeon AI PRO R9700 (gfx1201), two of them.** The image is compiled for gfx1201 only and this
  configuration serves tensor-parallel across two cards. It will not start with one.
- **Linux host with the amdgpu kernel driver**, exposing `/dev/kfd` and `/dev/dri`. ROCm userspace lives
  inside the image; nothing but the driver is needed on the host.
- **podman or docker.** podman is what this is developed against and is preferred (`--replace`,
  `keep-groups`); docker is handled by the launcher but is less exercised.
- **~60 GiB free disk** for a full setup: 19 source + 19 built checkpoint + 2 drafter + ~10 image. The
  source download can be deleted once the checkpoint is built; setup prints the command.
- No Python, no ROCm, no HF CLI on the host. Setup runs everything that needs them inside the image.

### Serving something other than MXFP4

`docker-compose.yml` serves **Qwen3.8-27B-FP8** and is the path for the FP8 and Gemma checkpoints; it is a
plain `vllm serve` with no patch prelude, so it is compose-shaped rather than script-shaped.

```bash
# put your model at ./models/Qwen/Qwen3.8-27B-FP8  (or set MODELS=/your/model/dir)
docker compose up -d          # start; follow with: docker compose logs -f
docker compose down           # stop
```

All of its tunables are `${VAR:-default}`, so override them from the shell or a `.env` file without editing
it. With podman, `podman compose` takes the same file. The per-model notes (35B-A3B's
`--max-num-batched-tokens >= 2240`, Gemma-4-31B's template and drafter) are in
**[DOCKERHUB.md](DOCKERHUB.md#tested-so-far)**.

---

## Troubleshooting

The launcher preflights the host and fails with the command that fixes it, so most of these surface as a
one-line error rather than a traceback. The rest are what the log looks like when something is wrong.

| Symptom | Cause and fix |
|---|---|
| `port 8080 is already in use` | Another server holds the port and, more importantly, the GPUs. Stop the container `podman ps` shows, or its systemd unit if it has one (`systemctl --user stop qwen_vllm_38` on the dev box). Or `PORT=8081 ./serve-mxfp4.sh` |
| `no checkpoint at .../Qwen3.8-27B-MXFP4-mtpfp8` | Run `./setup-mxfp4.sh`. AMD's release cannot be served directly -- see step 4 above |
| `no dflash drafter at ...` | `./setup-mxfp4.sh` fetches it, or serve without it: `SPEC_METHOD=mtp ./serve-mxfp4.sh` |
| `chat template not readable` | `CHAT_TEMPLATE=<path>`; unset uses this repo's `qwen3.8-enhanced.jinja`. It is mounted by path, so it must exist **on the host** |
| `/dev/kfd is missing` | The amdgpu kernel driver is not loaded. The image ships ROCm userspace, not the driver |
| `AssertionError: Attempted to load weight (torch.Size([5120, 10240]))` | You pointed it at AMD's raw checkpoint instead of the one `fp8_mtp.py` builds |
| Fluent but wrong output; perplexity in the hundreds of thousands | The stock libr4d NaNs the gated-delta-net. Confirm the launcher printed `[radiance] libr4d <pin> -> ...`; if you ran with `AUTO_R4D=0`, set `RADIANCE_MXFP4_SANITIZE=1` as a stopgap |
| `IndexError` in `rocm_unquantized_gemm_impl` at load | The int2 draft head against a libr4d that ships `r4d_gemm_w4a16_nt_m64`. `FAST_DRAFT=0` |
| Engine dies at startup on an `N=0` GEMM | A cache directory reused across a config that changes the traced graph. `rm -rf ~/.radiance-cache-w4a8-093*` and start again -- `CACHE` and `IMAGE` must always move together |
| `running the draft eagerly` in the log | The drafter lost its CUDA graph, which is the whole point of `dflash`. Check `DRAFT_ATTN` supports full graphs (`TRITON_ATTN` does) |
| `current platform does not support native MXFP4/MXFP6` | **False alarm.** It comes from a separate `supports_mx()` call. The line that matters is `[radiance] native MXFP4 enabled on gfx12x` |
| Startup is slow and looks hung | First run compiles Triton/inductor kernels. Later runs reuse `$CACHE` |
| OOM at startup after changing `MAXSEQS`, `CHUNK` or a graph-changing knob | The KV pin (`KV_MEM`) was derived at `MAXSEQS=8`. `KV_MEM=0` re-enables vLLM's own profiling |

Three lines in the log say the fast paths actually bound:

```
Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM     the W4A8 kernel won the selection
[radiance] native MXFP4 enabled on gfx12x              the aiter fp4 gate was relaxed
R4D selections table (RADIANCE_R4D_REPORT=1, on)       which kernels bound, and why not
```

A kernel that fails to bind **falls back silently** and costs performance rather than raising, so read that
table rather than assuming.

---

## Knobs

Every default below is the measured production configuration. They are all `${VAR:-default}`, so override
from the environment; nothing needs editing. `./serve-mxfp4.sh --help` is the short version of this table,
and `DRY_RUN=1 ./serve-mxfp4.sh` prints the container command a given set of overrides produces without
running it.

### Where and what

| | default | |
|---|---|---|
| `MODELS` | `~/models` | checkpoint directory, bind-mounted at `/models`. Both checkpoints must live under it -- it is the only mount |
| `SNAP` | `$MODELS/Qwen3.8-27B-MXFP4-mtpfp8` | the target checkpoint |
| `DRAFTER` | `$MODELS/Qwen3.8-27B-DFlash2-FP8` | the `dflash` drafter |
| `PORT` | `8080` | listen port |
| `NAME` | `vllmmxfp4074` | container name (historical; `podman logs -f <name>` uses it) |
| `IMAGE` | `stilldeadcode/vllm-radiance:0.9.3` | **moves with `CACHE`** |
| `CACHE` | `~/.radiance-cache-w4a8-093` + suffixes | compile cache. Keyed on model + torch/Triton version **and** on every knob that changes the traced graph; never share one across configurations |
| `RUNTIME` | auto | `podman` (preferred) or `docker` |
| `CHAT_TEMPLATE` | `./qwen3.8-enhanced.jinja` | must exist on the host; mounted by path |
| `HF_CACHE` | `~/.cache/huggingface` | mounted for tokenizer files |
| `DRY_RUN` / `PREPARE_ONLY` | off | print the command instead of running / do the one-time work and stop |

### Serving shape

| | default | |
|---|---|---|
| `SPEC_METHOD` | `dflash` | `dflash` (block-diffusion drafter, one graphed pass, needs the second checkpoint) or `mtp` (the head inside the target, no extra download) |
| `SPEC` | `7` dflash / `4` mtp | speculative depth. Under dflash it is **content-dependent**: 7 wins on a weighted mix (code/JSON run 4.7-6.0 tok/update), 5 wins on prose-heavy or batch-throughput serving. `RADIANCE_DYNAMIC_WIDTH` mostly dissolves the trade |
| `MAXSEQS` | `8` | max concurrent sequences. Above 8 the decode band widens to `RADIANCE_MXFP4_DECODE_MAX_M=128` automatically, and the `KV_MEM` pin no longer applies |
| `MAXLEN` | `262144` | context length. Only lower it for diagnostics -- the FLA GDN fallback allocates against this, not the chunk size |
| `CHUNK` | `8192` | prefill chunk. `RADIANCE_AR_MAX_KB` is derived from it, so raising it cannot silently drop prefill onto RCCL |
| `GPU_UTIL` | `0.98` | the ceiling on this box. Use `0.75` for perplexity work: `prompt_logprobs` allocates a 1-1.7 GiB transient vLLM does not reserve for |
| `KV_MEM` | `18563072000` | explicit KV cache size, which overrides `GPU_UTIL` and skips profiling. Worth 892,799 -> 943,581 tokens. Applied only at `GPU_UTIL=0.98` and `MAXSEQS=8`; `KV_MEM=0` re-enables profiling. Re-derive it after anything that moves weights, cudagraph sizes or `CHUNK` |
| `ASYNC` | `0` | async scheduling. vLLM refuses it together with `disable_padded_drafter_batch`, so the two are one switch; the unpad lever is ~+50% single-stream under mtp |
| `EXTRA` | empty | extra `vllm serve` flags (or just pass them as arguments) |

### Kernels

| | default | |
|---|---|---|
| `R4D_ATTN` | `1` | the R4D paged attention backend. +37.8% prefill at 260k against AITER unified attention; `0` falls back to it |
| `AUTO_R4D` | `1` | build the pinned libr4d on first run. `0` uses the image's, which **NaNs this model** |
| `R4D_SO` | unset | use your own libr4d checkout directory instead; nothing is rebuilt behind your back |
| `R4D_PIN` | `b9e42ab` | which libr4d commit to build. Each is cached separately, and the cache is keyed by the string, so `R4D_PIN=main` is fetched once and reused (`rm -rf ~/.cache/radiance-libr4d/main` to refresh) |
| `MIN_M` | `0` | M above which the hand-written W4A8 kernel takes over from aiter. `0` means always -- the comparison is `>`, so `1` would still send M=1 to aiter |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `64` (`128` if `MAXSEQS>8`) | the small-M decode GEMM band. Must cover `MAXSEQS x (SPEC+1)` rows or the biggest verify batches fall onto the prefill tile |
| `FAST_DRAFT` | `1` | the int2 draft head with an exact rerank: +6.5% decode under mtp, +5.1% under dflash |
| `RADIANCE_DRAFT_RERANK` | `80` dflash / `32` mtp | the candidate pool a top-k caller can draw from, not just a rescoring budget. Under dflash, 32 costs 5.3% of acceptance; 80 covers 4x the drafter's `selector_top_k` **and** 4x the sampler's `top_k` |
| `RADIANCE_VERIFY_HEAD` | `1` dflash / `0` mtp | the int2 head applied to the target's verify `lm_head` (one 2.02 ms GEMM per step, 5.9% of wall). +2.9% combined decode, output-equivalent |
| `RADIANCE_DYNAMIC_WIDTH` | `1` | scheduler-side per-request verify width from an acceptance EMA. Recovers static `SPEC=5`'s batch efficiency at `SPEC=7` without losing code depth. Lossless by construction |
| `RADIANCE_GDN_MERGE_INPROJ` | `1` | GDN `in_proj_qkvz` + `in_proj_ba` as one GEMM (-2.9% decode). **Changes the traced graph**, so it keys the cache directory |
| `RADIANCE_SKINNY_GEMM` | `1` | R4D split-K for skinny bf16 projections. `all` adds shapes that differ from rocBLAS at a bf16 ULP |
| `RADIANCE_MXFP4_SANITIZE` | `0` | zero non-finite activations. Only useful with `AUTO_R4D=0`, where it gets perplexity to 8.4004 instead of 653586 |

Diagnostics and bisect tools (`RADIANCE_MXFP4_CHECKALL`, `_SHADOW`, `_KERNEL_NK`, `_PERBLOCK_NK`,
`_MHIST`, `RADIANCE_GDN_NANTRACE`, `PROFILE_DIR`) are unset by default and documented where they are read
in `serve-mxfp4.sh`. The full image-level knob reference is in [DOCKERHUB.md](DOCKERHUB.md).

---

## How the MXFP4 path works

Quark OCP micro-scaling checkpoints (`quantization_config.quant_method: quark`, mxfp4 weights *and*
activations, group 32, e8m0 scales) run **natively** with `RADIANCE_MXFP4=1` -- e.g.
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Drop `--quantization`: the runtime reads the method from `config.json`
and routes it itself. [`serve-mxfp4.sh`](serve-mxfp4.sh) is the worked launch, and every
default in it carries the measurement that chose it; [MXFP4-NOTES.md](MXFP4-NOTES.md) collects the
longer form of that reasoning.

Without this, vLLM falls back to emulated MXFP4, which materialises every weight tensor in bf16 on each
forward. Nothing in the way was a compiler limitation -- Triton 3.6 does lower `tl.dot_scaled` on gfx1201 --
just three soft gates, all handled in `patch_quark_mxfp4.py`: an `is_fp4_avail()` allowlist that omits
gfx1201, an aiter module path that moved in 0.1.17, and gfx1250 tiles that ask for
`matrix_instr_nonkdim=32` when this card's WMMA is 16x16x16 only (`mxfp4-configs/` pins 16 across every
band). The native path is **bit-identical to emulation** -- the activation quantization is the same either
way -- so it is a speed change with no quality dimension: measured on gate_up 17408x5120, **6.1x at M=16,
4.7x at M=32, 2.5x at M=64**.

### The W4A8 fp8-WMMA kernel

`RADIANCE_MXFP4_W4A8=1` additionally routes linears to a hand-written fp8-WMMA HIP kernel
(`radiance_mxfp4_fp8.hip`). Triton lowers `tl.dot_scaled` by upconverting e2m1 to bf16 and using the 16-bit
WMMA; register-resident on this card, **fp8 WMMA runs 325 TFLOP/s against f16's 160**, while Triton's own
fp8 `tl.dot` manages 43 because it upconverts and pays conversion on top. Against the tuned aiter path it
measures **1.47-2.26x faster and 4.2x more accurate** (0.0265 vs 0.1119 relative error), since fp8
activations beat the mxfp4 ones aiter quantizes to. It is **off in the image because it changes numerics**
-- the layer becomes W4A8 rather than the checkpoint's declared W4A4, more precise than what the model was
calibrated against but no longer bit-identical -- and **on in `serve-mxfp4.sh`**, which is what every number
below was measured with. The N tile is M-keyed (`RADIANCE_MXFP4_TN4_MIN_M`, default
2048): the wide tile amortises A-tile staging for +10% at M=8192 but cannot fill below ~2048 rows.
`RADIANCE_MXFP4_W4A8_MIN_M` is where it takes over from aiter, and it now defaults to **0** -- our
kernel serves every M.

**There are two tilings, because prefill and decode are different problems.** The tile above
(BM=256 via TM=4) is sized for prefill. At decode M is 5 (batch 1 x `num_speculative_tokens`+1), where
it issues 51x more matrix MACs than useful -- 4352 WMMA per wave against 5 real rows. So small M goes
to a second kernel with TM=`ceil(M/16)`, no wasted M-fragments, and split-K to fill the CUs, gated by
`RADIANCE_MXFP4_DECODE_MAX_M` (default 64, or 128 at `MAXSEQS>8`; it must cover
`MAXSEQS x (SPEC+1)` rows or the biggest verify batches fall back onto the prefill tile). The split-K reduction is **fused**: the KS blocks covering
one output range race on an atomic counter and the last arrival reduces in place, so there is no
second launch — worth a further -3.9% of step time on top, at bit-identical output. It reverses one of the prefill answers: **BK=128 wins at
decode** (1.87x on gate_up) where it measured -34% at prefill, because that loss was purely the LDS
occupancy cliff and a 16-row A tile never reaches it.

**4-bit weights leave far more room for KV**: on 2x R9700 the 27B MXFP4 body occupies 9.24 GiB/GPU against
roughly 12.6 for the same model in FP8, and that headroom goes straight into context.

**`RADIANCE_MXFP4_MAX_M` is retired** (it was read up to 0.5.8). It handed big batches back to emulation as
a throughput win on paper; in practice quark's TileLang backend cannot initialise inside a vLLM worker, and
the branch was specialised into the compile graph during the `max-num-batched-tokens` profile run -- so it
killed startup rather than one request. That also means **the stock emulated path cannot serve these
checkpoints here at all**, which makes the native kernel the only way to run them on this card, not merely
the faster one.

### The drafter is FP8

Checkpoint is `Qwen3.8-27B-MXFP4-mtpfp8`: AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` body with the MTP
drafter requantized to fp8 by [`fp8_mtp.py`](fp8_mtp.py). The drafter must NOT be MXFP4 -- 4-bit
costs more acceptance than it saves in bandwidth, and AWQ does not rescue it. Data-free RTN
measured ~11.6% relative error and cost acceptance 2.5 -> 2.21; AWQ calibration improved that by
0-5% (the alpha search chose 0.1-0.2, and 0.0 for `mtp.fc`, because MXFP4's per-32 E8M0 block
exponent already does most of what per-channel scaling would). FP8 e4m3 per-channel is ~2-3%
relative error and holds acceptance at 2.60-2.80, removing ~17% of decode weight traffic instead
of 25% -- the smaller win that actually holds.

### Measured

These are the two gating measurements for the MXFP4 stack. Both were taken under `SPEC_METHOD=mtp`
at `SPEC=4`, which was the default at the time; the shipped default is now the `dflash` drafter, and
the numbers that moved with it are in [MXFP4-NOTES.md](MXFP4-NOTES.md).

Against the 0.5.8 MXFP4 build, same box (2x R9700, TP2), same harness:

| | 0.5.8 | 0.7.4 | |
|---|---|---|---|
| prefill 7.8k | 3873 | **4387** | +13.3% |
| prefill 26k | 3445 | **4138** | +20.1% |
| prefill 104k | 2310 | **3143** | +36.1% |
| prefill 182k | 1736 | **2511** | +44.6% |
| prefill 260k | 1393 | **2089** | +49.9% |
| decode short / medium | 63.0 / 67.4 | **67.1 / 67.5** | +6.5% / +0.1% |
| WikiText-2 PPL | 8.3335 | 8.3719 | +0.46% |

KV cache 857,399 tokens at `GPU_UTIL=0.98` under vLLM's own memory profiling; the explicit `KV_MEM`
pin the launcher now defaults to raises that to 943,581. All 304 linear layers run the W4A8 fp8-WMMA kernel;
`aiter` is not used at all now that `RADIANCE_MXFP4_W4A8_MIN_M` defaults to 0. The prefill gain
scales with context because it is mostly R4D's paged attention, whose share of prefill grows with
sequence length.

**The decode GEMM (`RADIANCE_MXFP4_DECODE_MAX_M`, default 64), measured against the same 0.7.4 with
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

Fixed in **StillDeadcode/libr4d PR #1** (merged 2026-08-24). Not in a tag yet, which is why setup
builds libr4d at a pinned commit instead of using the one in the image. The build is verified
reproducible: it produces an `r4d.so` byte-identical (sha256 `3026297b...`) to the one every number
here was measured with. The pin is deliberate -- nothing version-checks the library it loads, so a
later commit that renames an entry point or changes a compiled-in geometry constant makes
`radiance_gdn.py` set `ENABLED = False` and **fall back to the Triton path with one line on
stderr**, costing performance rather than raising.

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

## Building the image from source

You do not need any of this to serve: `setup-mxfp4.sh` pulls the published image, and the MXFP4
patches and kernels are applied at container start. Build from source only to change a pinned
component or a baked-in patch.

Everything the build needs is in this directory (a flat Docker build context). The version string lives in
one place, the `VERSION` file, which the tag and the build-arg both read (`podman build` takes the same
arguments):

```bash
docker build -t vllm-radiance:$(cat VERSION) --build-arg RADIANCE_VERSION=$(cat VERSION) .
```

That single command builds everything from source, in four stages. **builder** compiles PyTorch, Triton,
torchvision, AITER, and vLLM for `PYTORCH_ROCM_ARCH=gfx1201` against the official `rocm/dev-ubuntu-24.04`
base (digest-pinned) and leaves the wheels in `/wheels` (it also builds `rocm-bandwidth-test` for the startup
sweep). **rocmprune** (`prune_rocm.sh`) cuts the 19 GB ROCm tree down to this one GPU architecture.
**assemble** installs the wheels, applies the RDNA4 correctness patches, and clones and compiles
[libr4d](https://codeberg.org/StillDeadcode/libr4d) -- the hand-written gfx1201 kernel library (paged
attention, the fused gated-delta-net prefill scan, the P2P all-reduce, the skinny bf16 GEMM) -- with the
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
  removes the stock config's `M>=96` cliff for a lower prefill TTFT, lossless), plus the skinny bf16 GEMM
  below on the MoE gate. Both inert on models they do not apply to.
- **Skinny bf16 GEMM** (`RADIANCE_SKINNY_GEMM`, on): a projection small enough that rocBLAS lays it out as a
  handful of workgroups leaves most of the machine idle. The R4D split-K kernel takes those shapes for `M` in
  `[6,64]`. The default set is the ones that are a clear win alone; `all` adds shapes that differ from
  rocBLAS at a bf16 ULP, most importantly the gated-delta-net `in_proj_ba` -- 480 KiB run 48 times per step,
  28.5us against 3.6us.
- **Native MXFP4** for Quark OCP micro-scaling checkpoints (`RADIANCE_MXFP4`), bit-identical to vLLM's
  emulation and multiples faster, plus an optional hand-written fp8-WMMA W4A8 prefill GEMM
  (`RADIANCE_MXFP4_W4A8`) that reaches the fp8 matrix instruction Triton will not emit. See
  [How the MXFP4 path works](#how-the-mxfp4-path-works).
- **Lossless dynamic MTP drafting**: a per-request confidence gate plus verbatim n-gram tail that varies
  draft depth without changing what the model verifies. `mtp` only -- it works by stopping a serial loop of
  draft forwards early, and a `dflash` drafter has no such loop (it emits every position in one graphed pass).
- **The tuned drafter stack** (`RADIANCE_FAST_DRAFT`, one switch, opt-in): the draft head at 2 bits with an
  exact rerank (any drafter), plus a `dflash` drafter's decoder projections packed to signed symmetric int4
  -- one f16 scale per 128 input channels, no zero point, 4.25 bits per weight -- on two purpose-built
  gfx1201 kernels. Below 16 rows an f16 matrix-core kernel, which is already at the memory roofline there;
  above it an int8 one, because gfx1201's f16 matrix instruction is *half* the rate of its int8 one and a
  quarter of its int4 one. One packed weight feeds both: the nibble is a two's complement code the int8
  kernel reads by shifting it into a byte's high half, and the f16 kernel converts to offset binary in one
  XOR per dword. Codes are derived at load from the weight, so there is no calibration data, no offline step
  and nothing on disk. On Qwen3.8-27B with the DFlash2 drafter the draft pass falls 9.1% at a drafter batch
  of 64; with `RADIANCE_SKINNY_GEMM=all` the decode step falls 5.1% for +3.5% tokens/s over four paired
  compiles. Lossless in the same sense as any drafting change: the target verifies every proposed token with
  its own untouched weights.
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
`moe-configs/` and `mxfp4-configs/` GEMM configs, the chat templates, `Dockerfile`, and `docker-compose.yml` all live at the repo
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
radiance_mxfp4_fp8.so` rebuilds it against the image toolchain during development.

The MXFP4 entry points at the repo root:

| | |
|---|---|
| `setup-mxfp4.sh` | one-time setup: host check, image, checkpoints, kernels. Idempotent |
| `serve-mxfp4.sh` | the launcher. `--help` for the knobs, `DRY_RUN=1` to see the command it builds |
| `fp8_mtp.py` | builds the loadable checkpoint from AMD's release (setup drives it) |
| `MXFP4-NOTES.md` | the measurements, traps and history behind the defaults |
| `run_mxfp4_074.sh` | compatibility shim -- the launcher's old name, forwards to `serve-mxfp4.sh` |
| `run_mxfp4_minm.sh` | the 0.5.8 launch, frozen: it is the only way to reproduce the baseline the numbers above are measured against |
