# MXFP4 build notes

Design notes, measurements and traps behind `serve-mxfp4.sh`. None of this is needed to run the
server -- start at the [README quickstart](README.md#quickstart). It is kept because every default
in the launcher was chosen against a measurement, and the reasoning is what makes those defaults
re-derivable when something moves.

The text below is the original launcher header, preserved as written when the script was still
called `run_mxfp4_074.sh`.

---

PRODUCTION launcher: native MXFP4 body on gfx1201 with an FP8 drafter, on radiance 0.9.3 (libr4d).

The name still says 074 because that is what the file was called when it targeted the 0.7.4 image;
the defaults below track production and are what actually decide which server you get. The script
began as an evaluation harness for 0.7.4 and became the thing that starts the real server, which
is how its defaults came to lag two image versions behind what they launch.

The 0.5.8 form of this run is run_mxfp4_minm.sh, kept as-is because it is the only way to
reproduce the baseline these numbers are measured against:
  prefill 3602 / 3409 / 2283 / 1730 / 1390 tok/s at 7.8k/26k/104k/182k/259k
  real decode 55.4 / 61.7 tok/s | weights 9.24 GiB/GPU | KV ~856k tok | WikiText-2 PPL 8.3335

Checkpoint built by this repo's ./fp8_mtp.py from amd/Qwen3.8-27B-Quark-AWQ-MXFP4; run it once
before this script (it prints the command if the checkpoint is missing). AMD's release will not
load as-is: its mtp.* layers are bf16 but named in neither `exclude` nor `layer_quant_config`,
so vLLM applies the global mxfp4 scheme to them and asserts on a half-width parameter.

The drafter is FP8, not MXFP4, and that is a settled result: MXFP4 RTN cost acceptance
2.5 -> 2.21 and AWQ did not rescue it (MXFP4's per-32 E8M0 block exponent already does most of
what per-channel scaling would), while FP8 e4m3 per-channel holds acceptance at 2.60-2.80. Do
not point this at Qwen3.8-27B-MXFP4-mtpq or -mtpawq; those exist only for that comparison.

WHAT IS DIFFERENT FROM THE 0.5.8 RUN
  - image 0.5.8 -> 0.7.4; cache .radiance-cache-w4a8-058 -> -074. Cache dirs validate on model +
    torch/Triton version and MUST NOT be shared across configurations.
  - patch_quark_mxfp4.py is a different patch. vLLM 0.27 replaced QuarkOCP_MX's inline dispatch
    with a kernel plugin ABC, so instead of seven string hunks against one file it now registers
    RadianceMxfp4W4A8LinearKernel into _POSSIBLE_MXFP4_KERNELS and relaxes two aiter gates.
  - RADIANCE_MXFP4_MAX_M is GONE. It used to hand large batches back to emulation; that path
    could never run here anyway (quark's TileLang backend dies with "libamdhip64.so not found"
    inside the worker, and the branch got specialised into the compile graph during the M=8192
    profile run, killing startup), and with W4A8 on, large M belongs to the fp8-WMMA kernel.
  - RADIANCE_ATTN_TUNE is gone upstream (the AITER tune is unconditional now).
    RADIANCE_FAST_REDUCE -> RADIANCE_USE_R4D_AR, RADIANCE_AR_QUANT -> RADIANCE_USE_R4D_AR_QUANT,
    both under the master RADIANCE_USE_R4D.
  - RADIANCE_AR_MAX_KB is restored by patch_ar_maxbytes.py. Upstream hardcoded it at 48 MB,
    sized for its 4096-token chunk; at CHUNK=8192 the message is 80 MiB and every prefill
    reduction would silently fall back to RCCL. See that patch's docstring for the measurement.
  - --kv-cache-memory IS set now, to 18563072000 (17.29 GiB), but only when GPU_UTIL is the
    throughput default. Do NOT re-derive it from vLLM's "fit into requested memory" line: that
    number is computed against the GPU_UTIL budget (0.98 x 31.86 = 31.22 GiB), not against the
    card, and on 0.9.3 it suggests 15.47-15.61 GiB while the profiled run is already using 16.37
    and fits. Derive it from the card instead -- `rocm-smi --showmeminfo vram` INSIDE the
    container, under an 8-way 8192-chunk load, then hand KV everything but ~0.3 GiB. Measured:
    profiled 0.98 peaks at 30.64 of 31.86 GiB and the free 1.22 GiB stays free under load (KV is
    pre-allocated and the activation pool is reserved at profiling time), so 17.29 GiB peaks at
    31.57 with 0.29 free. Worth 892,799 -> 943,581 tokens. Re-derive after anything that moves
    weights, cudagraph sizes or CHUNK; getting it wrong OOMs at startup, not under load.
  - the chat template is deliberately unchanged for the parity run. Upstream re-derived
    qwen3.8-enhanced.jinja against the released official template (it was rendering booleans as
    True/False via `| string`); adopting that is a separate change, and doing it here would
    confound the benchmark.

STAGING. Land the enhancements one at a time -- a combined A/B cannot attribute a regression.
  (default)        R4D GDN + WHT6 all-reduce (RADIANCE_USE_R4D=1), AITER attention   <- parity gate
  R4D_ATTN=1       + the R4D paged attention backend -- ON BY DEFAULT, measured +37.8% prefill
                     at 260k, +30.9% at 182k, +20.9% at 104k against the 0.5.8 baseline
  FAST_DRAFT       int2 MTP draft head with an exact bf16 rerank. ON BY DEFAULT, worth
                     +6.5% decode short / +0.1% medium against the 0.5.8 baseline, and the
                     difference between 58.5 and 67.1 tok/s on this build.

                     It HUNG a worker at chunk 8192 before the libr4d GDN overflows were fixed:
                     the draft head was being fed NaN like everything else downstream of the
                     gated-delta-net. Fixing the kernel at source fixed the hang too, so it now
                     runs at the full chunk size. Historical note: it arms correctly
                     (this checkpoint's lm_head is bf16 and excluded from quantization, so the
                     exact-rerank guarantee holds), but a long-prompt sweep at chunk 8192 hung a
                     worker and killed the engine with an RPC TimeoutError in sample_tokens.
                     Upstream ships it opt-in at chunk 4096. Retry there before trusting it.
  DECODE_MAX_M     the small-M decode GEMM (M<=64, TM=ceil(M/16), split-K). ON BY DEFAULT at 64.
                     Needs MIN_M=0 to be reachable at all -- at MIN_M=16 the M=5 decode call
                     never enters our launcher. Measured: single-stream step time 35.1 -> 33.4 ms
                     (-4.8%), aggregate throughput +28.5% at 4 concurrent and +19.7% at 8, prefill
                     unchanged within 1.2%. GSM8K 500q paired: 486 both correct, 3/3 discordant,
                     sign test p=1.00, at 14% less wall. Same numbers in the README table.
                     Set 0 to fall back to the prefill-tiled kernel for every M.

                     Original note: it quantizes the drafter's
                     lm_head and reranks against an untouched bf16 copy, but this checkpoint's
                     drafter is FP8 and vLLM may be sharing the target's head, in which case
                     that bf16 copy -- and the exactness guarantee -- is not what it assumes.

NUMERICS REFERENCE (~/pibench-local/results/ppl/, WikiText-2, --chunks 300 --chars 3000,
208,539 tokens). Reproduce with GPU_UTIL=0.75 and `python3 ~/pibench-local/ppl.py --model
Qwen3.8-MXFP4 --tag <tag>`:
    8.3317  MXFP4 W4A8, exact bf16 all-reduce
    8.3335  MXFP4 W4A8, fp8 all-reduce   <- what 0.5.8 shipped
    8.3386  MXFP4 W4A4 (no W4A8), fp8 all-reduce
The 6-bit rotated payload replaces the fp8 one and is claimed slightly more accurate, so a
healthy 0.7.4 lands at or just under 8.3335. A jump well past it means the rewritten weight-prep
path is wrong, not that the all-reduce changed -- the whole AR spread is only 0.02%.

Port 8080 is prod's and this needs both GPUs, so stop production first:
  systemctl --user stop qwen_vllm_38          restore with: vllm-switch 38

WHAT TO CHECK IN THE LOG
  "Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"  -> our kernel won the selection
  "[radiance] native MXFP4 enabled on gfx12x"           -> the aiter fp4 gate was relaxed
  the R4D selections table (RADIANCE_R4D_REPORT=1)      -> which kernels bound, and why not
  the stock "current platform does not support native MXFP4/MXFP6" notice still prints and is a
  false alarm; it comes from a separate supports_mx() call.
