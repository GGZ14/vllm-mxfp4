"""Overlap the tensor-parallel all-reduce with the producing GEMM at prefill.

WHY. The RowParallel linears (mlp down_proj, attention o_proj, gdn out_proj) end in an 80 MiB
all-reduce at CHUNK=8192 that is strictly serialized behind the GEMM: 1.32 ms per call on the r4d
P2P kernel, 15.5% of prefill GPU time in the 40k profile, and transport-bound on PCIe -- the wht6
payload is already 6-bit, so the only lever left is hiding it. vLLM 0.27's async-TP machinery
(passes/fusion/collective_fusion.py) cannot do it here: it pattern-matches torch mm ops and
symmetric memory, and our GEMM is an opaque custom op it never sees -- the same reason the
rms+quant fusion matched zero of our sites.

HOW. Slice the row dimension: slice s's GEMM runs on the compute stream, and as soon as it is
done its all-reduce runs on a dedicated comm stream while slice s+1's GEMM computes. The whole
pipeline lives inside ONE custom op (radiance::mxfp4_linear_ar), so dynamo sees a single opaque
node -- streams, events and the M-dependent branch below are all runtime-internal. That branch
placement is load-bearing: a shape branch written in apply_weights splits the compile graph at
every linear (measured ~30% of decode when the M dispatch lived there).

  M >= RADIANCE_AR_OVERLAP_MIN_M : quantize once, run SLICES GEMM slices via mxfp4_linear_pq
                                   (row-slicing per-token quant and the GEMM is EXACT -- both are
                                   row-independent), pipeline each slice's AR on the comm stream.
  M below the threshold          : the stock unsliced path, all-reduce included, with no stream
                                   or event API at all -- this arm runs inside decode's captured
                                   CUDA graphs where event creation is not welcome.

The layer's own reduce_results is set False at install, so the op owns the reduction on BOTH
paths; a layer with a bias is left stock (nothing here serves one, and bias-after-reduce is the
stock contract).

NUMERICS. GEMM slices are bit-identical to the unsliced GEMM. The wht6 all-reduce quantizes per
call, so slicing moves its block boundaries and the reduced tensor is NOT bit-identical to the
unsliced reduction -- the same drift class as the AR quantization itself.

VERDICT (2026-08-29): MEASURED AND NEGATIVE -- ships default OFF, kept for re-ablation.
Serving prefill at 10k/40k/106k: baseline 5003/4648/3831 t/s; SLICES=1 (op structure, no
overlap) 4905/4518/3778 (-2 to -2.8%: the lost allreduce_rms fusion plus the assembly copies);
SLICES=4 4721/4390/3697 (-3.5 to -5.6%: WORSE than not overlapping). The r4d oneshot AR is a
spin-wait kernel: while it waits on the peer over PCIe it occupies CUs, so running it beside the
GEMM slices is CONTENTION, not hiding -- the overlap steals exactly the compute it hides under.
A truly-async AR needs the transfer on the SDMA engines (hipMemcpyPeerAsync + local reduce),
which cannot run the 6-bit wht6 compression and moves 2.7x the bytes; the projected net is
marginal against a full AR redesign. The first version of this file also OOMed the warmup:
allocating GEMM outputs and AR results under the comm stream piles up record_stream deferred
frees across 64 layers (>1 GiB dead reserved) -- hence the persistent-buffer design below, which
is the part of this experiment worth keeping.

CACHES. Enabling this changes the traced graph (the vllm::all_reduce node disappears into our
op), so it needs its own torch.compile cache dir -- a warm cache from a non-overlap serve replays
the OLD graph and silently ignores the feature. serve-mxfp4.sh keys the cache on the flag.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_AR_OVERLAP", "0") == "1"
MIN_M = int(os.environ.get("RADIANCE_AR_OVERLAP_MIN_M", "2048"))
SLICES = int(os.environ.get("RADIANCE_AR_OVERLAP_SLICES", "4"))

_comm_stream = None
_bufs = {"x8": None, "y": None, "xs": None}


def _log(msg):
    sys.stderr.write(f"[radiance.aroverlap] {msg}\n")
    sys.stderr.flush()


def _grow(name, numel, dtype, device):
    t = _bufs[name]
    if t is None or t.numel() < numel:
        _bufs[name] = t = torch.empty(numel, dtype=dtype, device=device)
    return t


@torch.library.custom_op("radiance::mxfp4_linear_ar", mutates_args=())
def mxfp4_linear_ar(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                    weight_ref: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce as _ar

    M = x.shape[0]
    if M < MIN_M:
        # Decode band: stock op (its own quant inside), then the reduction the layer no longer
        # does. No streams, no events, no persistent buffers -- CUDA-graph-capture safe.
        return _ar(torch.ops.radiance.mxfp4_linear(x, weight, weight_scale, weight_ref))

    # Sliced pipeline. ALLOCATION DISCIPLINE IS THE WHOLE DESIGN: the first version allocated the
    # GEMM outputs and AR results under the comm stream, and record_stream's deferred frees piled
    # up across 64 layers into >1 GiB of dead reserved memory -- the warmup OOMed on a stock
    # gate_up. Here the only cross-stream tensors are two PERSISTENT scratch buffers (x8, y),
    # the GEMM slices launch straight into them via _ext.launch with an explicit stream, and
    # everything allocated per call (scale, out, the AR results) lives on the MAIN stream with a
    # normal lifetime. Roles are swapped accordingly: the COMM stream runs the GEMM slices ahead
    # while MAIN alternates the all-reduces -- same overlap, no allocator churn.
    import radiance_mxfp4 as _rm
    from vllm import _custom_ops as _ops

    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()

    N, K = weight.shape[0], weight_scale.shape[0] * 32
    x8 = _grow("x8", M * K, torch.float8_e4m3fn, x.device)[: M * K].view(M, K)
    y = _grow("y", M * N, torch.bfloat16, x.device)[: M * N].view(M, N)
    _, xs = _ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True, output=x8)
    xs = xs.view(-1).float().contiguous()

    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    main = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record(main)
    _comm_stream.wait_event(ready)

    bounds = [(i * M) // SLICES for i in range(SLICES + 1)]
    evs = []
    for i in range(SLICES):
        a, b = bounds[i], bounds[i + 1]
        _rm._ext.launch(x8[a:b].data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                        weight_ref.data_ptr(), xs[a:b].data_ptr(), y[a:b].data_ptr(),
                        b - a, N, K, _comm_stream.cuda_stream)
        e = torch.cuda.Event()
        e.record(_comm_stream)
        evs.append((a, b, e))
    for a, b, e in evs:
        main.wait_event(e)
        out[a:b].copy_(_ar(y[a:b]))
    # main is now ordered after every comm-stream read of x8/y, so the next call's quant cannot
    # race the scratch buffers.
    return out


@mxfp4_linear_ar.register_fake
def _(x, weight, weight_scale, weight_ref):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


def install(model) -> None:
    """Route every our-kernel RowParallel linear through the pipelined op. Best-effort."""
    if not ENABLED:
        return
    try:
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.distributed import get_tensor_model_parallel_world_size
    except Exception as e:                          # noqa: BLE001
        _log(f"vllm imports failed, skipping: {e!r}")
        return
    if get_tensor_model_parallel_world_size() <= 1:
        _log("tp=1, nothing to overlap")
        return
    n = skipped = 0
    for m in model.modules():
        if not isinstance(m, RowParallelLinear):
            continue
        if not (m.reduce_results and getattr(m, "radiance_w4a8_ok", False)
                and getattr(m, "radiance_wref", None) is not None
                and getattr(m, "bias", None) is None):
            skipped += 1
            continue
        m.reduce_results = False
        m._rad_ar_overlap = True
        n += 1
    _log(f"pipelined AR on {n} RowParallel layers ({skipped} left stock), "
         f"slices={SLICES}, min_m={MIN_M}")
