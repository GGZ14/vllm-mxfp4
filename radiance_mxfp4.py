"""RADIANCE gfx1201 MXFP4 dispatch: routes large-M (prefill) MXFP4 linears to the hand-written
W4A8 fp8-WMMA kernel, leaving decode on aiter's Triton W4A4 path.

Why: Triton lowers tl.dot_scaled by upconverting e2m1 to bf16 and using the 16-bit WMMA. Measured
here, register-resident: fp8 WMMA 325.2 TFLOP/s vs f16 160.2 (2.03x), while Triton's own fp8 tl.dot
manages only 43.3 -- it will not emit the fp8 matrix instruction. The hand-written kernel measures
1.6-1.9x the tuned aiter path at prefill shapes.

This is a NUMERICS change, not just a speed one: the checkpoint declares W4A4 and this runs W4A8.
fp8 activations are strictly more precise than the fp4 the model was calibrated against, but output
is no longer bit-identical to emulation, so it is opt-in via RADIANCE_MXFP4_W4A8=1 and only above
RADIANCE_MXFP4_W4A8_MIN_M (default 256, where the fp8 kernel starts winning).

Integration: vLLM 0.27 replaced QuarkOCP_MX's inline dispatch with a kernel plugin ABC --
MxFp4LinearKernel, selected in priority order from _POSSIBLE_MXFP4_KERNELS[platform] by
init_mxfp4_linear_kernel(). RadianceMxfp4W4A8LinearKernel below is that plugin;
patch_quark_mxfp4.py does nothing but put it at the head of the ROCm list. Before 0.27 this took
seven string hunks against a single 389-line file, all of which the rewrite invalidated.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_MXFP4_W4A8", "0") == "1"
MIN_M = int(os.environ.get("RADIANCE_MXFP4_W4A8_MIN_M", "256"))

try:
    import radiance_mxfp4_fp8 as _ext
except Exception as e:                      # ext missing: stay on the aiter Triton path
    _ext = None
    ENABLED = False
    sys.stderr.write(f"[radiance.mxfp4] w4a8 ext import failed, disabled: {e!r}\n")

if ENABLED and _ext is not None:
    # Print WHICH .so was loaded. A stale copy sitting in the bind-mounted repo shadows the one
    # compiled into site-packages (the working directory precedes it on sys.path), and a
    # mismatched kernel fails silently -- fluent-looking garbage, no error anywhere.
    sys.stderr.write(
        f"[radiance.mxfp4] W4A8 fp8-WMMA GEMM ENABLED for M>{MIN_M} "
        f"(NOTE: W4A8, not the checkpoint's W4A4 -- more precise activations, not bit-identical)\n"
        f"[radiance.mxfp4] kernel: {getattr(_ext, '__file__', '?')}\n")


# The WHOLE W4A8 path -- activation quant included -- is one registered custom op. An earlier
# version called a plain Python helper from apply_weights instead, which sits inside the
# torch.compile region: dynamo could not trace through the module-object call, broke the graph at
# every one of ~450 linears, and cost 33% of DECODE throughput (61 -> 41 tok/s) even though decode
# never reaches the kernel. A registered op is a single opaque graph node, and the gate in front of
# it must be nothing but plain attribute/shape comparisons.
DEBUG = os.environ.get("RADIANCE_MXFP4_DEBUG", "0") == "1"
PURE_QUANT = os.environ.get("RADIANCE_MXFP4_PUREQUANT", "0") == "1"
# Diagnostic: synchronize after the raw kernel launch. Every layer verifies correct against an
# exact fp32 reference -- but that check calls .item(), which synchronizes, and it only runs for
# the first few calls. If forcing a sync makes the model coherent, the fault is ordering: the
# kernel is being launched on a stream the consumer does not wait on.
_fs = os.environ.get("RADIANCE_MXFP4_SYNC", "0")
FORCE_SYNC = _fs == "1"          # synchronize the stream we launched on
FORCE_DEVSYNC = _fs == "2"       # synchronize the whole device -- if this differs from the above,
                                 # the kernel is not running on the stream we handed it
# Diagnostic: return a copy of the kernel's output buffer instead of the buffer itself. `out` is
# allocated by torch.empty INSIDE the custom op; if returning that buffer is what breaks (lifetime,
# aliasing, or the allocator reusing it), a clone will be coherent where the original is not.
CLONE_OUT = os.environ.get("RADIANCE_MXFP4_CLONE", "0") == "1"
# Diagnostic: report the INPUT activation's health per layer. The exact-reference check derives its
# reference from x itself, so it cannot tell a correct kernel on corrupt input from a correct one.
CHECK_X = os.environ.get("RADIANCE_MXFP4_CHECKX", "0") == "1"
# Verify EVERY call at one N:K against exact fp32, with no dedup. The earlier check kept only the
# first call per (N,K,M), so a shape that is right once and wrong later reads as "ok".
_ca = os.environ.get("RADIANCE_MXFP4_CHECKALL", "").strip()
CHECK_ALL = tuple(int(v) for v in _ca.split(":")) if _ca else None
# Bisect which layer class our kernel breaks. Comma-separated N values our kernel is allowed to
# serve; every other layer is handed to aiter (which is known-coherent for the whole model).
# Empty = no restriction. Gating projections (N=48, in_proj_ba) feed exponentials in the GDN core,
# so a small numeric difference there can become Inf/NaN downstream where an MLP shape cannot.
_only = os.environ.get("RADIANCE_MXFP4_KERNEL_N", "").strip()
KERNEL_N = {int(v) for v in _only.split(",") if v} if _only else None
# Finer bisect: N alone is ambiguous. N=5120 is BOTH the MLP down_proj (K=8704) and the
# gated-delta-net out_proj (K=3072), which are very different layers. "N:K,N:K" pairs separate them.
_onlynk = os.environ.get("RADIANCE_MXFP4_KERNEL_NK", "").strip()
KERNEL_NK = {tuple(int(x) for x in pair.split(":")) for pair in _onlynk.split(",") if pair} \
    if _onlynk else None
# Shapes forced onto the kernel's per-block-rescale path (wref=0) instead of the folded path.
# The folded path pre-shifts the block exponent into the weight via kLUT2; the per-block path
# applies the scale in the inner loop instead. Same kernel, different numerics path.
_pb = os.environ.get("RADIANCE_MXFP4_PERBLOCK_NK", "").strip()
PERBLOCK_NK = {tuple(int(x) for x in pair.split(":")) for pair in _pb.split(",") if pair} \
    if _pb else None
# Diagnostic: compute the layer exactly (dequantize the weight, bf16 F.linear) and return THAT,
# while still doing all the surrounding work. Slow. It separates "our kernel's values are wrong in
# situ" from "something outside this op is wrong": if the model is coherent with this on, the op's
# wiring is fine and only the kernel output differs; if it is still garbage, the fault is elsewhere.
REF_LINEAR = os.environ.get("RADIANCE_MXFP4_REFLINEAR", "0") == "1"
_E2M1 = None
_dbg_seen = set()


def _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K, chunk=2048):
    """fp32 reference for this layer, chunked over N so it fits alongside a loaded model.

    Compared against AITER the earlier check was circular -- aiter is itself wrong on some shapes,
    so "rel 0.05-0.16 looks like the W4A8-vs-W4A4 spread" proved nothing. This is ground truth.
    """
    global _E2M1
    if _E2M1 is None:
        _E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                              -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=weight.device)
    xr = x_fp8.float() * x_scale.view(-1, 1).float()
    outs = []
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        wc = weight[a:b]
        codes = torch.stack([wc & 0x0F, (wc >> 4) & 0x0F], -1).reshape(b - a, K)
        sc = torch.pow(2.0, weight_scale[:, a:b].float() - 127.0).T.repeat_interleave(32, dim=1)
        outs.append(xr @ (_E2M1[codes.long()] * sc).T.float())
    return torch.cat(outs, dim=1)


def _debug_compare(x, weight, weight_scale, weight_ref, out, x_fp8, x_scale):
    """One-shot per (N,K): how far is our kernel from the aiter path on REAL activations?

    aiter quantizes x to mxfp4 where we use fp8, so a relative difference around 0.1 is expected
    and healthy (measured 0.0265 vs 0.1119 against exact arithmetic). Order-1 means we are wrong.
    """
    key = (int(weight.shape[0]), int(x.shape[1]), int(x.shape[0]))
    if key in _dbg_seen or len(_dbg_seen) > 40:
        return
    _dbg_seen.add(key)
    try:
        M, N, K = int(x.shape[0]), int(weight.shape[0]), int(x.shape[1])
        if M > 128:
            return                      # reference is only affordable at small M
        ref = _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
        sys.stderr.write(f"[radiance.mxfp4.exact] N={N} K={K} M={M} rel_vs_fp32={rel:.5f} "
                         f"|ours|={out.float().abs().mean():.5f} |ref|={ref.abs().mean():.5f} "
                         f"{'**WRONG**' if rel > 0.02 else 'ok'}\n")
        sys.stderr.flush()
        return
        sys.stderr.write(
            f"[radiance.mxfp4.debug] N={key[0]} K={key[1]} M={x.shape[0]} "
            f"x={tuple(x.shape)}/{x.dtype}/contig={x.is_contiguous()} "
            f"w={tuple(weight.shape)}/{weight.dtype} "
            f"ws={tuple(weight_scale.shape)}/{weight_scale.dtype}/contig={weight_scale.is_contiguous()} "
            f"wref={tuple(weight_ref.shape)}/{weight_ref.dtype} "
            f"xq={x_fp8.dtype} xs={tuple(x_scale.shape)}/{x_scale.dtype} "
            f"|ours|={out.float().abs().mean().item():.4f} |aiter|={ref.float().abs().mean().item():.4f} "
            f"rel={rel:.4f} nonfinite={(~torch.isfinite(out.float())).sum().item()}\n")
        sys.stderr.flush()
        if rel > 0.5:
            # Deterministic-wrong or a race? Re-run the kernel into a fresh buffer from the SAME
            # inputs, then dump everything so it can be replayed offline against the same .so.
            out2 = torch.empty_like(out)
            _ext.launch(x_fp8.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                        weight_ref.data_ptr(), x_scale.data_ptr(), out2.data_ptr(),
                        int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]),
                        torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            rerun = ((out2.float() - out.float()).norm()
                     / out.float().norm().clamp_min(1e-9)).item()
            rel2 = ((out2.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
            import os as _os
            path = f"/cache/badcase_{_os.getpid()}_{key[0]}_{key[1]}_{key[2]}.pt"
            torch.save({"x_fp8": x_fp8.cpu(), "x_scale": x_scale.cpu(),
                        "weight": weight.cpu(), "weight_scale": weight_scale.cpu(),
                        "weight_ref": weight_ref.cpu(), "out": out.cpu(),
                        "out_rerun": out2.cpu(), "aiter": ref.cpu(), "x": x.cpu(),
                        "M": int(x.shape[0]), "N": int(weight.shape[0]), "K": int(x.shape[1])}, path)
            sys.stderr.write(f"[radiance.mxfp4.debug] BAD CASE rel={rel:.4f} rerun_delta={rerun:.6f} "
                             f"rel_of_rerun={rel2:.4f} -> {path}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[radiance.mxfp4.debug] compare failed: {e!r}\n")


@torch.library.custom_op("radiance::mxfp4_linear", mutates_args=())
def mxfp4_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 weight_ref: torch.Tensor) -> torch.Tensor:
    """Owns the ENTIRE dispatch, because the branch must not be visible to dynamo.

    vLLM compiles the model with a dynamic token dimension, so a plain `x.shape[0] > 256` in
    apply_weights is a data-dependent branch: it splits the graph at every linear and cost ~30% of
    decode throughput even though decode never takes the W4A8 side. Inside a registered custom op
    the body runs eagerly and the Python `if` is free."""
    if not _stats_reported[0]:
        report_stats()          # safe here, and only here: the op body is opaque to dynamo
    # Eligibility is derived from the operands, not passed in. It used to arrive as a Python bool
    # read off the layer in apply_weights -- which dynamo TRACES and bakes into the graph as a
    # per-layer constant. Keeping scalars out of the traced region is the same rule the M
    # comparison already follows, and it means a stale compiled graph cannot carry a stale gate.
    # weight_ref encodes the route, because a Python scalar read in apply_weights would be traced
    # and baked into the compiled graph:
    #   numel == N -> our kernel, folded path (block exponent pre-shifted into the weight)
    #   numel == 2 -> our kernel, per-block path (scale applied in the inner loop); wref ptr = 0
    #   numel == 1 -> the kernel cannot serve this layer at all; hand it to aiter
    nref = weight_ref.numel()
    folded = nref == weight.shape[0]
    w4a8_ok = folded or nref == 2
    if not (w4a8_ok and x.shape[0] > MIN_M):
        STATS["aiter_calls"] = STATS.get("aiter_calls", 0) + 1
        if STATS["aiter_calls"] in (1, 100, 10000):
            sys.stderr.write(f"[radiance.mxfp4] AITER BRANCH TAKEN "
                             f"(call #{STATS['aiter_calls']}, M={x.shape[0]}, "
                             f"N={weight.shape[0]}, w4a8_ok={w4a8_ok})\n")
        # MIN_M <= 0 makes this unreachable: aiter's W4A4 path is measured WRONG on some shapes
        # (N=5120 K=3072 returns ~1/35th of the correct magnitude), so it is not a safe fallback.
        return torch.ops.vllm.gemm_with_dynamic_quant(x, weight, weight_scale, False,
                                                      torch.bfloat16)
    M, K = x.shape
    N = weight.shape[0]
    if PURE_QUANT:
        # Pure-torch per-token e4m3 quantization, no vLLM custom op. Diagnostic only: this exists
        # to answer whether calling torch.ops._C.dynamic_scaled_fp8_quant from INSIDE another
        # custom op is what disturbs the compiled graph.
        amax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12)
        sc = amax / 448.0
        x_fp8 = (x.float() / sc).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        x_scale = sc.view(-1).contiguous()
    else:
        from vllm import _custom_ops as ops
        x_fp8, x_scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
        x_scale = x_scale.view(-1).float().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch(x_fp8.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                weight_ref.data_ptr() if folded else 0,
                x_scale.data_ptr(), out.data_ptr(), M, N, K,
                torch.cuda.current_stream().cuda_stream)
    if CHECK_ALL is not None and (N, K) == CHECK_ALL and x.shape[0] <= 128:
        _ref = _exact_ref(x_fp8, x_scale, weight, weight_scale, N, K)
        _rel = ((out.float() - _ref).norm() / _ref.norm().clamp_min(1e-9)).item()
        STATS["checked"] = STATS.get("checked", 0) + 1
        if _rel > 0.02:
            STATS["wrong"] = STATS.get("wrong", 0) + 1
            if STATS["wrong"] <= 8:
                xf2 = x.float()
                sys.stderr.write(
                    f"[radiance.mxfp4.all] **WRONG** call#{STATS['checked']} N={N} K={K} "
                    f"M={x.shape[0]} rel={_rel:.4f} |ours|={out.float().abs().mean():.5f} "
                    f"|ref|={_ref.abs().mean():.5f} x_nonfin={(~torch.isfinite(xf2)).sum().item()} "
                    f"x_contig={x.is_contiguous()} x_stride={tuple(x.stride())}\n")
        elif STATS["checked"] in (1, 32, 64):
            sys.stderr.write(f"[radiance.mxfp4.all] ok call#{STATS['checked']} rel={_rel:.5f} "
                             f"(wrong so far: {STATS.get('wrong', 0)})\n")
        sys.stderr.flush()
    if CHECK_X:
        key = (N, K, int(x.shape[0]))
        if key not in _dbg_seen and len(_dbg_seen) < 60:
            _dbg_seen.add(key)
            xf = x.float()
            of = out.float()
            sys.stderr.write(f"[radiance.mxfp4.x] N={N} K={K} M={x.shape[0]} "
                             f"|x|={xf.abs().mean().item():.5f} xmax={xf.abs().max().item():.5f} "
                             f"x_nonfin={(~torch.isfinite(xf)).sum().item()} | "
                             f"|out|={of.abs().mean().item():.5f} "
                             f"outmax={of.abs().max().item():.5f} "
                             f"out_nonfin={(~torch.isfinite(of)).sum().item()} "
                             f"out_over448={(of.abs() > 448).sum().item()}\n")
            sys.stderr.flush()
    if FORCE_SYNC:
        torch.cuda.current_stream().synchronize()
    if FORCE_DEVSYNC:
        torch.cuda.synchronize()
    if DEBUG:
        _debug_compare(x, weight, weight_scale, weight_ref, out, x_fp8, x_scale)
    if REF_LINEAR:
        global _E2M1
        if _E2M1 is None:
            _E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                  -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                                 device=weight.device)
        codes = torch.stack([weight & 0x0F, (weight >> 4) & 0x0F], -1).reshape(N, K)
        w = _E2M1[codes.long()] * torch.pow(
            2.0, weight_scale.float() - 127.0).T.repeat_interleave(32, dim=1)
        return (x.float() @ w.T.float()).to(torch.bfloat16)
    return out.clone() if CLONE_OUT else out


@mxfp4_linear.register_fake
def _(x, weight, weight_scale, weight_ref):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


def make_row_ref(weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-output-row reference exponent, computed ONCE at load.

    Folding the MX block exponent into the weight needs a single reference per row; the kernel
    stores 2^(ref-127) back in the epilogue. weight_scale arrives as [K/32, N] (the layout the
    native path leaves behind), so the max is over dim 0."""
    return weight_scale.max(dim=0).values.contiguous()


# Per-layer accounting. A layer that fails layer_is_supported() still runs, on the kernel's slower
# per-block-rescale path, and a layer below MIN_M runs on aiter -- both silently. Count them so
# "no fallbacks" is something the log proves rather than something we assume.
STATS = {"fast": 0, "aiter": 0}
_stats_reported = [False]


def report_stats():
    """One-shot fast-path tally. Call ONLY from inside the custom op or at load time -- never from
    apply_weights, which dynamo traces: sys.stderr.write is a skipped builtin and compilation
    fails outright rather than falling back."""
    if _stats_reported[0]:
        return
    _stats_reported[0] = True
    n = STATS["fast"] + STATS["aiter"]
    sys.stderr.write(
        f"[radiance.mxfp4] linear layers: {STATS['fast']}/{n} on our kernel, "
        f"{STATS['aiter']} FORCED ONTO AITER (kernel cannot run them); "
        f"aiter below M={MIN_M} ({'DISABLED' if MIN_M <= 0 else 'ACTIVE'})\n")
    sys.stderr.flush()


def layer_is_supported(layer, K: int) -> bool:
    """Called ONCE per layer at load time, never in the forward path.

    Only K is a hard constraint: the launcher rejects K % BK, where BK=64. N is NOT -- the kernel
    masks a partial N tile, measured at N=48 (the gated-delta-net gate projection) as relRMSE
    0.00174 folded / 0.00155 per-block with zero out-of-bounds writes on either side of `out`.

    The old gate also demanded N % 64 == 0, which failed those 48 layers per rank and sent them to
    aiter -- NOT, as the name suggested, to this kernel's per-block path. That mattered: aiter's
    W4A4 Triton GEMM returns wrong values for some shape/M-band combinations here, so the strict
    gate was silently routing real layers onto a path that cannot be trusted."""
    try:
        return bool(ENABLED and K % 64 == 0
                    and layer.weight.shape[1] * 2 == K
                    and layer.weight_scale.dim() == 2
                    and layer.weight_scale.shape[0] == K // 32)
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# The vLLM 0.27 kernel plugin
# --------------------------------------------------------------------------------------------
# Deliberately NOT composed with AiterMxfp4LinearKernel. Its __init__ asserts is_supported(), which
# gates on current_platform.supports_mx() -- a CDNA4 (gfx950/gfx1250) allowlist that gfx1201 fails.
# The sub-MIN_M fallback calls torch.ops.vllm.gemm_with_dynamic_quant directly instead. That op is
# registered by vllm/model_executor/kernels/linear/mxfp4/aiter.py under
# `if is_aiter_found_and_supported():`, which does NOT consult supports_mx(), so it is present here.


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return bool(on_gfx12x())
    except Exception:
        return False


def _asm_gemm_enabled() -> bool:
    try:
        from vllm._aiter_ops import rocm_aiter_ops
        return bool(rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled())
    except Exception:
        return False


def _make_kernel_class():
    """Built lazily so importing this module never drags in vllm.model_executor.kernels."""
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearKernel,
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    class RadianceMxfp4W4A8LinearKernel(MxFp4LinearKernel):
        """MXFP4 weights x fp8 activations on gfx1201, via the hand-written fp8-WMMA GEMM."""

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not ENABLED or _ext is None:
                return False, "RADIANCE_MXFP4_W4A8 is not enabled, or the HIP extension is missing"
            if not _on_gfx12x():
                return False, "the radiance W4A8 MXFP4 kernel is compiled for gfx12x only"
            return True, None

        @classmethod
        def can_implement(cls, config: MxFp4LinearLayerConfig):
            if config.activation_quant_key != kMxfp4Dynamic:
                return False, "only supports MXFP4 dynamic activation"
            # The asm path stores weights shuffled (16,16) and the scale in a swizzled layout;
            # this kernel reads the plain packed weight and a [K/32, N] scale, and the sub-MIN_M
            # fallback would hand shuffled operands to the non-asm aiter GEMM. Decline instead of
            # silently computing the wrong thing -- AiterMxfp4LinearKernel takes it from here.
            if _asm_gemm_enabled():
                return False, "aiter asm fp4 GEMM is enabled; its weight layout is incompatible"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Same transpose AiterMxfp4LinearKernel's non-asm branch does: create_weights lays the
            # scale out as [N, K/32] and both the aiter GEMM and this kernel want [K/32, N].
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data.T.contiguous(), requires_grad=False)

            K = layer.weight.shape[1] * 2          # weights are 2 e2m1 codes per byte
            ok = layer_is_supported(layer, K)
            if ok and KERNEL_N is not None and int(layer.weight.shape[0]) not in KERNEL_N:
                ok = False          # bisect: hand this shape to aiter instead
            if ok and KERNEL_NK is not None and (int(layer.weight.shape[0]), K) not in KERNEL_NK:
                ok = False
            # Every layer carries the attribute so apply_weights never branches on hasattr, which
            # dynamo would have to guard. Ineligible layers get a 1-element placeholder, and the
            # op passes 0 for it, which makes the kernel take the per-block-rescale path.
            if ok:
                STATS["fast"] += 1
            else:
                STATS["aiter"] += 1
                sys.stderr.write(
                    f"[radiance.mxfp4] FALLBACK TO AITER (not our kernel): "
                    f"N={tuple(layer.weight.shape)[0]} K={K} "
                    f"ws={tuple(layer.weight_scale.shape)}\n")
            perblock = ok and PERBLOCK_NK is not None and \
                (int(layer.weight.shape[0]), K) in PERBLOCK_NK
            if perblock:
                STATS["fast"] -= 1
                STATS["perblock_forced"] = STATS.get("perblock_forced", 0) + 1
            if ok and not perblock:
                ref = make_row_ref(layer.weight_scale.data)          # folded
            else:
                # 2 elements = our kernel's per-block path; 1 = aiter (kernel cannot serve it)
                ref = torch.zeros(2 if perblock else 1,
                                  dtype=layer.weight_scale.dtype,
                                  device=layer.weight_scale.device)
            layer.radiance_wref = torch.nn.Parameter(ref, requires_grad=False)
            layer.radiance_w4a8_ok = bool(ok)   # record only; never read in the forward

        def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
            y = torch.ops.radiance.mxfp4_linear(
                x, layer.weight, layer.weight_scale, layer.radiance_wref)
            if bias is not None:
                y = y + bias
            return y

    return RadianceMxfp4W4A8LinearKernel


_KERNEL_CLS = None


def kernel_class():
    """The plugin class, built once. Returns None if anything about it is unavailable."""
    global _KERNEL_CLS
    if _KERNEL_CLS is None:
        try:
            _KERNEL_CLS = _make_kernel_class()
        except Exception as e:
            sys.stderr.write(f"[radiance.mxfp4] kernel class unavailable, disabled: {e!r}\n")
            _KERNEL_CLS = False
    return _KERNEL_CLS or None
