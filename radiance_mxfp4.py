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
    sys.stderr.write(
        f"[radiance.mxfp4] W4A8 fp8-WMMA GEMM ENABLED for M>{MIN_M} "
        f"(NOTE: W4A8, not the checkpoint's W4A4 -- more precise activations, not bit-identical)\n")


# The WHOLE W4A8 path -- activation quant included -- is one registered custom op. An earlier
# version called a plain Python helper from apply_weights instead, which sits inside the
# torch.compile region: dynamo could not trace through the module-object call, broke the graph at
# every one of ~450 linears, and cost 33% of DECODE throughput (61 -> 41 tok/s) even though decode
# never reaches the kernel. A registered op is a single opaque graph node, and the gate in front of
# it must be nothing but plain attribute/shape comparisons.
@torch.library.custom_op("radiance::mxfp4_linear", mutates_args=())
def mxfp4_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 weight_ref: torch.Tensor, w4a8_ok: bool, asm: bool) -> torch.Tensor:
    """Owns the ENTIRE dispatch, because the branch must not be visible to dynamo.

    vLLM compiles the model with a dynamic token dimension, so a plain `x.shape[0] > 256` in
    apply_weights is a data-dependent branch: it splits the graph at every linear and cost ~30% of
    decode throughput even though decode never takes the W4A8 side. Inside a registered custom op
    the body runs eagerly and the Python `if` is free."""
    if not (w4a8_ok and x.shape[0] > MIN_M):
        return torch.ops.vllm.gemm_with_dynamic_quant(x, weight, weight_scale, asm,
                                                      torch.bfloat16)
    from vllm import _custom_ops as ops
    M, K = x.shape
    N = weight.shape[0]
    x_fp8, x_scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    x_scale = x_scale.view(-1).float().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch(x_fp8.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                # numel()==1 is the placeholder every ineligible layer carries; passing 0 makes
                # the kernel take the per-block path
                weight_ref.data_ptr() if weight_ref.numel() == weight.shape[0] else 0,
                x_scale.data_ptr(), out.data_ptr(), M, N, K,
                torch.cuda.current_stream().cuda_stream)
    return out


@mxfp4_linear.register_fake
def _(x, weight, weight_scale, weight_ref, w4a8_ok, asm):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


def make_row_ref(weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-output-row reference exponent, computed ONCE at load.

    Folding the MX block exponent into the weight needs a single reference per row; the kernel
    stores 2^(ref-127) back in the epilogue. weight_scale arrives as [K/32, N] (the layout the
    native path leaves behind), so the max is over dim 0."""
    return weight_scale.max(dim=0).values.contiguous()


def layer_is_supported(layer, K: int) -> bool:
    """Called ONCE per layer at load time, never in the forward path.

    Kernel tiles are BM128/BN64/BK64, so K and N must be multiples of 64, and the weight scale must
    be the transposed [K/32, N] that the native path leaves behind."""
    try:
        N = layer.weight.shape[0]
        return bool(ENABLED and K % 64 == 0 and N % 64 == 0
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
            # Every layer carries the attribute so apply_weights never branches on hasattr, which
            # dynamo would have to guard. Ineligible layers get a 1-element placeholder, and the
            # op passes 0 for it, which makes the kernel take the per-block-rescale path.
            ref = make_row_ref(layer.weight_scale.data) if ok else torch.zeros(
                1, dtype=layer.weight_scale.dtype, device=layer.weight_scale.device)
            layer.radiance_wref = torch.nn.Parameter(ref, requires_grad=False)
            layer.radiance_w4a8_ok = bool(ok)

        def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
            y = torch.ops.radiance.mxfp4_linear(
                x, layer.weight, layer.weight_scale, layer.radiance_wref,
                layer.radiance_w4a8_ok, False)
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
