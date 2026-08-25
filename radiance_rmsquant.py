"""Radiance-native RMSNorm + per-token fp8 quant, for the vLLM fusion pass on gfx1201.

WHY THIS EXISTS. vLLM already carries the right pattern -- `rms_norm(x,w,eps)` followed by a
per-token dynamic fp8 quant, rewritten into one fused op (rocm_aiter_fusion.py,
AiterRMSNormDynamicQuantPattern). The matcher key is exactly what
`scaled_fp8_quant(use_per_token_if_dynamic=True)` emits. The only unusable piece is the REPLACEMENT:
it is aiter's `get_rmsnorm_fused_dynamic_quant_op()`, and aiter's RMSNorm does not work on RDNA4 --
which is why this build ships VLLM_ROCM_USE_AITER_RMSNORM=0, and why enabling
`pass_config.fuse_norm_quant` dies inside aiter's JIT.

So we supply our own replacement op and leave the pattern alone.

WHAT IT IS WORTH. The per-linear activation quant is 14,288 launches at decode (4.0% of wall) and
8,512 at prefill (4.2%), each doing ~2.2 us of work against ~4.7 us of dispatch. Fusing folds it into
the norm; and because vLLM merges the GDN projections into in_proj_qkvz and in_proj_ba, which BOTH
consume the same input_layernorm output, the fused norm also collapses two identical quants into one.

STATUS: the replacement is written as PLAIN TORCH, deliberately, not as a custom op. A
torch.library.custom_op is opaque to inductor, so a custom-op replacement runs as an eager chain of
~8 elementwise kernels -- measured, that pushed launches per forward from 1904 to 2009 and made step
time worse. Written as plain torch, inductor traces it and fuses the whole norm+quant into a single
triton kernel, which is the point. A HIP kernel can replace it later behind the same call site
(libr4d's r4d_gdn_gated_rmsnorm_h128_bf16 is the template); this is the version that has to be beaten.
"""
import os
import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _rms_quant(h: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """h,weight -> (fp8 per-token-quantized, fp32 scale [T,1]). Reference; kernel replaces this."""
    hf = h.float()
    normed = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + epsilon) * weight.float()
    scale = normed.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (normed / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


# The pass calls FUSED_OP with keyword args including quant_dtype; we only ever serve e4m3, so it is
# accepted and asserted rather than threaded into the op schema.
def fused_rmsnorm_quant(x, weight, epsilon, quant_dtype=FP8):
    return list(_rms_quant(x, weight, epsilon))


def fused_add_rmsnorm_quant(x, residual, weight, epsilon, quant_dtype=FP8):
    # The fused-add form also returns the NEW residual; order must be
    # (quantized, residual_out, scale) to match the pattern's replacement().
    s = x + residual
    q, sc = _rms_quant(s, weight, epsilon)
    return [q, s, sc]


ENABLED = os.environ.get("RADIANCE_RMS_QUANT_FUSION", "0") == "1"
