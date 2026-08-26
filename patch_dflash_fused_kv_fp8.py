#!/usr/bin/env python3
"""Let the DFlash context-KV precompute work with a quantized drafter.

`precompute_and_store_context_kv` fuses every layer's K/V projection into one GEMM, and it builds
that fused weight by slicing the raw parameter: `a.qkv_proj.weight[a.q_size:]`. Every assumption
behind that slice fails once the drafter checkpoint is quantized:

  * the parameter is float8_e4m3fn, so the F.linear against bf16 activations raises outright;
  * the rows are not where the slice expects them -- our preshuffle load hook rewrites block-fp8
    weights to [N//16, K*16], so row `q_size` is no longer the first K row; and
  * the buffer is built at the end of load_weights, before the quant method has processed the
    weights at all, so nothing can be read out of them yet.

Rather than teach this path about every weight layout, ask the layer for its own effective weight:
running the identity through `qkv_proj` returns W^T whatever the quant method and layout underneath.
A one-hot row has amax 1, so its activation quantization is exact and the result is the dequantized
weight rounded to the compute dtype -- precisely what the fused GEMM would have multiplied by. The
build is deferred to first use, which is the profile run, so the weights are fully processed and the
allocation still happens long before any CUDA graph capture. Steady-state memory is unchanged: the
fused buffer is the same bf16 tensor the unquantized path builds.

An unquantized drafter keeps the original eager slice, so the bf16 configuration is untouched.

The test is on the DENSE dtypes rather than on a list of quantized ones. The identity trick is
scheme-agnostic by construction -- it asks the layer for its own effective weight -- so the only
question is whether `weight` can be sliced directly, and that is true exactly when it is a real
float matrix. Enumerating quantized dtypes instead silently misroutes any scheme not on the list:
an MXFP4 checkpoint stores `weight` as packed uint8 of shape [N, K/2], which is not an fp8 dtype,
so it took the dense slice and died in precompute_and_store_context_kv with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (8192x5120 and 2560x5120)

where 2560 is K/2 -- packed nibbles being multiplied as if they were a bf16 matrix. Exactness holds
for MXFP4 too: the identity's entries are 0.0 and 1.0, both representable in e2m1, and a block of
one 1.0 with 31 zeros takes scale 2^-2 so the element encodes as 4.0 with no rounding.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/qwen3_dflash.py"

HELPER_OLD = """@support_torch_compile
class DFlashQwen3Model(nn.Module):"""

HELPER_NEW = '''_DFLASH_DENSE = (torch.bfloat16, torch.float16, torch.float32)


def _dflash_kv_weight_rows(qkv_proj, q_size: int) -> torch.Tensor:
    """The K/V rows of a qkv projection as a dense compute-dtype matrix."""
    weight = qkv_proj.weight
    if weight.dtype in _DFLASH_DENSE:
        return weight[q_size:]
    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)
    eye = torch.eye(
        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device
    )
    out = qkv_proj(eye)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, q_size:].t().contiguous()


@support_torch_compile
class DFlashQwen3Model(nn.Module):'''

SLICE_OLD = """        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)"""
SLICE_NEW = """        self._kv_source_attn = layers_attn
        if layers_attn[0].qkv_proj.weight.dtype not in _DFLASH_DENSE:
            # Deferred: a quantized qkv_proj cannot be read until its quant method has processed
            # the weights, which happens after load_weights returns.
            self._fused_kv_weight = None
        else:
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)"""

LAZY_OLD = """        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )"""
LAZY_NEW = """        if self._fused_kv_weight is None:
            self._fused_kv_weight = torch.cat(
                [
                    _dflash_kv_weight_rows(a.qkv_proj, a.q_size)
                    for a in self._kv_source_attn
                ],
                dim=0,
            )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )"""


def main() -> None:
    apply(F, HELPER_OLD, HELPER_NEW, "_dflash_kv_weight_rows", "dflash: quantized fused KV helper")
    apply(F, SLICE_OLD, SLICE_NEW, "self._kv_source_attn", "dflash: defer fused KV build")
    apply(F, LAZY_OLD, LAZY_NEW, "if self._fused_kv_weight is None",
          "dflash: materialize fused KV on first use")


if __name__ == "__main__":
    main()
