#!/usr/bin/env python3
"""Let DFlash2's fused KV projection work with any quantized qkv_proj, not just fp8.

DFlash2 precomputes the drafter's context K/V for every layer in ONE GEMM, which means reaching
past the qkv_proj module and slicing its raw `.weight`. That slice is only valid when the weight
is a dense compute-dtype matrix, so upstream special-cases fp8: it recovers the dense rows by
pushing an identity matrix through the layer (`qkv_proj(eye)`), which runs whatever apply() the
quant method installed and hands back a bf16 result.

The trick is scheme-agnostic, but the guard is a hardcoded fp8 dtype tuple. An MXFP4 checkpoint
stores `weight` as packed uint8 of shape [N, K/2], falls into the dense branch, and dies in
precompute_and_store_context_kv with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (8192x5120 and 2560x5120)

where 2560 is K/2 -- packed nibbles being read as a bf16 matrix. This patch inverts both tests:
take the direct slice only for real float dtypes, and use the identity trick for everything else.

Exactness: the identity's entries are 0.0 and 1.0, both exactly representable in e2m1 and in
e4m3, and a block of one 1.0 with 31 zeros gets scale 2^-2 so the element encodes as 4.0 with no
rounding. So the recovered rows equal the dequantized weight rather than approximating it.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
DF = SP / "vllm" / "model_executor" / "models" / "qwen3_dflash.py"

ANCHOR_ROWS = '''    weight = qkv_proj.weight
    if weight.dtype not in _DFLASH_FP8:
        return weight[q_size:]
'''
NEW_ROWS = '''    weight = qkv_proj.weight
    # radiance (patch_dflash_mxfp4_kv.py): the direct slice is valid only for a dense
    # compute-dtype weight. Upstream tested "is it fp8"; an MXFP4 weight is packed uint8 of shape
    # [N, K/2] and passed that test, then blew up as a 2560-wide bf16 matrix. Test for the dense
    # case instead, so every quantized scheme takes the identity path below.
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight[q_size:]
'''

ANCHOR_DEFER = '''        if layers_attn[0].qkv_proj.weight.dtype in _DFLASH_FP8:
'''
NEW_DEFER = '''        # radiance (patch_dflash_mxfp4_kv.py): defer for ANY quantized weight, not just fp8 --
        # the dense rows cannot be read until the quant method has processed the weights.
        if layers_attn[0].qkv_proj.weight.dtype not in (
            torch.bfloat16, torch.float16, torch.float32
        ):
'''


def main():
    apply(DF, ANCHOR_ROWS, NEW_ROWS, "the direct slice is valid only for a dense",
          "dflash: fused-KV rows for any quantized qkv_proj")
    apply(DF, ANCHOR_DEFER, NEW_DEFER, "defer for ANY quantized weight",
          "dflash: defer fused-KV build for any quantized qkv_proj")


if __name__ == "__main__":
    main()
