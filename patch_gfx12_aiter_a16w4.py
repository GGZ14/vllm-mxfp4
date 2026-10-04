#!/usr/bin/env python3
"""Serve MXFP4 MoE experts on gfx12 (RDNA4, R9700) through AITER's Triton a16w4 lane
(Mxfp4MoeBackend.AITER_MXFP4_BF16 -> AiterW4A16ExpertsMonolithic -> aiter moe_gemm_a16w4).

Measured on one R9700 (MOE-GFX1201.md): correct vs a torch reference (rel err ~3e-3), ~57 us per
MoE layer at M=1 vs ~370 us for TRITON_UNFUSED and ~3.6 ms for EMULATION.

Every change is gated on `on_gfx12x()` (gfx12 minus the CDNA-classified gfx1250), so gfx950 /
gfx1250 / CUDA take exactly the code paths they took before:
  1. AiterW4A16ExpertsMonolithic._supports_current_device also accepts gfx12, for SILU only (gpt-oss
     SWIGLUOAI checkpoints are already interleaved and keep their previous lanes on gfx12). The CK
     AiterExperts already declines MXFP4 on anything but gfx950 (rocm_aiter_moe._supports_quant_scheme).
  2. convert_gpt_oss_weight_to_mxfp4_moe_kernel_format: on gfx12, AITER_MXFP4_BF16 takes the
     Triton/StridedLayout prep (as gfx1250 does) instead of the CK shuffle_weight_a16w4 one, and
     interleaves the w13 rows (+ scales, bias) gate/up along N: aiter's fused `_swiglu` reads
     column pairs (2j = gate, 2j+1 = up), the Quark/HF checkpoint stores [gate; up].
  3. quark_moe: on gfx12 AITER_MXFP4_BF16 carries triton_kernels Tensors + PrecisionConfigs like
     the TRITON lanes do (store on the layer, scales in precision configs), and keeps the experts
     class the selector validated instead of backend_to_kernel_cls()[0] (= CK AiterExperts).
  4. aiter_triton_kernel_w4a16_moe_forward: SILU with no configured clamp passes limit=None (the
     stock fallback clamps at 7.0, a gpt-oss constant), and installs a gfx1201 tile table for
     moe_gemm_a16w4 (cold-cache sweeps; MOE-GFX1201.md).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
MOE = SP / "vllm/model_executor/layers/fused_moe"
AITER_W4 = MOE / "experts/aiter_mxfp4_w4a8_moe.py"
ORACLE = MOE / "oracle/mxfp4.py"
QUARK = SP / "vllm/model_executor/layers/quantization/quark/quark_moe.py"

# ---------------------------------------------------------------- 1. device gate
apply(
    AITER_W4,
    "        from vllm.platforms.rocm import on_gfx950, on_gfx1250\n"
    "\n"
    "        return on_gfx950() or on_gfx1250()\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_no_act_and_mul() -> bool:\n"
    "        return False\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_quant_scheme(\n"
    "        weight_key: QuantKey | None,\n"
    "        activation_key: QuantKey | None,\n"
    "    ) -> bool:\n"
    "        return (weight_key, activation_key) == (kMxfp4Static, None)\n",
    "        from vllm.platforms.rocm import on_gfx950, on_gfx1250\n"
    "        # radiance: gfx12 (RDNA4) runs the a16w4 Triton lane (StridedLayout weights)\n"
    "        from vllm.platforms.rocm import on_gfx12x\n"
    "\n"
    "        return on_gfx950() or on_gfx1250() or on_gfx12x()\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_no_act_and_mul() -> bool:\n"
    "        return False\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_quant_scheme(\n"
    "        weight_key: QuantKey | None,\n"
    "        activation_key: QuantKey | None,\n"
    "    ) -> bool:\n"
    "        return (weight_key, activation_key) == (kMxfp4Static, None)\n",
    "radiance: gfx12 (RDNA4) runs the a16w4 Triton lane",
    "aiter a16w4: AiterW4A16ExpertsMonolithic accepts gfx12",
)

apply(
    AITER_W4,
    "    @staticmethod\n"
    "    def _supports_activation(activation: MoEActivation) -> bool:\n"
    "        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)\n",
    "    @staticmethod\n"
    "    def _supports_activation(activation: MoEActivation) -> bool:\n"
    "        # radiance: gfx12 serves only plain SiLU here - its [gate; up] checkpoints are\n"
    "        # interleaved at load; SWIGLUOAI (gpt-oss) keeps the lanes it had before.\n"
    "        from vllm.platforms import current_platform\n"
    "\n"
    "        if current_platform.is_rocm():\n"
    "            from vllm.platforms.rocm import on_gfx12x\n"
    "\n"
    "            if on_gfx12x():\n"
    "                return activation == MoEActivation.SILU\n"
    "        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)\n",
    "radiance: gfx12 serves only plain SiLU here",
    "aiter a16w4: gfx12 restricted to SILU",
)

# ---------------------------------------------------------------- 4. SILU limit + tile table
TILES = '''
_RADIANCE_A16W4_TILES = False


def _radiance_gfx12_a16w4_cfg(orig, m, n, k, routing_data):
    """radiance: gfx1201 tiles for aiter moe_gemm_a16w4 (R9700 cold-cache sweeps on the
    Qwen3.5-35B-A3B shapes: gate_up N=1024 K=2048, down N=2048 K=512). m = tokens * top_k.
    "long K" (K >= 1024) is the gate_up GEMM; the down GEMM keeps aiter's table at block_m 16/32
    except the large-m case. bm >= 64 replaces aiter's bn512/w8/bk256, which is 5x slower here."""
    c = orig(m, n, k, routing_data)
    bm = c["block_m"]
    upd = {}
    if bm == 16:
        if k >= 1024:
            if m <= 8:
                upd = dict(block_n=64, num_warps=4)
            elif m <= 16:
                upd = dict(block_n=64, num_warps=4, block_k=256, num_stages=2, split_k=4)
            elif m <= 128:
                upd = dict(block_n=32, num_warps=2, block_k=128, num_stages=2, split_k=4)
            elif m <= 384:
                upd = dict(block_n=32, num_warps=2, block_k=256, num_stages=1, split_k=2)
            else:
                upd = dict(block_n=32, num_warps=2, block_k=512, num_stages=1, split_k=1)
        elif m >= 256:
            upd = dict(block_n=256, num_warps=8, block_k=256, num_stages=2)
    elif bm == 64:
        upd = dict(block_n=256, num_warps=8, block_k=128, num_stages=1, split_k=1)
    elif bm >= 128:
        upd = dict(block_n=128, num_warps=8, block_k=128, num_stages=1, split_k=1)
    sk = upd.get("split_k", 1)
    if sk > 1 and k % (upd.get("block_k", c["block_k"]) * sk) != 0:
        upd["split_k"] = 1
    c.update(upd)
    return c


def _radiance_install_gfx12_a16w4_tiles():
    global _RADIANCE_A16W4_TILES
    if _RADIANCE_A16W4_TILES:
        return
    _RADIANCE_A16W4_TILES = True
    import functools

    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as _a16w4

    _a16w4.get_kernel_config = functools.partial(
        _radiance_gfx12_a16w4_cfg, _a16w4.get_kernel_config
    )


def aiter_triton_kernel_w4a16_moe_forward(
'''
apply(
    AITER_W4,
    "\ndef aiter_triton_kernel_w4a16_moe_forward(\n",
    TILES,
    "def _radiance_gfx12_a16w4_cfg(",
    "aiter a16w4: gfx1201 tile table for moe_gemm_a16w4",
)

apply(
    AITER_W4,
    "    swiglu_limit = (\n"
    "        quant_config.gemm1_clamp_limit\n"
    "        if quant_config.gemm1_clamp_limit is not None\n"
    "        else 7.0\n"
    "    )\n"
    "\n"
    "    # SILU on gfx1250: use the verified a8w4 kernel (dynamic MXFP8); a16w4 faults.\n",
    "    swiglu_limit = (\n"
    "        quant_config.gemm1_clamp_limit\n"
    "        if quant_config.gemm1_clamp_limit is not None\n"
    "        else 7.0\n"
    "    )\n"
    "    # radiance: gfx12 - plain SiLU has no clamp unless the model configures one; tiles\n"
    "    from vllm.platforms.rocm import on_gfx12x as _radiance_on_gfx12x\n"
    "\n"
    "    if _radiance_on_gfx12x():\n"
    "        if (\n"
    "            activation == MoEActivation.SILU\n"
    "            and quant_config.gemm1_clamp_limit is None\n"
    "        ):\n"
    "            swiglu_limit = None\n"
    "        _radiance_install_gfx12_a16w4_tiles()\n"
    "\n"
    "    # SILU on gfx1250: use the verified a8w4 kernel (dynamic MXFP8); a16w4 faults.\n",
    "radiance: gfx12 - plain SiLU has no clamp",
    "aiter a16w4: SILU limit=None + tiles on gfx12",
)

# ---------------------------------------------------------------- 2. weight prep
apply(
    ORACLE,
    "    elif mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:\n"
    "        from vllm._aiter_ops import rocm_aiter_ops\n"
    "\n"
    "        if w13_bias is not None:\n"
    "            w13_bias = w13_bias.data.to(torch.float32)\n",
    "    elif (\n"
    "        mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16\n"
    "        and current_platform.is_rocm()\n"
    "        and __import__(\"vllm.platforms.rocm\", fromlist=[\"on_gfx12x\"]).on_gfx12x()\n"
    "    ):\n"
    "        # radiance: gfx12 a16w4 Triton lane - StridedLayout (the gfx1250 prep, not the CK\n"
    "        # shuffle) plus a gate/up row interleave along N: aiter's fused _swiglu reads\n"
    "        # column pairs (2j = gate, 2j+1 = up); the checkpoint stores [gate; up].\n"
    "        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig\n"
    "\n"
    "        e, n, k = w13_weight.shape\n"
    "\n"
    "        def _gu_interleave(t):\n"
    "            t = t.data if hasattr(t, \"data\") else t\n"
    "            return (\n"
    "                t.reshape(e, 2, n // 2, *t.shape[2:])\n"
    "                .transpose(1, 2)\n"
    "                .reshape(t.shape)\n"
    "                .contiguous()\n"
    "            )\n"
    "\n"
    "        if w13_bias is not None:\n"
    "            w13_bias = _gu_interleave(w13_bias.to(torch.float32))\n"
    "        if w2_bias is not None:\n"
    "            w2_bias = w2_bias.to(torch.float32)\n"
    "        w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(\n"
    "            _gu_interleave(w13_weight), _gu_interleave(w13_weight_scale)\n"
    "        )\n"
    "        w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(\n"
    "            w2_weight.data, w2_weight_scale.data\n"
    "        )\n"
    "        w13_precision_config = PrecisionConfig(\n"
    "            weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)\n"
    "        )\n"
    "        w2_precision_config = PrecisionConfig(\n"
    "            weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)\n"
    "        )\n"
    "        del layer.w13_weight\n"
    "        del layer.w2_weight\n"
    "        del layer.w13_weight_scale\n"
    "        del layer.w2_weight_scale\n"
    "        return (\n"
    "            w13_weight,\n"
    "            w2_weight,\n"
    "            w13_precision_config,\n"
    "            w2_precision_config,\n"
    "            w13_bias,\n"
    "            w2_bias,\n"
    "        )\n"
    "\n"
    "    elif mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:\n"
    "        from vllm._aiter_ops import rocm_aiter_ops\n"
    "\n"
    "        if w13_bias is not None:\n"
    "            w13_bias = w13_bias.data.to(torch.float32)\n",
    "radiance: gfx12 a16w4 Triton lane - StridedLayout",
    "mxfp4 oracle: gfx12 AITER_MXFP4_BF16 weight prep (StridedLayout + gate/up interleave)",
)

# ---------------------------------------------------------------- 3. quark: triton-style storage
_GFX12_AITER = (
    "(\n"
    "            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16\n"
    "            and __import__(\"vllm.platforms.rocm\", fromlist=[\"on_gfx12x\"]).on_gfx12x()\n"
    "        )"
)
apply(
    QUARK,
    "        # Handle weight/scale assignment based on backend type\n"
    "        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (\n"
    "            Mxfp4MoeBackend.AITER_MXFP4_FP8,\n"
    "        ):\n",
    "        # Handle weight/scale assignment based on backend type\n"
    "        # radiance: gfx12 AITER_MXFP4_BF16 returns triton tensors + PrecisionConfigs\n"
    "        if (\n"
    "            self.mxfp4_backend in TRITON_BACKENDS\n"
    "            or self.mxfp4_backend in (Mxfp4MoeBackend.AITER_MXFP4_FP8,)\n"
    "            or " + _GFX12_AITER + "\n"
    "        ):\n",
    "radiance: gfx12 AITER_MXFP4_BF16 returns triton tensors",
    "quark moe: gfx12 AITER_MXFP4_BF16 stores triton tensors (setup)",
)
apply(
    QUARK,
    "            # Determine scale source based on backend type\n"
    "            if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (\n"
    "                Mxfp4MoeBackend.AITER_MXFP4_FP8,\n"
    "            ):\n",
    "            # Determine scale source based on backend type\n"
    "            # radiance: gfx12 AITER_MXFP4_BF16 scales live in PrecisionConfigs\n"
    "            if (\n"
    "                self.mxfp4_backend in TRITON_BACKENDS\n"
    "                or self.mxfp4_backend in (Mxfp4MoeBackend.AITER_MXFP4_FP8,)\n"
    "                or " + _GFX12_AITER.replace("\n        ", "\n            ") + "\n"
    "            ):\n",
    "radiance: gfx12 AITER_MXFP4_BF16 scales live in PrecisionConfigs",
    "quark moe: gfx12 AITER_MXFP4_BF16 precision-config scales (quant config)",
)

# backend_to_kernel_cls(AITER_MXFP4_BF16)[0] is the CK AiterExperts, which cannot run MXFP4 on gfx12;
# keep the experts class the selector actually validated (AiterW4A16ExpertsMonolithic).
apply(
    QUARK,
    "        self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]\n",
    "        # radiance: on gfx12 keep the selector's validated class for AITER_MXFP4_BF16 (list\n"
    "        # element [0] is the CK AiterExperts, which declines MXFP4 on gfx12)\n"
    "        if not (\n"
    "            self.experts_cls is not None\n"
    "            and " + _GFX12_AITER.replace("\n        ", "\n            ") + "\n"
    "        ):\n"
    "            self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]\n",
    "radiance: on gfx12 keep the selector's validated class",
    "quark moe: gfx12 AITER_MXFP4_BF16 keeps AiterW4A16ExpertsMonolithic",
)
