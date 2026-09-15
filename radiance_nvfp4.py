"""NVFP4 checkpoints on gfx1201: requantize to MXFP4 at load, serve on the radiance W4A8 kernel.

What the AMD ROCm blog "NVFP4 to MXFP4 online requantization" does for CDNA4 inside SGLang, done
here for RDNA4 inside vLLM's compressed-tensors loader. An NVFP4 linear (e2m1 codes, e4m3 scale per
16, fp32 scale per tensor -- the unsloth/Qwen3.8-27B-NVFP4 layout) is loaded exactly as the
checkpoint stores it, then in process_weights_after_loading:

  1. dequantized to fp32, PER PARTITION of a merged linear (gate_up_proj carries two global scales
     and the stock scheme collapses them with .max(), which is a real accuracy loss);
  2. requantized to e2m1 + e8m0 per 32 (RADIANCE_NVFP4_EXP picks the block exponent: ocp = the
     OCP rule floor(log2(amax)) - 2, which clips the top of the block; noclip = ceil(log2(amax/6));
     mse = whichever of noclip / noclip-1 has the lower squared error per block, the default);
  3. handed to the MXFP4 kernel plugin exactly as a Quark/compressed-tensors MXFP4 layer would be:
     layer.weight [N, K/2] uint8 (low nibble = even k), layer.weight_scale [N, K/32] uint8 e8m0.
     With RADIANCE_MXFP4_W4A8=1 that plugin is radiance_mxfp4's fp8-WMMA GEMM (fp8 activations,
     folded e8m0, fragment-order weights, decode band, fp8 stream) -- the same kernel that serves
     the AMD Quark MXFP4 checkpoint.

The step is lossy in principle (32-wide power-of-two blocks are coarser than 16-wide e4m3 ones);
the per-layer relative error is logged at load and the serving gate is GSM8K, as for every other
kernel change here. RADIANCE_NVFP4_FP8_LAYERS=mxfp4 additionally requantizes the checkpoint's
FP8 per-channel linears (attention, GDN projections, the last MLPs) to MXFP4 so the whole model
runs on one kernel; lm_head always stays FP8. This is the DEFAULT: =fp8 keeps them on vLLM's FP8
path (torch._scaled_mm -> hipBLASLt), which on gfx1201 wedged the GPU (driver reset) twice under
sustained 8-way concurrency, 4-8 minutes into GSM8K, while the all-MXFP4 form ran the same gate
clean and scored the same 97.40%. Measured 2026-09-15; treat =fp8 as diagnostic only.

Nothing here runs unless patch_nvfp4_mxfp4.py is applied and RADIANCE_NVFP4_MXFP4=1.
"""
import os
import sys
import time

import torch
from torch.nn import Parameter

EXP_MODE = os.environ.get("RADIANCE_NVFP4_EXP", "mse")
FP8_LAYERS = os.environ.get("RADIANCE_NVFP4_FP8_LAYERS", "mxfp4")   # mxfp4 | fp8
# Regex (re.search on the vLLM module prefix) of UNQUANTIZED bf16 linears to requantize to MXFP4 as
# well. Default "in_proj_ba": the checkpoint leaves the GDN a/b gate projections in bf16, and the GDN
# in_proj merge (radiance_gdnmerge) only fuses a layer whose qkvz AND ba sides are both on the
# radiance kernel. With it all 48 GDN layers merge (96 launches/forward gone), the fp8 stream covers
# the whole model, decode 23.9 -> 22.2 ms/step, GSM8K 97.60 (2026-09-15). Empty = leave them bf16.
BF16_LAYERS = os.environ.get("RADIANCE_NVFP4_BF16_LAYERS", "in_proj_ba").strip()
# lm_head: "bf16" (default) dequantizes the checkpoint's FP8 per-channel lm_head to bf16 at load so it
# runs on the plain bf16 GEMM like every other unit here (and the int2 draft/verify heads take their
# native path); "fp8" leaves it on vLLM's FP8 path (torch._scaled_mm -> hipBLASLt). With the FP8
# linears requantized, that lm_head was the LAST hipBLASLt fp8 call left in the model, and the
# serve still wedged the GPU at 8-way concurrency with it in place (2026-09-15, reset #4).
LMHEAD = os.environ.get("RADIANCE_NVFP4_LMHEAD", "bf16")
LOG_EVERY = os.environ.get("RADIANCE_NVFP4_LOG", "1") == "1"
ROW_CHUNK = 4096            # rows per requant pass: bounds the fp32 transient at ~1.3 GiB for K=17408

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _log(msg):
    sys.stderr.write(f"[radiance.nvfp4] {msg}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------------------------
# NVFP4 -> fp32
# --------------------------------------------------------------------------------------------
def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 -> [N, K] fp32. Low nibble is element 2i (vLLM's break_fp4_bytes order)."""
    n = packed.shape[0]
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(n, -1)
    grid = _E2M1.to(packed.device)
    mag = grid[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def dequant_nvfp4(packed: torch.Tensor, scale_e4m3: torch.Tensor, global_divisor: float) -> torch.Tensor:
    """W[n, k] = e2m1 * fp32(scale_e4m3[n, k//16]) / global_divisor.

    global_divisor is the checkpoint's weight_global_scale AS STORED (compressed-tensors stores
    (6 * 448) / amax, a divisor; vLLM's own scheme inverts it the same way)."""
    n = packed.shape[0]
    k = packed.shape[1] * 2
    v = unpack_e2m1(packed).reshape(n, k // 16, 16)
    s = scale_e4m3.to(torch.float32) / float(global_divisor)
    return (v * s.unsqueeze(-1)).reshape(n, k)


# --------------------------------------------------------------------------------------------
# fp32 -> MXFP4 (e2m1 + e8m0 per 32)
# --------------------------------------------------------------------------------------------
def e2m1_index(a: torch.Tensor) -> torch.Tensor:
    """|value| (already divided by the block scale, clamped to 6) -> code index 0..7 with
    round-to-nearest, ties to the EVEN code, matching the argmin+tie rule of the offline
    quantizer (quantize_dflash_mxfp4.py). torch.round is half-to-even on the integers it sees:
      a < 2      : step 0.5 -> round(2a)        (indices 0..4)
      2 <= a < 4 : step 1   -> round(a) + 2     (indices 4..6)
      a >= 4     : step 2   -> round(a/2) + 4   (indices 6..7)"""
    a = a.clamp(max=6.0)
    lo = torch.round(a * 2.0)
    mid = torch.round(a) + 2.0
    hi = torch.round(a * 0.5) + 4.0
    idx = torch.where(a < 2.0, lo, torch.where(a < 4.0, mid, hi))
    return idx.to(torch.uint8)


def _block_exponent(amax: torch.Tensor, mode: str) -> torch.Tensor:
    """Unbiased block exponent e (scale = 2^e) for each 32-block from its amax. amax == 0 -> 0."""
    safe = amax.clamp(min=torch.finfo(torch.float32).tiny)
    if mode == "ocp":
        e = torch.floor(torch.log2(safe)) - 2.0          # amax lands in [4, 8): values above 6 clip
    else:
        e = torch.ceil(torch.log2(safe / 6.0))           # amax lands in (3, 6]: never clips
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    return e.clamp(-127.0, 127.0)


def _encode_blocks(wb: torch.Tensor, e: torch.Tensor):
    """wb [R, G, 32] fp32, e [R, G] exponent -> (idx uint8 [R, G, 32], sq err [R, G])."""
    scale = torch.exp2(e).unsqueeze(-1)
    v = wb / scale
    idx = e2m1_index(v.abs())
    grid = _E2M1.to(wb.device)
    q = grid[idx.long()] * torch.sign(v) * scale
    err = ((q - wb) ** 2).sum(-1)
    return idx, err


def quant_mxfp4(w: torch.Tensor, mode: str = EXP_MODE):
    """[N, K] fp32 -> (packed [N, K/2] uint8 low-nibble-first, e8m0 [N, K/32] uint8, sq err)."""
    n, k = w.shape
    assert k % 32 == 0, k
    wb = w.float().reshape(n, k // 32, 32)
    amax = wb.abs().amax(-1)
    if mode == "mse":
        e0 = _block_exponent(amax, "noclip")
        i0, r0 = _encode_blocks(wb, e0)
        e1 = e0 - 1.0                                   # finer grid, top of block clips at 6
        i1, r1 = _encode_blocks(wb, e1)
        take1 = r1 < r0
        e = torch.where(take1, e1, e0)
        idx = torch.where(take1.unsqueeze(-1), i1, i0)
        err = torch.where(take1, r1, r0)
    else:
        e = _block_exponent(amax, mode)
        idx, err = _encode_blocks(wb, e)
    sign = (torch.signbit(wb) & (idx != 0)).to(torch.uint8) << 3   # -0 stays code 0
    code = (idx | sign).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    e8m0 = (e + 127.0).to(torch.uint8).contiguous()
    return packed, e8m0, err.sum()


def dequant_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    n = packed.shape[0]
    k = packed.shape[1] * 2
    v = unpack_e2m1(packed).reshape(n, k // 32, 32)
    return (v * torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)).reshape(n, k)


def requant_rows(w: torch.Tensor, mode: str = EXP_MODE):
    """Chunked quant_mxfp4 over rows; returns packed, e8m0, relative RMS error of the requant."""
    packed, scales, err, ssq = [], [], 0.0, 0.0
    for a in range(0, w.shape[0], ROW_CHUNK):
        wc = w[a:a + ROW_CHUNK].float()
        p, s, e = quant_mxfp4(wc, mode)
        packed.append(p)
        scales.append(s)
        err += float(e)
        ssq += float((wc ** 2).sum())
    rel = (err / ssq) ** 0.5 if ssq > 0 else 0.0
    return torch.cat(packed), torch.cat(scales), rel


# --------------------------------------------------------------------------------------------
# The compressed-tensors schemes (built lazily: importing vLLM at module scope would drag HIP in)
# --------------------------------------------------------------------------------------------
_STATS = {"nvfp4_layers": 0, "fp8_layers": 0, "bf16_layers": 0, "worst_rel": 0.0, "seconds": 0.0}
_CLS = {}


def _finish_layer(layer, packed, e8m0, kernel, rel, kind, t0):
    layer.weight = Parameter(packed, requires_grad=False)
    layer.weight_scale = Parameter(e8m0, requires_grad=False)
    _STATS[kind] += 1
    _STATS["worst_rel"] = max(_STATS["worst_rel"], rel)
    _STATS["seconds"] += time.time() - t0
    if LOG_EVERY:
        _log(f"{kind} N={packed.shape[0]} K={packed.shape[1] * 2} -> MXFP4 ({EXP_MODE}) "
             f"requant relRMS {rel:.4f}  [{_STATS['nvfp4_layers'] + _STATS['fp8_layers'] + _STATS['bf16_layers']} layers, "
             f"{_STATS['seconds']:.1f}s total]")
    kernel.process_weights_after_loading(layer)


def _make_classes():
    from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsScheme,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
    from vllm.model_executor.parameter import (
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PerTensorScaleParameter,
    )

    class RadianceNvfp4ToMxfp4(CompressedTensorsScheme):
        """nvfp4-pack-quantized weights -> MXFP4 at load -> the MXFP4 kernel plugin."""

        def __init__(self):
            self.group_size = 16
            self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            # Identical to CompressedTensorsW4A4Fp4.create_weights so the checkpoint loads
            # untouched; input_global_scale is declared only so the loader has a home for it.
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight_packed", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition // 2, dtype=torch.uint8),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("weight_global_scale", PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader))
            layer.register_parameter("weight_scale", GroupQuantScaleParameter(
                data=torch.empty(n, input_size_per_partition // self.group_size,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("input_global_scale", PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            t0 = time.time()
            packed_nv = layer.weight_packed.data
            scale_nv = layer.weight_scale.data
            gs = layer.weight_global_scale.data.float().tolist()
            widths = list(layer.logical_widths)
            if len(gs) != len(widths):
                raise RuntimeError(f"[radiance.nvfp4] {len(gs)} global scales for {len(widths)} partitions")
            packed, scales, errs, ssqs = [], [], [], []
            r0 = 0
            for width, g in zip(widths, gs):
                w = dequant_nvfp4(packed_nv[r0:r0 + width], scale_nv[r0:r0 + width], g)
                p, s, rel = requant_rows(w)
                packed.append(p)
                scales.append(s)
                ssq = float((w.float() ** 2).sum())
                errs.append((rel ** 2) * ssq)
                ssqs.append(ssq)
                del w
                r0 += width
            rel = (sum(errs) / max(sum(ssqs), 1e-30)) ** 0.5
            del layer.weight_packed, layer.weight_global_scale, layer.input_global_scale
            _finish_layer(layer, torch.cat(packed), torch.cat(scales), self.kernel, rel,
                          "nvfp4_layers", t0)

        def apply_weights(self, layer, x, bias=None):
            return self.kernel.apply_weights(layer, x, bias)

    class RadianceFp8ChannelToMxfp4(CompressedTensorsScheme):
        """float-quantized W8 per-channel (+ dynamic per-token A8) -> MXFP4 at load. Opt-in via
        RADIANCE_NVFP4_FP8_LAYERS=mxfp4; the checkpoint author kept these at 8 bits on purpose,
        so this trades accuracy for the single-kernel fp8 stream. Gate it with GSM8K."""

        def __init__(self):
            self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition, dtype=torch.float8_e4m3fn),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("weight_scale", ChannelQuantScaleParameter(
                data=torch.empty(n, 1, dtype=torch.float32), output_dim=0,
                weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            t0 = time.time()
            w8 = layer.weight.data
            s = layer.weight_scale.data.float()
            packed, scales, errs, ssqs = [], [], [], []
            for a in range(0, w8.shape[0], ROW_CHUNK):
                w = w8[a:a + ROW_CHUNK].to(torch.float32) * s[a:a + ROW_CHUNK]
                p, sc, e = quant_mxfp4(w)
                packed.append(p)
                scales.append(sc)
                errs.append(float(e))
                ssqs.append(float((w ** 2).sum()))
            rel = (sum(errs) / max(sum(ssqs), 1e-30)) ** 0.5
            del layer.weight
            _finish_layer(layer, torch.cat(packed), torch.cat(scales), self.kernel, rel,
                          "fp8_layers", t0)

        def apply_weights(self, layer, x, bias=None):
            return self.kernel.apply_weights(layer, x, bias)

    class RadianceBf16ToMxfp4(CompressedTensorsScheme):
        """An unquantized bf16 linear (no compressed-tensors group matches it) -> MXFP4 at load.
        Opt-in per regex via RADIANCE_NVFP4_BF16_LAYERS."""

        def __init__(self):
            self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition, dtype=params_dtype),
                input_dim=1, output_dim=0, weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            t0 = time.time()
            packed, scales, rel = requant_rows(layer.weight.data)
            del layer.weight
            _finish_layer(layer, packed, scales, self.kernel, rel, "bf16_layers", t0)

        def apply_weights(self, layer, x, bias=None):
            return self.kernel.apply_weights(layer, x, bias)

    class RadianceFp8ChannelToBf16(CompressedTensorsScheme):
        """float-quantized W8 per-channel -> plain bf16 weight at load (lm_head)."""

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition, dtype=torch.float8_e4m3fn),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("weight_scale", ChannelQuantScaleParameter(
                data=torch.empty(n, 1, dtype=torch.float32), output_dim=0,
                weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            w8 = layer.weight.data
            s = layer.weight_scale.data.float()
            out = torch.empty(w8.shape, dtype=layer.params_dtype, device=w8.device)
            for a in range(0, w8.shape[0], ROW_CHUNK):
                out[a:a + ROW_CHUNK] = (w8[a:a + ROW_CHUNK].to(torch.float32) * s[a:a + ROW_CHUNK]).to(out.dtype)
            del layer.weight, layer.weight_scale
            layer.weight = Parameter(out, requires_grad=False)
            _STATS["bf16_heads"] = _STATS.get("bf16_heads", 0) + 1
            _log(f"fp8 lm_head N={out.shape[0]} K={out.shape[1]} -> {out.dtype} at load")

        def apply_weights(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    return RadianceNvfp4ToMxfp4, RadianceFp8ChannelToMxfp4, RadianceBf16ToMxfp4, RadianceFp8ChannelToBf16


def scheme_class():
    """The NVFP4 scheme class, or None if it cannot be built (never blocks model load)."""
    if "nv" not in _CLS:
        try:
            _CLS["nv"], _CLS["fp8"], _CLS["bf16"], _CLS["head"] = _make_classes()
        except Exception as e:                                        # noqa: BLE001
            _log(f"scheme unavailable, stock path: {e!r}")
            _CLS["nv"] = _CLS["fp8"] = _CLS["bf16"] = _CLS["head"] = None
    return _CLS["nv"]


def bf16_scheme_class(layer_name):
    """The bf16 -> MXFP4 scheme for an unquantized linear whose prefix matches
    RADIANCE_NVFP4_BF16_LAYERS, else None."""
    import re
    if not BF16_LAYERS or not layer_name or not re.search(BF16_LAYERS, layer_name):
        return None
    scheme_class()
    return _CLS["bf16"]


def fp8_scheme_class(layer_name):
    """The FP8-per-channel -> MXFP4 scheme when RADIANCE_NVFP4_FP8_LAYERS=mxfp4, else None.
    lm_head is never converted (its N exceeds the decode kernel's scratch and the draft/verify
    heads read it as an 8-bit matrix)."""
    if layer_name and layer_name.endswith("lm_head"):
        if LMHEAD == "bf16":
            scheme_class()
            return _CLS["head"]
        return None
    if FP8_LAYERS != "mxfp4":
        return None
    scheme_class()
    return _CLS["fp8"]


def report():
    _log(f"converted {_STATS['nvfp4_layers']} NVFP4 + {_STATS['fp8_layers']} FP8 + {_STATS['bf16_layers']} bf16 linears to MXFP4 "
         f"(exp rule {EXP_MODE}), worst requant relRMS {_STATS['worst_rel']:.4f}, "
         f"{_STATS['seconds']:.1f}s")
