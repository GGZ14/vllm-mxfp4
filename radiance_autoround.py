"""AutoRound int4 (group-128, symmetric) W4A8 for radiance on gfx1201.

The checkpoint is `auto_round:auto_gptq`: qweight [K/8, N] int32 with eight 4-bit codes packed
along K, scales [K/128, N] fp16, qzeros [K/128, N/8] int32, and no g_idx (desc_act is off, so
groups are contiguous along K).

Two properties of the format drive everything here, and both are ASSERTED at load rather than
assumed:

  1. qzeros is uniformly 0x77777777. GPTQ stores zero-1, so nibble 7 means zero = 8 for every
     group -- the checkpoint is symmetric. Code c therefore means the integer (c - 8) in [-8, 7],
     and all sixteen of those are exactly representable in e4m3, so the zero point folds into the
     kernel's unpack table and never enters the matmul. If a future checkpoint is asymmetric this
     kernel is WRONG for it, hence the check.
  2. group_size is 128 and K per partition stays a multiple of 128 under TP, so the kernel's slab
     structure lines up with the groups.

The 99 `linear_attn.in_proj_a` / `in_proj_b` modules are declared bits=16 in extra_config and are
routed to vLLM's unquantized linear method.

Enable with RADIANCE_AUTOROUND=1.
"""
import os
import re
import sys

import torch

# Only what registration itself needs may be imported at module scope.
#
# patch_autoround.py imports this module from the END of vLLM's quantization package __init__,
# which is the one point that is both after `register_quantization_config` exists and before
# ModelConfig resolves the checkpoint's quant_method. At that moment `vllm.config` is still
# partially initialized, so pulling in vllm.model_executor.layers.linear here raises
# ImportError("cannot import name 'get_current_vllm_config' ... circular import"). Everything that
# reaches into the model-executor layers is therefore deferred to first use, and the linear method
# is built by a factory rather than declared at import time -- the same shape radiance_mxfp4.py
# uses for its kernel class, for the same reason.
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

import radiance_autoround_kernel as _ext

ENABLED = os.environ.get("RADIANCE_AUTOROUND", "0") == "1"
GROUP = 128
PACK = 8                       # int4 codes per int32

# Split-K scratch for the decode kernel, sized like the MXFP4 one: KS(4) x maxM(64) x maxN(32768).
# Registered once per worker process and kept alive for the lifetime of the process. It is
# allocated HERE rather than lazily inside the kernel because a lazy hipMalloc from launch() lands
# inside CUDA-graph capture whenever the torch.compile cache is warm, where it is illegal.
_DEC_KS, _DEC_MAX_M, _DEC_MAX_N = 4, 64, 32768
_scratch = [None, None]
_scratch_ready = [False]


def _ensure_scratch(device):
    if _scratch_ready[0]:
        return
    # Rows must cover the decode band; sized from the env (mirrors radiance_mxfp4.py) so the
    # default 64 allocates exactly what it always has and only a 16-concurrent serve
    # (RADIANCE_AR_DECODE_MAX_M=128) pays the extra 32 MiB.
    _scratch[0] = torch.empty(
        _DEC_KS * max(_DEC_MAX_M,
                      int(os.environ.get("RADIANCE_AR_DECODE_MAX_M", "0"))) * _DEC_MAX_N,
        dtype=torch.float32,
                              device=device)
    _scratch[1] = torch.zeros(_DEC_MAX_N // 128 + 8, dtype=torch.int32, device=device)
    _ext.set_decode_scratch(_scratch[0].data_ptr(), _scratch[0].numel() * 4,
                            _scratch[1].data_ptr())
    _scratch_ready[0] = True
    sys.stderr.write("[radiance.autoround] decode split-K scratch registered "
                     f"({_scratch[0].numel() * 4 / 2**20:.0f} MiB)\n")


# Optional in-serve numerics gate: RADIANCE_AR_CHECKALL="N:K,N:K" compares the kernel against an
# exact fp32 dequant for calls at or below RADIANCE_AR_CHECK_MAX_M rows. Needs --enforce-eager,
# because the comparison syncs and that is illegal under CUDA-graph capture.
_ca = os.environ.get("RADIANCE_AR_CHECKALL", "").strip()
CHECK_ALL = ({tuple(int(v) for v in p.split(":")) for p in _ca.split(",") if p} if _ca else None)
CHECK_MAX_M = int(os.environ.get("RADIANCE_AR_CHECK_MAX_M", "128"))
_checked = set()


def _exact_ref(x_fp8, x_scale, qweight, scales, N, K):
    """Dequantize to fp32 and matmul, for the CHECKALL gate. Deliberately slow and obvious."""
    codes = torch.empty((N, K), dtype=torch.int32, device=qweight.device)
    w32 = qweight.view(torch.int32)
    for j in range(PACK):
        codes[:, j::PACK] = (w32 >> (4 * j)) & 0xF
    w = (codes.float() - 8.0) * scales.float().t().repeat_interleave(GROUP, dim=1)
    xr = x_fp8.float() * x_scale.view(-1, 1).float()
    return (xr @ w.t()).to(torch.bfloat16)


@torch.library.custom_op("radiance::autoround_linear", mutates_args=())
def autoround_linear(x: torch.Tensor, qweight: torch.Tensor,
                     scales: torch.Tensor) -> torch.Tensor:
    """Owns the whole dispatch so no shape branch is visible to dynamo.

    vLLM compiles with a dynamic token dimension, so a `M <= 64` test written in apply() is a
    data-dependent branch that splits the graph at every linear. The MXFP4 path measured that at
    ~30% of decode throughput. Inside a registered custom op the body runs eagerly and the Python
    branch is free -- the kernel's prefill/decode choice is made in C++ regardless, but the
    activation quant and the buffer allocation live here too.
    """
    N, K = qweight.shape[0], qweight.shape[1] * PACK
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    from vllm import _custom_ops as ops

    _ensure_scratch(x.device)
    x_fp8, x_scale = ops.scaled_fp8_quant(x2, scale=None, use_per_token_if_dynamic=True)
    x_scale = x_scale.view(-1).float().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch(x_fp8.data_ptr(), qweight.data_ptr(), scales.data_ptr(), x_scale.data_ptr(),
                out.data_ptr(), M, N, K, torch.cuda.current_stream().cuda_stream)
    if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M \
            and (N, K, M) not in _checked:
        _checked.add((N, K, M))
        ref = _exact_ref(x_fp8, x_scale, qweight, scales, N, K)
        num = (out.float() - ref.float()).pow(2).sum().sqrt()
        den = ref.float().pow(2).sum().sqrt().clamp_min(1e-30)
        sys.stderr.write(f"[radiance.autoround] CHECKALL N={N} K={K} M={M} "
                         f"rel={float(num / den):.5f}\n")
    return out.view(*x.shape[:-1], N)


@autoround_linear.register_fake
def _(x, qweight, scales):
    return torch.empty((*x.shape[:-1], qweight.shape[0]), device=x.device, dtype=torch.bfloat16)


_traced = {"q": 0, "u": 0}


def _trace(prefix: str, quantized: bool) -> None:
    """Print the first few routing decisions of each kind, then go quiet.

    Routing is the part of this integration with no compile-time check on it, and getting it
    wrong fails late and confusingly during weight load. A handful of lines at startup makes the
    split visible without flooding a 64-layer model's log.
    """
    k = "q" if quantized else "u"
    _traced[k] += 1
    n = _traced[k]
    # First few of each kind, then powers-of-ten, so the split is visible on a 64-layer model
    # without flooding the log. A cap alone hid a real bug once: the vision tower used up the
    # whole budget and the language layers being misrouted never showed.
    if n <= 3 or n in (10, 100, 400, 1000):
        sys.stderr.write(
            f"[radiance.autoround] {'kernel' if quantized else 'bf16  '} #{n:<4} {prefix}\n")


@register_quantization_config("auto-round")
class AutoRoundConfig(QuantizationConfig):
    """int4 g128 symmetric, W4A8 through the radiance gfx1201 kernel."""

    def __init__(self, group_size: int, sym: bool, fp16_patterns: list[str],
                 quantize_blocks: list[str]):
        super().__init__()
        self.group_size = group_size
        self.sym = sym
        self.fp16_patterns = fp16_patterns
        self._fp16_re = [re.compile(p) for p in fp16_patterns]
        # Only these prefixes were quantized. Everything else -- the whole qwen2.5-VL vision
        # tower, lm_head, the norms -- is plain bf16 in the checkpoint and must be routed to
        # vLLM's unquantized method. Without this the vision tower's proj (K=576) reaches
        # create_weights and trips the group-size check, because 576 is not a multiple of 128.
        self.quantize_blocks = quantize_blocks

    def __repr__(self):
        return (f"AutoRoundConfig(group_size={self.group_size}, sym={self.sym}, "
                f"blocks={self.quantize_blocks}, "
                f"unquantized_modules={len(self.fp16_patterns)})")

    @classmethod
    def get_name(cls):
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0        # gfx1201 does not report a CUDA capability

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "AutoRoundConfig":
        bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        sym = cls.get_from_keys(config, ["sym"])
        if bits != 4:
            raise ValueError(f"radiance AutoRound kernel supports 4 bits only, got {bits}")
        if group_size != GROUP:
            raise ValueError(
                f"radiance AutoRound kernel is built for group_size={GROUP}, got {group_size}. "
                "The kernel's slab structure is aligned to the group; a different group size "
                "needs a different BK.")
        if not sym:
            raise ValueError(
                "radiance AutoRound kernel requires sym=true. The asymmetric case needs a "
                "per-group zero-point correction against activation row-sums, which this kernel "
                "deliberately does not carry.")
        # extra_config names modules held at higher precision; anything with bits >= 16 is not
        # quantized and must go to the unquantized linear method.
        fp16 = [k for k, v in (config.get("extra_config") or {}).items()
                if int(v.get("bits", bits)) >= 16]
        # "model.language_model.layers,mtp.layers" -- the only blocks AutoRound touched.
        blocks = [b.strip() for b in
                  str(config.get("block_name_to_quantize", "")).split(",") if b.strip()]
        return cls(group_size, bool(sym), fp16, blocks)

    def _in_quantized_block(self, prefix: str) -> bool:
        """Is this module inside a block AutoRound actually quantized?

        `block_name_to_quantize` names CHECKPOINT paths ("model.language_model.layers,mtp.layers"),
        while the prefix vLLM passes here is its own MODULE path, and for this architecture the two
        do not line up -- vLLM applies an hf_to_vllm_mapper, and the observed vLLM prefixes are
        things like "visual.blocks.0.attn.qkv" with no "model." at all. Matching the two literally
        sent EVERY linear to the unquantized method, and loading then died on
        `'MergedColumnParallelLinear' object has no attribute 'data'`, because vLLM's
        `getattr(self, name, self)` falls back to the LAYER when `qweight` is not a registered
        parameter.

        So use the structural fact instead of the name mapping: AutoRound quantized the decoder
        blocks and nothing else, and every decoder linear -- language model or MTP -- sits under a
        "layers.<idx>." component, while the qwen2.5-VL tower sits under "visual" and lm_head and
        the norms have no "layers" component at all. Individual exceptions inside the decoder
        (in_proj_a / in_proj_b) are handled separately by extra_config.

        This matters beyond tidiness: the vision merger's proj has K=576, which is not a multiple
        of the 128 group size, so letting it reach create_weights trips the group check.
        """
        if not self.quantize_blocks:
            return True
        parts = prefix.split(".")
        if "visual" in parts:
            return False
        return "layers" in parts

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        """Claim auto-round checkpoints before vLLM's INC config does.

        vLLM ships an INC (Intel Neural Compressor) config whose override maps quant_method
        "auto-round" straight to "inc", and INC refuses to run on ROCm -- "inc quantization is
        currently not supported in rocm" -- so without this the engine dies in ModelConfig before a
        single layer is built. ModelConfig probes CUSTOM registered methods before the built-in
        override list, and "auto-round" is not in the QuantizationMethods literal, so claiming it
        here wins the race without patching vLLM.

        Only claim what this kernel actually supports; anything else falls through to whatever vLLM
        would have done, rather than being taken over and then failing later at load.
        """
        if hf_quant_cfg.get("quant_method") != "auto-round":
            return None
        if hf_quant_cfg.get("bits") != 4 or hf_quant_cfg.get("group_size") != GROUP:
            return None
        if not hf_quant_cfg.get("sym", False):
            return None
        return cls.get_name()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        if not isinstance(layer, LinearBase):
            return None
        # Outside the quantized blocks -> bf16 in the checkpoint.
        if not self._in_quantized_block(prefix):
            _trace(prefix, False)
            return UnquantizedLinearMethod()
        # Inside them, extra_config can still hold individual modules at higher precision.
        for rx in self._fp16_re:
            if rx.fullmatch(prefix) or rx.search(prefix):
                _trace(prefix, False)
                return UnquantizedLinearMethod()
        _trace(prefix, True)
        return _linear_method_cls()(self)


_LINEAR_METHOD_CLS = None


def _linear_method_cls():
    """Build the linear method on first use.

    Declaring it at module scope would need LinearMethodBase and the parameter classes as base
    classes at import time, which is exactly the circular import described at the top of the file.
    By the time a layer asks for a quant method, vLLM is fully initialized.
    """
    global _LINEAR_METHOD_CLS
    if _LINEAR_METHOD_CLS is not None:
        return _LINEAR_METHOD_CLS

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import (GroupQuantScaleParameter, PackedvLLMParameter)

    class AutoRoundLinearMethod(LinearMethodBase):

        def __init__(self, quant_config: AutoRoundConfig):
            self.quant_config = quant_config

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            del input_size, output_size
            out_part = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")
            g = self.quant_config.group_size

            if input_size_per_partition % g:
                raise ValueError(
                    f"K per partition ({input_size_per_partition}) is not a multiple of the group "
                    f"size ({g}); the group structure would straddle a TP shard boundary.")

            # Layouts are the AutoGPTQ ones and the parameter classes are vLLM's own, so TP sharding
            # is handled by the loader: input_dim=0 shards along K, output_dim=1 shards along N.
            # desc_act is off for this checkpoint, so scales shard with K rather than being replicated.
            qweight = PackedvLLMParameter(
                data=torch.empty(input_size_per_partition // PACK, out_part, dtype=torch.int32),
                input_dim=0, output_dim=1, packed_dim=0, packed_factor=PACK,
                weight_loader=weight_loader)
            # fp16 deliberately, not params_dtype: the checkpoint stores F16 scales and bf16 has three
            # fewer mantissa bits, so casting them to bf16 would quantize the scale itself.
            scales = GroupQuantScaleParameter(
                data=torch.empty(input_size_per_partition // g, out_part, dtype=torch.float16),
                input_dim=0, output_dim=1, weight_loader=weight_loader)
            qzeros = PackedvLLMParameter(
                data=torch.empty(input_size_per_partition // g, out_part // PACK, dtype=torch.int32),
                input_dim=0, output_dim=1, packed_dim=1, packed_factor=PACK,
                weight_loader=weight_loader)

            layer.register_parameter("qweight", qweight)
            layer.register_parameter("scales", scales)
            layer.register_parameter("qzeros", qzeros)

        def process_weights_after_loading(self, layer) -> None:
            # Assert the symmetry the kernel's folded zero point depends on, before anything else.
            z = layer.qzeros.data.view(torch.int32)
            if z.numel():
                bad = (z != torch.tensor(0x77777777, dtype=torch.int32, device=z.device)).sum()
                if int(bad):
                    raise ValueError(
                        f"AutoRound checkpoint is not symmetric: {int(bad)} of {z.numel()} qzeros "
                        "words differ from 0x77777777 (zero != 8). This kernel folds a CONSTANT zero "
                        "of 8 into its unpack table and would be silently wrong here.")
            del layer.qzeros
            layer.qzeros = None

            # The kernel reads the weight as [N, K/8] so a wave's read walks one row contiguously;
            # the checkpoint's [K/8, N] would stride by N between consecutive k.
            qw = layer.qweight.data
            layer.qweight = torch.nn.Parameter(qw.t().contiguous(), requires_grad=False)
            layer.scales = torch.nn.Parameter(layer.scales.data.contiguous(), requires_grad=False)

        def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            out = torch.ops.radiance.autoround_linear(x, layer.qweight, layer.scales)
            if bias is not None:
                out = out + bias
            return out

    _LINEAR_METHOD_CLS = AutoRoundLinearMethod
    return _LINEAR_METHOD_CLS
