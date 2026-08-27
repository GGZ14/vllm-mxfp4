"""vLLM quantization method for EschaLabs' escha (EXL3 trellis) W2 checkpoints on gfx1201.

THE FORWARD is not a plain dequant-and-matmul. EXL3 applies its incoherence transform to
ACTIVATIONS, so the layer is three kernels, taken verbatim from the reference runtime's serving
path (escha/linear.py::_forward_runtime_had and sglang .../quantization/escha.py::_prefill_recon):

    y = Had128( (x * s_in) * rin ) @ decode(code)  ->  Had128  ->  * rout  ->  * s_out

Two easy mistakes, both load-bearing: `rin` is a PRE-scale applied before its Hadamard while `rout`
is a POST-scale applied after its own -- they are not symmetric; and `rin` already has the weight
scale folded in, so nothing may re-apply it.

The checkpoint's per-output `bias` vectors are deliberately NOT applied. The model card states the
reference runtime does not apply them and that every published number was produced without them,
so applying them would diverge from the results we are trying to reproduce.

WHY EVERYTHING IS PER-SHARD. vLLM merges gate_proj+up_proj into one gate_up_proj (and
in_proj_qkv+in_proj_z into in_proj_qkvz). In this checkpoint gate_proj is coded at K=2 and up_proj
at K=3 -- in all 64 layers -- so the two halves have different bit rates and different code-tensor
shapes. They cannot be one parameter or one GEMM, so each merged shard keeps its own code / rout /
s_out and gets its own kernel call. `rin` and `s_in` are shared: the input axis is common.
"""
import os
import sys
import torch

_ext = None
_LINEAR_METHOD_CLS = None
_HAD = 128
_TILE = 16                      # code tensor is [IC//16, OC//16, 16*K]


def _load_ext():
    global _ext
    if _ext is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import radiance_escha_kernel as k
        _ext = k
    return _ext


# Split-K partial slab + block counter, allocated once per device and handed to the module. A lazy
# hipMalloc from inside launch() would land in CUDA-graph capture, where it is illegal.
_scratch = {}


def _ensure_scratch(device, need_bytes):
    key = (device.type, device.index)
    cur = _scratch.get(key)
    if cur is None or cur[2] < need_bytes:
        partial = torch.empty(need_bytes // 4, dtype=torch.float32, device=device)
        cnt = torch.zeros(4096, dtype=torch.int32, device=device)
        _scratch[key] = (partial, cnt, need_bytes)
        _load_ext().set_decode_scratch(partial.data_ptr(), need_bytes, cnt.data_ptr())
    return _scratch[key]


@torch.library.custom_op("radiance::escha_linear", mutates_args=())
def escha_linear(x: torch.Tensor, code: torch.Tensor, rin: torch.Tensor, rout: torch.Tensor,
                 s_in: torch.Tensor, s_out: torch.Tensor, kbits: int) -> torch.Tensor:
    """One escha projection. Owns the dispatch so no shape branch is visible to dynamo.

    vLLM compiles with a dynamic token dimension, so an `M <= 64` test written in apply() is a
    data-dependent branch that splits the graph at every linear; the MXFP4 path measured that at
    ~30% of decode throughput. Inside a registered custom op the body runs eagerly and the
    prefill/decode choice is made in C++.
    """
    IC = code.shape[0] * _TILE
    OC = code.shape[1] * _TILE
    x2 = x.reshape(-1, IC)
    M = x2.shape[0]
    dev = x.device
    ext = _load_ext()

    nblk = (OC + 127) // 128
    _ensure_scratch(dev, max(8 * 64 * OC * 4, 1 << 20))

    A = torch.empty((M, IC), device=dev, dtype=torch.uint8)
    As = torch.empty(M, device=dev, dtype=torch.float32)
    C = torch.empty((M, OC), device=dev, dtype=torch.bfloat16)
    out = torch.empty((M, OC), device=dev, dtype=torch.bfloat16)
    ext.launch(x2.data_ptr(), code.data_ptr(), rin.data_ptr(), rout.data_ptr(),
               s_in.data_ptr(), s_out.data_ptr(), A.data_ptr(), As.data_ptr(),
               C.data_ptr(), out.data_ptr(), M, OC, IC, int(kbits),
               torch.cuda.current_stream().cuda_stream)
    del nblk
    return out.view(*x.shape[:-1], OC)


@escha_linear.register_fake
def _(x, code, rin, rout, s_in, s_out, kbits):
    return torch.empty((*x.shape[:-1], code.shape[1] * _TILE), device=x.device,
                       dtype=torch.bfloat16)


# --------------------------------------------------------------------------------------------
# Quantization config
# --------------------------------------------------------------------------------------------
def _quant_config_cls():
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    class EschaConfig(QuantizationConfig):
        """Reads `quantization_config: {"quant_method": "escha", ...}` from config.json.

        `layer_meta` names every coded projection in CHECKPOINT namespace, which is what makes the
        routing decision exact rather than heuristic -- vLLM asks about MERGED prefixes, so a
        merged module is quantized iff any of its constituent checkpoint tensors is coded.
        """

        # vLLM asks about the merged module; the checkpoint names the pieces.
        MERGED = {
            "gate_up_proj": ["gate_proj", "up_proj"],
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }

        def __init__(self, layer_meta: dict):
            super().__init__()
            self.layer_meta = layer_meta or {}

        def __repr__(self):
            ks = {}
            for m in self.layer_meta.values():
                ks[m.get("K")] = ks.get(m.get("K"), 0) + 1
            return f"EschaConfig(coded={len(self.layer_meta)}, K={ks})"

        @classmethod
        def get_name(cls):
            return "escha"

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.bfloat16, torch.float16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @staticmethod
        def get_config_filenames() -> list:
            return []

        @classmethod
        def from_config(cls, config: dict) -> "EschaConfig":
            return cls(config.get("layer_meta", {}))

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            """Claim escha checkpoints before anything else can.

            Same defensive reason as the AutoRound path: an unknown quant_method can otherwise be
            picked up by a generic handler that then fails deep inside weight loading, where the
            error names the wrong subsystem.
            """
            try:
                m = (hf_quant_cfg or {}).get("quant_method", "")
            except AttributeError:
                m = getattr(hf_quant_cfg, "quant_method", "")
            if str(m).lower() == "escha" and user_quant in (None, "escha"):
                return "escha"
            return None

        def _names_for(self, prefix: str):
            """Checkpoint tensor names that feed a vLLM module prefix, in shard order."""
            base, _, leaf = prefix.rpartition(".")
            if leaf in self.MERGED:
                return [f"{base}.{p}" for p in self.MERGED[leaf]]
            return [prefix]

        def is_coded(self, prefix: str) -> bool:
            names = self._names_for(prefix)
            hit = [n for n in names if n in self.layer_meta]
            if hit and len(hit) != len(names):
                # A partly-coded merge would need a mixed quantized/dense GEMM pair; the
                # checkpoint does not do this, and silently treating it as dense would be wrong.
                raise ValueError(
                    f"escha: merged module {prefix} is only partly coded ({len(hit)}/{len(names)}"
                    f": {hit}); this loader cannot mix coded and dense shards.")
            return bool(hit)

        def kbits_for(self, prefix: str):
            return [int(self.layer_meta[n]["K"]) for n in self._names_for(prefix)]

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
            if not isinstance(layer, LinearBase):
                return None
            if not self.is_coded(prefix):
                return UnquantizedLinearMethod()
            return _linear_method_cls()(self, prefix)

    return EschaConfig


# --------------------------------------------------------------------------------------------
# Linear method
# --------------------------------------------------------------------------------------------
def _linear_method_cls():
    """Built on first use. Declaring it at module scope would need LinearMethodBase as a base
    class at import time, which is the circular import the AutoRound path documents."""
    global _LINEAR_METHOD_CLS
    if _LINEAR_METHOD_CLS is not None:
        return _LINEAR_METHOD_CLS

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.utils import set_weight_attrs

    SUFFIXES = ("escha_code", "escha_rin", "escha_rout", "escha_s_in", "escha_s_out",
                "escha_config")
    OC_SIDE = {"escha_rout", "escha_s_out"}
    IC_SIDE = {"escha_rin", "escha_s_in"}

    class EschaLinearMethod(LinearMethodBase):

        def __init__(self, quant_config, prefix: str):
            self.quant_config = quant_config
            self.prefix = prefix
            self.kbits = quant_config.kbits_for(prefix)

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp = get_tensor_model_parallel_world_size()
            nshard = len(output_partition_sizes)
            if nshard != len(self.kbits):
                raise ValueError(
                    f"escha: {self.prefix} has {nshard} output partitions but "
                    f"{len(self.kbits)} coded source tensors {self.kbits}")
            if input_size_per_partition % _HAD:
                raise ValueError(
                    f"escha: {self.prefix} K per rank ({input_size_per_partition}) is not a "
                    f"multiple of {_HAD}; the incoherence transform is blocked at 128 and a "
                    f"shard boundary inside a block would change the transform itself.")
            for i, oc in enumerate(output_partition_sizes):
                if oc % _HAD:
                    raise ValueError(
                        f"escha: {self.prefix} shard {i} N per rank ({oc}) is not a multiple of "
                        f"{_HAD}; same reason as K above, on the output transform.")

            layer.escha_ic = input_size_per_partition
            layer.escha_oc = list(output_partition_sizes)
            layer.escha_kbits = list(self.kbits)
            # Which axis is sharded decides which tensors get sliced. Inferring it from the sizes
            # is ambiguous at TP=1 (both tests pass), so ask the layer what it is.
            from vllm.model_executor.layers.linear import RowParallelLinear
            layer.escha_row_parallel = isinstance(layer, RowParallelLinear)
            layer.escha_raw = [dict() for _ in range(nshard)]
            layer.escha_prefix = self.prefix

            # Placeholders so vLLM can resolve the checkpoint names; the real tensors have
            # per-shard shapes (K=2 and K=3 shards differ in the code tensor's last dim), so they
            # cannot live in one parameter and are collected by the loader below instead.
            for suffix in SUFFIXES:
                p = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": self._make_loader(layer, suffix)})
                layer.register_parameter(suffix, p)

        # ---------------------------------------------------------------- loading
        def _shard_index(self, layer, shard_id):
            if shard_id is None:
                return 0
            if isinstance(shard_id, int):
                return shard_id
            names = self.quant_config._names_for(self.prefix)
            leaves = [n.rpartition(".")[2] for n in names]
            for i, leaf in enumerate(leaves):
                if shard_id == leaf or shard_id == leaf.replace("_proj", ""):
                    return i
            raise ValueError(f"escha: {self.prefix} unknown shard id {shard_id!r} "
                             f"(expected one of {leaves})")

        def _make_loader(self, layer, suffix):
            def loader(param, loaded_weight, shard_id=None, *args, **kwargs):
                from vllm.distributed import (get_tensor_model_parallel_rank,
                                              get_tensor_model_parallel_world_size)
                del param, args, kwargs
                i = self._shard_index(layer, shard_id)
                t = loaded_weight
                tp = get_tensor_model_parallel_world_size()
                r = get_tensor_model_parallel_rank()
                if tp > 1:
                    row = layer.escha_row_parallel
                    if suffix == "escha_code":
                        # [IC//16, OC//16, 16K]: whichever axis this layer shards, shard it in
                        # TILE units. Both param and checkpoint are in the same units, so the
                        # divisor cancels -- but the 128-block transform still requires the slice
                        # to be 128-aligned, which create_weights asserted.
                        ax = 0 if row else 1
                        per = t.shape[ax] // tp
                        t = t.narrow(ax, r * per, per)
                    elif (suffix in IC_SIDE) == bool(row):
                        # IC-side tensors shard only on row-parallel; OC-side only on column.
                        per = t.shape[0] // tp
                        t = t.narrow(0, r * per, per)
                layer.escha_raw[i][suffix] = t.contiguous().clone()
            return loader

        def process_weights_after_loading(self, layer) -> None:
            code, rout, s_out, rin, s_in = [], [], [], [], []
            for i, raw in enumerate(layer.escha_raw):
                missing = [s for s in ("escha_code", "escha_rin", "escha_rout",
                                       "escha_s_in", "escha_s_out") if s not in raw]
                if missing:
                    raise ValueError(f"escha: {layer.escha_prefix} shard {i} never received "
                                     f"{missing}; the checkpoint or the name mapping is wrong.")
                c = raw["escha_code"]
                # int16 [IC//16, OC//16, 16K] -> uint32 words, exactly the kernel's tile layout.
                if c.dtype != torch.int16:
                    raise ValueError(f"escha: code dtype {c.dtype}, expected int16")
                k_from_shape = c.shape[2] // 16
                if k_from_shape != layer.escha_kbits[i]:
                    raise ValueError(
                        f"escha: {layer.escha_prefix} shard {i} code implies K={k_from_shape} "
                        f"but layer_meta says K={layer.escha_kbits[i]}")
                if c.shape[0] * _TILE != layer.escha_ic:
                    raise ValueError(
                        f"escha: {layer.escha_prefix} shard {i} code IC={c.shape[0] * _TILE} "
                        f"!= partition IC={layer.escha_ic}")
                if c.shape[1] * _TILE != layer.escha_oc[i]:
                    raise ValueError(
                        f"escha: {layer.escha_prefix} shard {i} code OC={c.shape[1] * _TILE} "
                        f"!= partition OC={layer.escha_oc[i]}")
                code.append(c.view(torch.uint8).view(torch.int32).contiguous())
                rout.append(raw["escha_rout"].to(torch.float16).contiguous())
                s_out.append(raw["escha_s_out"].to(torch.float32).contiguous())
                # rin is PER SHARD, not shared, even though the shards consume the same x: it
                # carries the projection's weight scale folded in ("Wscale already folded in"), so
                # gate_proj and up_proj disagree on it. Measured, not assumed -- an earlier version
                # of this loader asserted they matched and the assert fired on layer 0. Each shard
                # therefore gets its own pre-rotation of the activations.
                rin.append(raw["escha_rin"].to(torch.float16).contiguous())
                s_in.append(raw["escha_s_in"].to(torch.float32).contiguous())

            for s in SUFFIXES:
                if hasattr(layer, s):
                    delattr(layer, s)
            layer.escha_code = code
            layer.escha_rout, layer.escha_s_out = rout, s_out
            layer.escha_rin, layer.escha_s_in = rin, s_in
            layer.escha_raw = None

        def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
            outs = [torch.ops.radiance.escha_linear(
                        x, layer.escha_code[i], layer.escha_rin[i], layer.escha_rout[i],
                        layer.escha_s_in[i], layer.escha_s_out[i], layer.escha_kbits[i])
                    for i in range(len(layer.escha_code))]
            out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
            if bias is not None:
                out = out + bias
            return out

    _LINEAR_METHOD_CLS = EschaLinearMethod
    return _LINEAR_METHOD_CLS


def _install_weight_shim():
    """Dequantize the checkpoint's int8 embedding and output head as weights stream past.

    escha stores `embed_tokens` and `lm_head` as `weight_int8` (I8 [V, H]) plus `weight_scale`
    (F16 [V]) -- there is no `weight` tensor at all, so vLLM finds nothing to load and the model
    comes up with an uninitialized embedding, which fails as garbage output rather than as an
    error. Both are plain per-row symmetric int8, so they are reconstituted here and renamed to
    the `.weight` vLLM expects; sharding and placement then proceed untouched.

    Doing it in the weights stream rather than by rewriting the checkpoint keeps 5.1 GB off disk,
    and the dequantized head is the same size vLLM would have held for a bf16 checkpoint anyway.
    """
    from vllm.model_executor.model_loader import default_loader as dl

    if getattr(dl.DefaultModelLoader, "_radiance_escha_shim", False):
        return
    orig = dl.DefaultModelLoader.get_all_weights

    # Only these two are int8-stored. Restricting by name matters: `weight_scale` is a common
    # suffix and blindly consuming it would swallow other schemes' scales.
    def _is_target(base):
        return base.endswith("embed_tokens") or base.endswith("lm_head")

    def get_all_weights(self, model_config, model):
        q, sc = {}, {}
        for name, t in orig(self, model_config, model):
            base = None
            if name.endswith(".weight_int8"):
                base = name[: -len(".weight_int8")]
                store = q
            elif name.endswith(".weight_scale"):
                base = name[: -len(".weight_scale")]
                store = sc
            if base is None or not _is_target(base):
                yield name, t
                continue
            store[base] = t
            if base in q and base in sc:
                w = q.pop(base).to(torch.bfloat16) * sc.pop(base).to(torch.bfloat16).unsqueeze(1)
                yield base + ".weight", w
        left = set(q) | set(sc)
        if left:
            raise ValueError(f"escha: int8 weight/scale pairs never completed for {sorted(left)}")

    dl.DefaultModelLoader.get_all_weights = get_all_weights
    dl.DefaultModelLoader._radiance_escha_shim = True


def register():
    """Register the escha method with vLLM's quantization registry."""
    from vllm.model_executor.layers.quantization import register_quantization_config
    _install_weight_shim()
    cls = _quant_config_cls()
    try:
        register_quantization_config("escha")(cls)
    except ValueError:
        pass          # already registered (module imported twice)
    return cls
