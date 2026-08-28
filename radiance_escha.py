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


@torch.library.custom_op("radiance::escha_linear_into", mutates_args=("out",))
def escha_linear_into(out: torch.Tensor, x: torch.Tensor, code: torch.Tensor, rin: torch.Tensor,
                      rout: torch.Tensor, s_in: torch.Tensor, s_out: torch.Tensor,
                      kbits: int, col0: int) -> None:
    """One escha projection, written straight into its column slice of `out`.

    Owns the dispatch so no shape branch is visible to dynamo: vLLM compiles with a dynamic token
    dimension, so an `M <= 64` test in apply() is a data-dependent branch that splits the graph at
    every linear -- the MXFP4 path measured that at ~30% of decode throughput. Inside a registered
    custom op the body runs eagerly and the prefill/decode choice is made in C++.

    It writes into a caller-owned `out` rather than returning its own buffer because a merged
    module runs one of these per source tensor, and concatenating the results afterwards costs a
    full output-sized read plus write -- 570 MB on gate_up at M=8192.
    """
    IC = code.shape[0] * _TILE
    OC = code.shape[1] * _TILE
    # The kernels index raw pointers with a fixed row stride, so a non-contiguous or wrong-dtype
    # activation walks off the allocation -- which surfaces as a GPU memory fault inside
    # escha_pre_quant and names nothing useful.
    x2 = x.reshape(-1, IC)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    M = x2.shape[0]
    if M == 0:
        return
    if rin.numel() != IC or s_in.numel() != IC:
        raise ValueError(f"escha: rin/s_in length {rin.numel()}/{s_in.numel()} != IC {IC}")
    if rout.numel() != OC or s_out.numel() != OC:
        raise ValueError(f"escha: rout/s_out length {rout.numel()}/{s_out.numel()} != OC {OC}")
    dev = x.device
    ext = _load_ext()
    _ensure_scratch(dev, max(8 * 64 * OC * 4, 1 << 20))

    A = torch.empty((M, IC), device=dev, dtype=torch.uint8)
    As = torch.empty(M, device=dev, dtype=torch.float32)
    amax = torch.zeros(M, device=dev, dtype=torch.float32)
    C = torch.empty((M, OC), device=dev, dtype=torch.bfloat16)
    ldo = out.shape[-1]
    o2 = out.reshape(-1, ldo)
    ext.launch(x2.data_ptr(), code.data_ptr(), rin.data_ptr(), rout.data_ptr(),
               s_in.data_ptr(), s_out.data_ptr(), A.data_ptr(), As.data_ptr(),
               amax.data_ptr(), C.data_ptr(), o2.data_ptr(), M, OC, IC, int(kbits),
               ldo, int(col0), torch.cuda.current_stream().cuda_stream)


@escha_linear_into.register_fake
def _(out, x, code, rin, rout, s_in, s_out, kbits, col0):
    return None


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
            # Match on the SUFFIX from "layers." onward, not the full path. layer_meta is written
            # in CHECKPOINT namespace ("model.language_model.layers.0.mlp.gate_proj") while vLLM
            # asks with its own module path, and the two arrangements differ for this
            # architecture. Keying on the tail makes the lookup independent of that.
            self._by_tail = {self._tail(n): m for n, m in self.layer_meta.items()}

        @staticmethod
        def _tail(name: str) -> str:
            i = name.find("layers.")
            return name[i:] if i >= 0 else name.rpartition(".")[2]

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
            hit = [n for n in names if self._tail(n) in self._by_tail]
            if hit and len(hit) != len(names):
                # A partly-coded merge would need a mixed quantized/dense GEMM pair; the
                # checkpoint does not do this, and silently treating it as dense would be wrong.
                raise ValueError(
                    f"escha: merged module {prefix} is only partly coded ({len(hit)}/{len(names)}"
                    f": {hit}); this loader cannot mix coded and dense shards.")
            return bool(hit)

        def kbits_for(self, prefix: str):
            return [int(self._by_tail[self._tail(n)]["K"]) for n in self._names_for(prefix)]

        def out_features_for(self, prefix: str):
            return [int(self._by_tail[self._tail(n)]["out_features"])
                    for n in self._names_for(prefix)]

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            # Install the int8 embedding shim HERE, not at registration. Registration runs from
            # inside vllm.model_executor.layers.quantization.__init__, where importing the model
            # loader re-enters a half-built vllm.config:
            #   ImportError: cannot import name 'ModelConfig' from partially initialized module
            # By the time a layer asks for its quant method vLLM is fully up, and model
            # construction still precedes weight loading, so the shim is in place in time.
            _install_weight_shim()
            from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
            if not isinstance(layer, LinearBase):
                return None
            coded = self.is_coded(prefix)
            if os.environ.get("RADIANCE_ESCHA_TRACE"):
                sys.stderr.write(f"[radiance.escha] route {prefix} coded={coded}\n")
            if not coded:
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
            # One shard per SOURCE TENSOR, which is not one per output partition: the GDN
            # in_proj_qkvz has four partitions (q, k, v, z) but only two coded tensors, because
            # in_proj_qkv covers q+k+v. Each source tensor is one GEMM, so shards are counted
            # from the sources and each shard's width is taken from its own code tensor later.
            nshard = len(self.kbits)
            if input_size_per_partition % _HAD:
                raise ValueError(
                    f"escha: {self.prefix} K per rank ({input_size_per_partition}) is not a "
                    f"multiple of {_HAD}; the incoherence transform is blocked at 128 and a "
                    f"shard boundary inside a block would change the transform itself.")
            from vllm.model_executor.layers.linear import RowParallelLinear
            layer.escha_row_parallel = isinstance(layer, RowParallelLinear)
            layer.escha_ic = input_size_per_partition
            layer.escha_parts = list(output_partition_sizes)
            layer.escha_oc_total = sum(output_partition_sizes)
            # Per-rank width of each SOURCE tensor, from layer_meta. Needed to turn vLLM's
            # partition-index shard ids back into a source index -- see _shard_index.
            # Row-parallel shards the INPUT, so its output width is the full one; column-parallel
            # shards the output and each source contributes out_features/tp per rank.
            _div = 1 if layer.escha_row_parallel else tp
            layer.escha_src_w = [w // _div for w in self.quant_config.out_features_for(self.prefix)]
            # Which output partitions each source tensor covers. Usually one, but the GDN
            # in_proj_qkv is ONE tensor spanning q, k and v -- and that distinction decides how it
            # is sharded: a single contiguous narrow would hand rank 0 all of q, all of k and a
            # slice of v instead of each projection's own slice.
            groups, j, acc = [], 0, 0
            for w in layer.escha_src_w:
                g = []
                while acc < w and j < len(layer.escha_parts):
                    g.append(layer.escha_parts[j]); acc += layer.escha_parts[j]; j += 1
                if acc != w:
                    raise ValueError(f"escha: {self.prefix} output partitions "
                                     f"{layer.escha_parts} do not group into source widths "
                                     f"{layer.escha_src_w}")
                groups.append(g); acc = 0
            layer.escha_src_parts = groups
            layer.escha_kbits = list(self.kbits)
            # Which axis is sharded decides which tensors get sliced. Inferring it from the
            # sizes is ambiguous at TP=1 (both tests pass), so the layer type decides; it is set
            # above, before the source widths that depend on it.
            layer.escha_raw = [dict() for _ in range(nshard)]
            layer.escha_prefix = self.prefix

            # Placeholders so vLLM can resolve the checkpoint names; the real tensors have
            # per-shard shapes (K=2 and K=3 shards differ in the code tensor's last dim), so they
            # cannot live in one parameter and are collected by the loader below instead.
            if os.environ.get("RADIANCE_ESCHA_TRACE"):
                sys.stderr.write(f"[radiance.escha] create_weights {self.prefix} "
                                 f"ic={input_size_per_partition} oc={list(output_partition_sizes)} "
                                 f"K={self.kbits} row={layer.escha_row_parallel}\n")
            for suffix in SUFFIXES:
                p = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": self._make_loader(layer, suffix)})
                layer.register_parameter(suffix, p)

        # ---------------------------------------------------------------- loading
        def _shard_index(self, layer, shard_id):
            """vLLM's shard id -> index of the SOURCE TENSOR it belongs to.

            Three forms turn up. A name ("gate"/"up"/"q") maps directly. An int or a TUPLE of ints
            is a partition index, and partitions are finer than sources: the GDN in_proj_qkvz has
            partitions (q, k, v, z) but in_proj_qkv supplies the first three, so vLLM passes
            (0, 1, 2) for one tensor. Those are resolved by turning the lowest partition index
            into a column offset and asking which source's span contains it.
            """
            if shard_id is None:
                return 0
            names = self.quant_config._names_for(self.prefix)
            leaves = [n.rpartition(".")[2] for n in names]
            if isinstance(shard_id, str):
                for i, leaf in enumerate(leaves):
                    if shard_id == leaf or shard_id == leaf.replace("_proj", ""):
                        return i
                raise ValueError(f"escha: {self.prefix} unknown shard id {shard_id!r} "
                                 f"(expected one of {leaves})")
            first = min(shard_id) if isinstance(shard_id, (tuple, list)) else int(shard_id)
            off = sum(layer.escha_parts[:first])
            acc = 0
            for i, w in enumerate(layer.escha_src_w):
                if off < acc + w:
                    return i
                acc += w
            raise ValueError(f"escha: {self.prefix} shard id {shard_id!r} -> column {off} falls "
                             f"outside the source widths {layer.escha_src_w}")

        def _make_loader(self, layer, suffix):
            def loader(param, loaded_weight, shard_id=None, *args, **kwargs):
                from vllm.distributed import (get_tensor_model_parallel_rank,
                                              get_tensor_model_parallel_world_size)
                del param, args, kwargs
                if os.environ.get("RADIANCE_ESCHA_TRACE"):
                    sys.stderr.write(f"[radiance.escha] load {layer.escha_prefix}.{suffix} "
                                     f"shard={shard_id!r} shape={tuple(loaded_weight.shape)}\n")
                i = self._shard_index(layer, shard_id)
                t = loaded_weight
                tp = get_tensor_model_parallel_world_size()
                r = get_tensor_model_parallel_rank()
                if tp > 1:
                    row = layer.escha_row_parallel
                    if row:
                        # Row-parallel: the input axis is sharded and every source covers exactly
                        # one projection, so a contiguous narrow is right.
                        if suffix == "escha_code" or suffix in IC_SIDE:
                            per = t.shape[0] // tp
                            t = t.narrow(0, r * per, per)
                    else:
                        # Column-parallel: slice each of the source's output partitions
                        # separately and re-concatenate, because a source may span several.
                        if suffix == "escha_code" or suffix in OC_SIDE:
                            ax = 1 if suffix == "escha_code" else 0
                            unit = _TILE if suffix == "escha_code" else 1
                            off, pieces = 0, []
                            for w in layer.escha_src_parts[i]:
                                full = w * tp
                                pieces.append(t.narrow(ax, (off + r * w) // unit, w // unit))
                                off += full
                            t = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=ax)
                # Move to the worker's GPU HERE. vLLM hands weight loaders CPU tensors and
                # normally they are copied into a pre-allocated device parameter; these are kept
                # as plain tensors, so without this they stay on the host and the kernels get a
                # host pointer -- which surfaces as a GPU memory fault naming escha_pre_quant and
                # nothing else. Transferring per shard also avoids holding every shard in host
                # RAM at once.
                dev = torch.device("cuda", torch.cuda.current_device())
                layer.escha_raw[i][suffix] = t.contiguous().to(dev, copy=True)
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
                oc_i = c.shape[1] * _TILE
                if oc_i % _HAD:
                    raise ValueError(
                        f"escha: {layer.escha_prefix} shard {i} N per rank ({oc_i}) is not a "
                        f"multiple of {_HAD}; the output transform is blocked at 128 and a shard "
                        f"boundary inside a block would change the transform itself.")
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

            got = sum(c.shape[1] * _TILE for c in code)
            if got != layer.escha_oc_total:
                raise ValueError(
                    f"escha: {layer.escha_prefix} shards cover N={got} but the layer's partitions "
                    f"total {layer.escha_oc_total}")
            for s in SUFFIXES:
                if hasattr(layer, s):
                    delattr(layer, s)
            layer.escha_code = code
            layer.escha_rout, layer.escha_s_out = rout, s_out
            layer.escha_rin, layer.escha_s_in = rin, s_in
            layer.escha_raw = None

        def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
            # One buffer for the whole layer; each shard writes its own columns. A merged module
            # is several coded tensors (gate K=2 and up K=3 cannot share a GEMM), and cat-ing
            # their outputs afterwards is a full output-sized copy per layer.
            out = torch.empty((*x.shape[:-1], layer.escha_oc_total), device=x.device,
                              dtype=torch.bfloat16)
            col = 0
            for i in range(len(layer.escha_code)):
                torch.ops.radiance.escha_linear_into(
                    out, x, layer.escha_code[i], layer.escha_rin[i], layer.escha_rout[i],
                    layer.escha_s_in[i], layer.escha_s_out[i], layer.escha_kbits[i], col)
                col += layer.escha_code[i].shape[1] * _TILE
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
    cls = _quant_config_cls()
    try:
        register_quantization_config("escha")(cls)
    except ValueError:
        pass          # already registered (module imported twice)
    return cls
