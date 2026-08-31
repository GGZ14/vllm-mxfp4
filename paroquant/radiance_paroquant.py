"""ParoQuant int4 (group-128, asymmetric, pairwise-rotated) W4A8 for radiance on gfx1201.

Checkpoint format (z-lab/Qwen3.8-27B-PARO, quant_method="paroquant"): per projection, AWQ buffers
qweight [K, N/8] int32 / qzeros [K/128, N/8] int32 / scales [K/128, N] fp16 (AWQ nibble reorder
0,2,4,6,1,3,5,7), PLUS the rotation that was applied to the weights before quantization:
pairs [krot, K] int16 (Givens pair indices, local to each 128-channel group), theta [krot, K/2]
fp16, channel_scales [1, K] fp16 stored pre-inverted (multiply activations by it).

Inference identity: with Q the stored codes and R the composed rotations,
    y = x W^T = ((x * channel_scales) R^T) dequant(Q)^T
so the serving path is: rotate+scale the activations (fused with per-group fp8 quantization in
one kernel), then an int4-asymmetric x fp8 GEMM whose zero point is carried as a
row-sum correction (see par_kernels.h). Rotations are PER PROJECTION, so a merged linear (QKV,
gate_up, the GDN in_proj merge) quantizes P differently-rotated copies of x and the GEMM selects
the right one per n-block from the partition boundaries -- one launch either way.

Properties asserted at load rather than assumed:
  1. group_size is 128 and K per partition stays a multiple of 128 under TP.
  2. bits is 4 and krot <= 8 (the prologue's LDS table is sized for 8).
  3. partition boundaries land on multiples of 128 (the decode n-block), so no GEMM block
     straddles two rotations.

Unquantized modules (fp16 in the checkpoint): the visual tower, linear_attn.in_proj_a/b, lm_head.
Enable with RADIANCE_PAROQUANT=1 (registration is import-time; the env only gates logging).
"""
import os
import re
import sys

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.model_executor.layers.linear import LinearMethodBase

import radiance_paroquant_kernel as _ext

GROUP = 128
PACK = 8                       # int4 codes per int32
KROT_MAX = 8
_AWQ_INV = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])   # argsort of the AWQ reorder (0,2,4,6,1,3,5,7)
_SHARD_INDEX = {"q": 0, "k": 1, "v": 2}

# Split-K scratch, sized like the AutoRound/MXFP4 ones: KS(4) x maxM(64) x maxN(32768).
# Allocated at first use from Python and registered with the extension -- a lazy hipMalloc from
# launch() lands inside CUDA-graph capture whenever the torch.compile cache is warm.
_DEC_KS, _DEC_MAX_M, _DEC_MAX_N = 4, 64, 32768
_scratch = [None, None]
_scratch_ready = [False]


def _ensure_scratch(device):
    if _scratch_ready[0]:
        return
    _scratch[0] = torch.empty(_DEC_KS * _DEC_MAX_M * _DEC_MAX_N, dtype=torch.float32,
                              device=device)
    _scratch[1] = torch.zeros(_DEC_MAX_N // 128 + 8, dtype=torch.int32, device=device)
    _ext.set_decode_scratch(_scratch[0].data_ptr(), _scratch[0].numel() * 4,
                            _scratch[1].data_ptr())
    _scratch_ready[0] = True
    sys.stderr.write("[radiance.paroquant] decode split-K scratch registered "
                     f"({_scratch[0].numel() * 4 / 2**20:.0f} MiB)\n")


# In-serve numerics gate, same contract as RADIANCE_AR_CHECKALL: "N:K,N:K" compares the kernel
# against an exact fp32 dequant for calls at or below RADIANCE_PQ_CHECK_MAX_M rows. Needs
# --enforce-eager (the comparison syncs, illegal under CUDA-graph capture).
_ca = os.environ.get("RADIANCE_PQ_CHECKALL", "").strip()
CHECK_ALL = ({tuple(int(v) for v in p.split(":")) for p in _ca.split(",") if p} if _ca else None)
CHECK_MAX_M = int(os.environ.get("RADIANCE_PQ_CHECK_MAX_M", "128"))
# The decode band (per-group scales, fused single-launch prologue) serves M up to this; larger M
# takes the per-token prefill path (pass A rotate -> pass C token quant -> PTOK GEMM).
# RADIANCE_PQ_PTOK=0 forces the per-group path at every M: ~7% slower prefill, finer-grained fp8
# (GSM8K 500q measured 98.0 per-group vs 97.4 per-token -- inside binomial noise, but the lever
# is one env var if a future gate disagrees).
DECODE_MAX_M = int(os.environ.get("RADIANCE_PQ_DECODE_MAX_M", "64"))
PTOK_ENABLED = os.environ.get("RADIANCE_PQ_PTOK", "1") == "1"
_checked = set()

_E4M3_TABLE = [None]


def _e4m3_table(device):
    if _E4M3_TABLE[0] is None or _E4M3_TABLE[0].device != device:
        t = torch.zeros(256, dtype=torch.float32)
        for b in range(256):
            s = -1.0 if b >> 7 else 1.0
            E, m = (b >> 3) & 0xF, b & 7
            t[b] = s * m * 2.0**-9 if E == 0 else s * (1 + m / 8.0) * 2.0**(E - 7)
            if E == 0xF and m == 7:
                t[b] = float("nan")
        _E4M3_TABLE[0] = t.to(device)
    return _E4M3_TABLE[0]


def _exact_ref(a_codes, asg, rs, qweight, sz, N, K, pb1, pb2, as_tok=None):
    """Dequantize to fp32 and matmul, mirroring the kernel algebra. Deliberately slow/obvious.
    Per-group mode: asg [P,M,G], rs = rowsum*asg. Per-token mode: as_tok [P,M], rs plain."""
    device = qweight.device
    G = K // GROUP
    sc = sz.view(G, N, 2)[..., 0].float()            # [G, N]
    zsc = sz.view(G, N, 2)[..., 1].float()
    P, M = a_codes.shape[0], a_codes.shape[1]
    aval = _e4m3_table(device)[a_codes.view(torch.uint8).long()]     # [P, M, K] f32
    out = torch.zeros((M, N), dtype=torch.float64, device=device)
    bounds = [0, min(pb1, N), min(pb2, N), N]
    CHUNK = 512      # n-columns at a time: the ref must not OOM a 0.92-util worker
    for p in range(P):
        n0, n1 = bounds[p], bounds[p + 1]
        for c0 in range(n0, n1, CHUNK):
            c1 = min(n1, c0 + CHUNK)
            w32 = qweight[c0:c1].view(torch.int32).to(torch.int32)
            codes = torch.empty((c1 - c0, K), dtype=torch.float32, device=device)
            for j in range(PACK):
                codes[:, j::PACK] = ((w32 >> (4 * j)) & 0xF).float()
            wv = (codes - 8.0).view(c1 - c0, G, GROUP).double()
            dot = torch.einsum("mgk,ngk->mng", aval[p].view(M, G, GROUP).double(), wv)
            if as_tok is None:
                # rs carries rowsum*asg (prologue); the correction term has no asg factor
                term = (sc[:, c0:c1].T.double() * dot * asg[p].double().unsqueeze(1)
                        - zsc[:, c0:c1].T.double() * rs[p].double().unsqueeze(1))
            else:
                term = (sc[:, c0:c1].T.double() * dot
                        - zsc[:, c0:c1].T.double() * rs[p].double().unsqueeze(1)
                        ) * as_tok[p].double().view(-1, 1, 1)
            out[:, c0:c1] = term.sum(dim=-1)
    return out.to(torch.bfloat16)


@torch.library.custom_op("radiance::paroquant_linear", mutates_args=())
def paroquant_linear(x: torch.Tensor, qweight: torch.Tensor, sz: torch.Tensor,
                     rec: torch.Tensor, cs: torch.Tensor, pb1: int, pb2: int) -> torch.Tensor:
    """Owns the whole dispatch so no shape branch is visible to dynamo (see the AutoRound module
    for why: a data-dependent M branch in apply() splits the compiled graph at every linear)."""
    N, K = qweight.shape[0], qweight.shape[1] * PACK
    P, krot = rec.shape[0], rec.shape[1]
    G = K // GROUP
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    _ensure_scratch(x.device)
    a_codes = torch.empty((P, M, K), device=x.device, dtype=torch.uint8)
    asg = torch.empty((P, M, G), device=x.device, dtype=torch.float32)
    rs = torch.empty((P, M, G), device=x.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    ptok = PTOK_ENABLED and M > DECODE_MAX_M
    as_tok = None
    if ptok:
        # Prefill: pass A (rotate -> bf16 scratch + per-group scales), pass C (token scale +
        # encode + plain row-sums), PTOK GEMM (AutoRound-cost fold, As in the epilogue).
        xr = torch.empty((P, M, K), device=x.device, dtype=torch.bfloat16)
        as_tok = torch.empty((P, M), device=x.device, dtype=torch.float32)
        _ext.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(), xr.data_ptr(),
                                 asg.data_ptr(), rs.data_ptr(), M, K, P, krot, 1, stream)
        _ext.launch_token_quant(xr.data_ptr(), asg.data_ptr(), a_codes.data_ptr(),
                                as_tok.data_ptr(), rs.data_ptr(), M, K, P, stream)
        gemm_scale = as_tok
    else:
        _ext.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(),
                                 a_codes.data_ptr(), asg.data_ptr(), rs.data_ptr(), M, K, P,
                                 krot, 0, stream)
        gemm_scale = asg
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _ext.launch_gemm(a_codes.data_ptr(), qweight.data_ptr(), sz.data_ptr(),
                     gemm_scale.data_ptr(), rs.data_ptr(), out.data_ptr(), M, N, K, pb1, pb2,
                     1 if ptok else 0, stream)
    if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M \
            and (N, K, M) not in _checked:
        _checked.add((N, K, M))
        ref = _exact_ref(a_codes, asg, rs, qweight, sz, N, K, pb1, pb2, as_tok=as_tok)
        num = (out.float() - ref.float()).pow(2).sum().sqrt()
        den = ref.float().pow(2).sum().sqrt().clamp_min(1e-30)
        sys.stderr.write(f"[radiance.paroquant] CHECKALL N={N} K={K} M={M} P={P} "
                         f"rel={float(num / den):.5f}\n")
    return out.view(*x.shape[:-1], N)


@paroquant_linear.register_fake
def _(x, qweight, sz, rec, cs, pb1, pb2):
    return torch.empty((*x.shape[:-1], qweight.shape[0]), device=x.device, dtype=torch.bfloat16)


def _narrow_tp(target_last, loaded):
    """Slice a rotation tensor along its last (input-channel) dim for row-parallel TP shards.

    Pair indices are local to their 128 group and TP boundaries are multiples of 128, so a plain
    narrow is exact."""
    if target_last == loaded.shape[-1]:
        return loaded
    if loaded.shape[-1] % target_last != 0:
        raise ValueError(f"paroquant rotation loader: incompatible input dims "
                         f"{target_last} vs {loaded.shape[-1]}")
    from vllm.distributed import get_tensor_model_parallel_rank
    return loaded.narrow(-1, get_tensor_model_parallel_rank() * target_last, target_last)


def _rotation_weight_loader(param, loaded_weight, loaded_shard_id=None):
    """Load per-projection rotation params into the partitioned param tensor.

    Shard id conventions (mirrors the upstream paroquant vLLM plugin):
      None        -> single projection, partition 0
      "q"/"k"/"v" -> QKV merge
      int         -> gate/up (or other MergedColumnParallelLinear) index
      tuple       -> fused projections; copy to each listed index
    """
    if loaded_shard_id is None:
        target = param.data[0]
        target.copy_(_narrow_tp(target.shape[-1], loaded_weight).reshape(target.shape))
        return
    indices = (loaded_shard_id if isinstance(loaded_shard_id, tuple)
               else (_SHARD_INDEX.get(loaded_shard_id, loaded_shard_id),))
    for idx in indices:
        target = param.data[idx]
        target.copy_(_narrow_tp(target.shape[-1], loaded_weight).reshape(target.shape))


@register_quantization_config("paroquant")
class ParoQuantConfig(QuantizationConfig):
    """int4 g128 asymmetric + pairwise rotations, W4A8 through the radiance gfx1201 kernels."""

    def __init__(self, bits: int, group_size: int, krot: int, fp16_patterns: list[str]):
        super().__init__()
        if bits != 4:
            raise ValueError(f"radiance paroquant kernel supports 4 bits only, got {bits}")
        if group_size != GROUP:
            raise ValueError(f"radiance paroquant kernel is built for group_size={GROUP}, got "
                             f"{group_size}; the slab structure is aligned to the group.")
        if not (1 <= krot <= KROT_MAX):
            raise ValueError(f"krot={krot} outside the prologue's supported 1..{KROT_MAX}")
        self.bits = bits
        self.group_size = group_size
        self.krot = krot
        self.fp16_patterns = fp16_patterns
        self._fp16_re = [re.compile(p) for p in fp16_patterns]

    def __repr__(self):
        return (f"ParoQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                f"krot={self.krot}, unquantized_patterns={len(self.fp16_patterns)})")

    @classmethod
    def get_name(cls):
        return "paroquant"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0        # gfx1201 does not report a CUDA capability

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "ParoQuantConfig":
        bits = cls.get_from_keys_or(config, ["bits"], 4)
        group_size = cls.get_from_keys_or(config, ["group_size"], GROUP)
        krot = cls.get_from_keys_or(config, ["krot"], 8)
        # This checkpoint family keeps the visual tower and the tiny GDN gate projections in
        # fp16; there is no extra_config in the quant config, so the list is fixed here and
        # extendable via env for future checkpoints.
        pats = [r".*visual.*", r".*in_proj_a.*", r".*in_proj_b.*"]
        extra = os.environ.get("RADIANCE_PQ_SKIP", "").strip()
        pats += [p for p in extra.split(",") if p]
        return cls(bits, group_size, krot, pats)

    def get_quant_method(self, layer, prefix: str):
        if not isinstance(layer, LinearBase):
            return None
        for rx in self._fp16_re:
            if rx.fullmatch(prefix) or rx.search(prefix):
                return UnquantizedLinearMethod()
        return ParoQuantLinearMethod(self)


class ParoQuantLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: ParoQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size, output_size
        out_part = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        g = self.quant_config.group_size
        krot = self.quant_config.krot
        n_parts = len(output_partition_sizes)

        if input_size_per_partition % g:
            raise ValueError(
                f"K per partition ({input_size_per_partition}) is not a multiple of the group "
                f"size ({g}); the group structure would straddle a TP shard boundary.")
        # More than 3 vLLM partitions is fine at LOAD time as long as they collapse to <= 3
        # DISTINCT rotations afterwards -- qwen3_5's in_proj_qkvz has 4 partitions (q,k,v,z) but
        # q/k/v share the single in_proj_qkv rotation. Deduplication and the hard <=3 check
        # happen in process_weights_after_loading.
        for b in output_partition_sizes[:-1]:
            if b % 128:
                raise ValueError(f"partition boundary {b} not a multiple of the 128 n-block; "
                                 "a GEMM block would straddle two rotations.")

        # AWQ layouts; vLLM's own parameter classes handle TP/merge sharding: input_dim=0 shards
        # along K, output_dim=1 along N. fp16 deliberately, not params_dtype (bf16 would quantize
        # the scales themselves).
        qweight = PackedvLLMParameter(
            data=torch.empty(input_size_per_partition, out_part // PACK, dtype=torch.int32),
            input_dim=0, output_dim=1, packed_dim=1, packed_factor=PACK,
            weight_loader=weight_loader)
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

        # Rotation params: one slot per output partition, loaded by shard id.
        for name, shape, dtype in [
            ("theta", (n_parts, krot, input_size_per_partition // 2), torch.float16),
            ("pairs", (n_parts, krot, input_size_per_partition), torch.int16),
            ("channel_scales", (n_parts, input_size_per_partition), torch.float16),
        ]:
            init = torch.ones if name == "channel_scales" else torch.zeros
            p = torch.nn.Parameter(init(shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _rotation_weight_loader
            layer.register_parameter(name, p)

        layer.pq_output_partition_sizes = list(output_partition_sizes)

    def process_weights_after_loading(self, layer) -> None:
        device = layer.qweight.device
        K = layer.qweight.data.shape[0]
        N = layer.scales.data.shape[1]
        G = K // GROUP
        krot = self.quant_config.krot
        inv = _AWQ_INV.to(device)

        def unpack_awq(t):     # [R, C/8] int32 -> [R, C] uint8, AWQ nibble reorder undone
            shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
            v = (t.unsqueeze(-1) >> shifts) & 0xF
            return v[..., inv].reshape(t.shape[0], -1).to(torch.uint8)

        # qweight -> kernel layout [N, K/8] u32 packed along K, low nibble = lowest k.
        codes = unpack_awq(layer.qweight.data).t().contiguous()          # [N, K]
        cw = codes.reshape(N, K // PACK, PACK).to(torch.int64)
        shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int64)
        packed = (cw << shifts).sum(dim=-1)                              # exact: disjoint nibbles
        # int64 -> int32 bit pattern without relying on silent-wrap casts
        qweight = ((packed + 2**31) % 2**32 - 2**31).to(torch.int32).contiguous()

        # scales + zeros -> interleaved SZ [G, N, 2] f16 {scale, scale*(zp-8)}.
        zeros = unpack_awq(layer.qzeros.data).to(torch.float32)          # [G, N]
        sc = layer.scales.data.to(torch.float32)                         # [G, N]
        sz = torch.empty((G, N, 2), dtype=torch.float16, device=device)
        sz[..., 0] = sc.to(torch.float16)
        sz[..., 1] = (sc * (zeros - 8.0)).to(torch.float16)

        # Collapse consecutive vLLM partitions that share one rotation (in_proj_qkvz: q,k,v all
        # carry the in_proj_qkv rotation) into runs; the GEMM selects per DISTINCT rotation.
        sizes_all = layer.pq_output_partition_sizes
        keep = [0]
        run_sizes = [sizes_all[0]]
        for i in range(1, len(sizes_all)):
            same = (torch.equal(layer.theta.data[i], layer.theta.data[keep[-1]])
                    and torch.equal(layer.pairs.data[i], layer.pairs.data[keep[-1]])
                    and torch.equal(layer.channel_scales.data[i],
                                    layer.channel_scales.data[keep[-1]]))
            if same:
                run_sizes[-1] += sizes_all[i]
            else:
                keep.append(i)
                run_sizes.append(sizes_all[i])
        if len(keep) > 3:
            raise ValueError(f"paroquant GEMM partition-select carries at most 3 distinct "
                             f"rotations, got {len(keep)} (sizes {sizes_all})")
        for b in run_sizes[:-1]:
            if b % 128:
                raise ValueError(f"distinct-rotation boundary {b} not a multiple of the 128 "
                                 "n-block")
        layer.pq_output_partition_sizes = run_sizes

        # Rotation records [P, krot, K/2, 4] u16: {i | j<<8, cos f16, sin f16, 0}.
        pairs = layer.pairs.data[keep].to(torch.int64)                   # [P, krot, K]
        if int(pairs.min()) < 0 or int(pairs.max()) >= GROUP:
            raise ValueError("paroquant: pair indices not local to the 128 group")
        theta = layer.theta.data[keep].to(torch.float32)                 # [P, krot, K/2]
        P = pairs.shape[0]
        ij = pairs[..., 0::2] | (pairs[..., 1::2] << 8)                     # [P, krot, K/2]
        rec = torch.zeros((P, krot, K // 2, 4), dtype=torch.int16, device=device)
        rec[..., 0] = ij.to(torch.int16)   # max 127|127<<8 = 32639 < 2^15, no wrap
        rec[..., 1] = torch.cos(theta).to(torch.float16).view(torch.int16)
        rec[..., 2] = torch.sin(theta).to(torch.float16).view(torch.int16)

        cs = layer.channel_scales.data[keep].contiguous()                # [P, K]

        sizes = layer.pq_output_partition_sizes
        layer.pq_pb1 = sizes[0] if len(sizes) > 1 else (1 << 30)
        layer.pq_pb2 = sizes[0] + sizes[1] if len(sizes) > 2 else (1 << 30)

        del layer.qweight, layer.scales, layer.qzeros, layer.theta, layer.pairs
        del layer.channel_scales
        layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.sz = torch.nn.Parameter(sz.contiguous(), requires_grad=False)
        layer.rec = torch.nn.Parameter(rec.contiguous(), requires_grad=False)
        layer.cs = torch.nn.Parameter(cs, requires_grad=False)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.ops.radiance.paroquant_linear(x, layer.qweight, layer.sz, layer.rec,
                                                  layer.cs, layer.pq_pb1, layer.pq_pb2)
        if bias is not None:
            out = out + bias
        return out


if os.environ.get("RADIANCE_PAROQUANT", "0") == "1":
    sys.stderr.write("[radiance.paroquant] registered (int4 g128 asym + rotations, W4A8, "
                     "gfx1201)\n")
