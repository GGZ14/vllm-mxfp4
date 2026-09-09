"""ParoQuant rotations on MXFP4 weights (e2m1 + e8m0/32), W4A8 through the radiance MXFP4 kernel.

Checkpoint (quant_method="paroquant_mxfp4", built by paroquant/build_hybrid.py): per projection the
Quark MXFP4 buffers -- weight [N, K/2] uint8 (two e2m1 per byte, even index in the low nibble) and
weight_scale [N, K/32] uint8 (e8m0, value 2^(E-127)) -- PLUS z-lab's trained rotation: pairs
[krot, K] int16 (group-local Givens pairs), theta [krot, K/2] fp16, channel_scales [1, K] fp16
stored pre-inverted (multiply activations by it).

Why this exists: the prefill GEMM's inner-loop cost is linear in VALU per tile-group, and the int4
path pays two FMAs there (fp16 group scale, asymmetric zero point). MXFP4 has neither -- the e8m0
scale folds at weight staging and e2m1 has no zero point -- so the loop is the 0-VALU one already
used for AMD's MXFP4 release. Same 4.25 bits/weight, so decode is unchanged; this is prefill.

Serving path = the int4 module's per-token prologue composed with the MXFP4 GEMM:
    rotate+scale (pq_rotate_quant, mode 1 -> bf16)  ->  per-token e4m3 (pq_token_quant)
    ->  mxfp4_linear_pq(x_fp8, x_scale, W, Ws, wref)   once per distinct rotation (partition)
The row-sum the int4 prologue also emits is simply unused. Rotations are per projection, so a
merged linear carries P differently-rotated activation copies and the GEMM runs P times on the
matching N-slices (P <= 3, boundaries on multiples of 128, identical adjacent rotations deduped).

A-tiled prefill: the int4 prologue's tiled writer (pq_token_quant_tiled) and the MXFP4 GEMM's
tiled reader use the SAME fragment layout -- [m-tile][k-step][half][row 16][8 B], m-tile-major --
so above RADIANCE_MXFP4_A_TILED_MIN_M the prologue writes tiled A straight into the MXFP4 kernel's
a_tiled register and the GEMM takes its -12..-16% path with no relayout.

Rotation stream: radiance_paroquant.install_stream patches the decoder layers and asks each
consumer linear's quant method for its producers; ours (pqm_add_rms_rot, pqm_ew_rot) fuse the norm /
silu-mul / attention gate / GDN gated norm with rotate + per-TOKEN quant in one launch and hand the
GEMM (A, AS) directly. RADIANCE_PQ_ROT_STREAM=1/ROT_STREAM2=1 turn it on (launcher defaults).
"""
import os
import re
import sys

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

import radiance_mxfp4 as _mx                      # the MXFP4 GEMM, its layout helpers, its gates
import radiance_paroquant as _pq                  # rotation records, TP-aware rotation loaders
import radiance_paroquant_kernel as _pqk          # the rotation / token-quant prologue kernels

GROUP, KROT_MAX, MXBLOCK = _pq.GROUP, _pq.KROT_MAX, 32

# In-serve numerics gate, same contract as the int4 module's: "N:K,N:K" compares each partition's
# GEMM against an fp32 dequant of the same codes for calls at or below CHECK_MAX_M rows. Needs
# --enforce-eager. This is what catches the e8m0 fold flushing small codes on a checkpoint whose
# exponent spread differs from AMD's.
_ca = os.environ.get("RADIANCE_PQM_CHECKALL", "").strip()
CHECK_ALL = ({tuple(int(v) for v in p.split(":")) for p in _ca.split(",") if p} if _ca else None)
CHECK_MAX_M = int(os.environ.get("RADIANCE_PQM_CHECK_MAX_M", "128"))
# Single-launch rotate + per-token quant for the non-tiled path (default on).
FUSED_TOKQ = os.environ.get("RADIANCE_PQM_FUSED_TOKQ", "1") == "1"
# One GEMM launch per merged linear (partition select by n-block in the kernel) instead of one per
# rotated partition plus an output cat (default on; 0 = the per-partition loop, kept for A/Bs).
SINGLE_LAUNCH = os.environ.get("RADIANCE_PQM_SINGLE_LAUNCH", "1") == "1"
_checked: set = set()


def _ensure_mxfp4_decode_scratch(device) -> None:
    """The MXFP4 decode kernel's split-K slab, allocated at weight-load time.

    Mirrors what the MXFP4 kernel class does on its first layer: it cannot be lazy, because with a
    warm compile cache the first GEMM runs under CUDA-graph capture, where hipMalloc is illegal.
    Our layers never pass through that class, so we do it here, guarded by the same flag."""
    if _mx._decode_scratch_ready[0] or _mx.DECODE_MAX_M <= 0:
        return
    _mx._decode_scratch_ready[0] = True
    _mx._decode_scratch[0] = torch.empty(4 * max(64, _mx.DECODE_MAX_M) * 32768,
                                         dtype=torch.float32, device=device)
    _mx._decode_scratch[1] = torch.zeros(32768 // 128 + 8, dtype=torch.int32, device=device)
    _mx._ext.set_decode_scratch(_mx._decode_scratch[0].data_ptr(),
                                _mx._decode_scratch[0].numel() * 4,
                                _mx._decode_scratch[1].data_ptr())
    sys.stderr.write(f"[radiance.paroquant_mxfp4] MXFP4 decode kernel ON (M<={_mx.DECODE_MAX_M}), "
                     f"{_mx._decode_scratch[0].numel() * 4 >> 20} MiB split-K scratch\n")


def _partitions(N: int, pb1: int, pb2: int):
    bounds = [0, min(pb1, N), min(pb2, N), N]
    return [(bounds[i], bounds[i + 1]) for i in range(3) if bounds[i + 1] > bounds[i]]


def _linear_impl(x2, weight, ws_t, wref, rec, cs, pb1, pb2, pre=None):
    """Whole dispatch, opaque to dynamo (a data-dependent M branch in apply() would split the
    compiled graph at every linear)."""
    N, K = weight.shape[0], weight.shape[1] * 2
    P, krot = rec.shape[0], rec.shape[1]
    G, G32 = K // GROUP, K // MXBLOCK
    M = x2.shape[0]
    _pq._ensure_scratch(x2.device)
    stream = torch.cuda.current_stream().cuda_stream

    tiled = _tiled(M)
    if pre is not None:
        # rotation stream: the producer (pqm_add_rms_rot / pqm_ew_rot) already rotated + token-
        # quantized this linear's input, in the tiled layout when M is in the A-tiled band (the
        # producer takes the same _tiled(M) decision); x2 (hs) is not read
        a_codes, as_tok = pre
        if a_codes.shape[0] != P:
            raise RuntimeError(f"paroquant_mxfp4: pre-quantized tuple has {a_codes.shape[0]} "
                               f"partition(s), layer has {P} -- a producer was hooked to the wrong linear")
        as_tok = as_tok.contiguous()
        if tiled and a_codes.shape[1] != ((M + 15) // 16) * 16 * K:
            raise RuntimeError("paroquant_mxfp4: stream tuple is not in the tiled layout the consumer expects")
    elif tiled:
        as_tok = torch.empty((P, M), device=x2.device, dtype=torch.float32)
        Mt = (M + 15) // 16
        a_codes = torch.empty((P, Mt * 16 * K), device=x2.device, dtype=torch.uint8)
        if FUSED_TOKQ:
            # prefill: the fused kernel writes the fragment-tiled layout the A-tiled GEMM reads
            # directly (par_harness tokqt: byte-exact vs pass A + tiled pass C, 1.3-1.4x faster)
            _pqk.launch_rotate_tokquant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(),
                                        a_codes.data_ptr(), as_tok.data_ptr(), M, K, P, krot, stream, 1)
        else:
            # pass A (rotate, bf16 out) then tiled pass C -- the A/B path
            xr = torch.empty((P, M, K), device=x2.device, dtype=torch.bfloat16)
            asg = torch.empty((P, M, G), device=x2.device, dtype=torch.float32)
            rs = torch.empty((P, M, G), device=x2.device, dtype=torch.float32)
            _pqk.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(), xr.data_ptr(),
                                     asg.data_ptr(), rs.data_ptr(), M, K, P, krot, 1, stream)
            _pqk.launch_token_quant(xr.data_ptr(), asg.data_ptr(), a_codes.data_ptr(),
                                    as_tok.data_ptr(), rs.data_ptr(), M, K, P, stream, 1)
    else:
        as_tok = torch.empty((P, M), device=x2.device, dtype=torch.float32)
        # decode band and row-major prefill: ONE launch does channel-scale + rotate + token amax
        # + e4m3 encode, bit-identical to pass A + pass C (par_harness tokq). Saves a launch and
        # the HBM round trip of the rotated row per linear; RADIANCE_PQM_FUSED_TOKQ=0 falls back.
        a_codes = torch.empty((P, M, K), device=x2.device, dtype=torch.uint8)
        if FUSED_TOKQ:
            _pqk.launch_rotate_tokquant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(),
                                        a_codes.data_ptr(), as_tok.data_ptr(), M, K, P, krot, stream)
        else:
            xr = torch.empty((P, M, K), device=x2.device, dtype=torch.bfloat16)
            asg = torch.empty((P, M, G), device=x2.device, dtype=torch.float32)
            rs = torch.empty((P, M, G), device=x2.device, dtype=torch.float32)
            _pqk.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(), xr.data_ptr(),
                                     asg.data_ptr(), rs.data_ptr(), M, K, P, krot, 1, stream)
            _pqk.launch_token_quant(xr.data_ptr(), asg.data_ptr(), a_codes.data_ptr(),
                                    as_tok.data_ptr(), rs.data_ptr(), M, K, P, stream, 0)

    if SINGLE_LAUNCH:
        # one GEMM launch over the whole N: partition p's n-blocks read rotated copy p of A
        # (radiance_mxfp4_fp8 launch_p). Merged linears (qkv P=3, gate_up / in_proj P=2) thus cost
        # one launch and no output concatenation; boundaries are 128-aligned (loader-checked).
        if tiled:
            _mx.a_tiled_register(a_codes, M, K)
        out = torch.ops.radiance.mxfp4_linear_pqp(a_codes, as_tok, weight, ws_t, wref, M, pb1, pb2)
        if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M:
            for p, (n0, n1) in enumerate(_partitions(N, pb1, pb2)):
                if (N, K, M, p) in _checked:
                    continue
                _checked.add((N, K, M, p))
                x_rm = _pq.untile_a(a_codes, P, M, K)[p] if tiled else a_codes[p]
                ws_p = ws_t[:, n0:n1].contiguous()
                ref = _mx._exact_ref(x_rm.view(torch.float8_e4m3fn), as_tok[p].view(M, 1),
                                     weight[n0:n1], ws_p, n1 - n0, K)
                y = out[:, n0:n1]
                num = (y.float() - ref.float()).pow(2).sum().sqrt()
                den = ref.float().pow(2).sum().sqrt()
                verdict = "zero-input" if float(den) < 1e-20 else f"rel={float(num / den.clamp_min(1e-30)):.5f}"
                sys.stderr.write(f"[radiance.paroquant_mxfp4] CHECKALL N={N} K={K} M={M} P={P} part={p} "
                                 f"path={'tiled' if tiled else 'rowmajor'}+single"
                                 f"{'+pre' if pre is not None else ''} {verdict}\n")
        return out
    ys = []
    for p, (n0, n1) in enumerate(_partitions(N, pb1, pb2)):
        w_p = weight[n0:n1]                                              # rows: contiguous
        ws_p = ws_t[:, n0:n1].contiguous()      # A/B path only: a column slice is not contiguous
        if tiled:
            # an [M, K] view over the padded tiled storage: the kernel reads by data_ptr and its
            # own tiled addressing; the shape only carries M (see mxfp4_linear_pq)
            x_p = a_codes[p].narrow(0, 0, M * K).view(M, K)
            _mx.a_tiled_register(x_p, M, K)
        else:
            x_p = a_codes[p]
        y = torch.ops.radiance.mxfp4_linear_pq(x_p, as_tok[p], w_p, ws_p, wref[n0:n1])
        ys.append(y)
        if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M \
                and (N, K, M, p) not in _checked:
            _checked.add((N, K, M, p))
            # _exact_ref un-permutes the fragment-order weight ITSELF under WPERM; passing an
            # already un-permuted copy double-applied the inverse and reported the kernel as wrong
            # at rel 1.6-9 while the served model was answering correctly.
            x_rm = _pq.untile_a(a_codes, P, M, K)[p] if tiled else a_codes[p]
            ref = _mx._exact_ref(x_rm.view(torch.float8_e4m3fn), as_tok[p].view(M, 1),
                                 w_p, ws_p, n1 - n0, K)
            num = (y.float() - ref.float()).pow(2).sum().sqrt()
            den = ref.float().pow(2).sum().sqrt()
            # vLLM's profile run feeds zero activations: both sides are 0 and rel prints 0.00000,
            # which proves nothing. Say so instead of looking like a pass.
            verdict = "zero-input" if float(den) < 1e-20 else f"rel={float(num / den.clamp_min(1e-30)):.5f}"
            sys.stderr.write(f"[radiance.paroquant_mxfp4] CHECKALL N={N} K={K} M={M} P={P} "
                             f"part={p} path={'tiled' if tiled else 'rowmajor'}"
                             f"{'+pre' if pre is not None else ''} {verdict}\n")
    # one output tensor per linear: the single-partition case IS the GEMM output (no copy), merged
    # linears pay one cat instead of P slice copies
    return ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)


@torch.library.custom_op("radiance::paroquant_mxfp4_linear", mutates_args=())
def paroquant_mxfp4_linear(x: torch.Tensor, weight: torch.Tensor, ws_t: torch.Tensor,
                           wref: torch.Tensor, rec: torch.Tensor, cs: torch.Tensor,
                           pb1: int, pb2: int) -> torch.Tensor:
    K = weight.shape[1] * 2
    out = _linear_impl(x.reshape(-1, K), weight, ws_t, wref, rec, cs, pb1, pb2)
    return out.view(*x.shape[:-1], weight.shape[0])


@paroquant_mxfp4_linear.register_fake
def _(x, weight, ws_t, wref, rec, cs, pb1, pb2):
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=torch.bfloat16)


@torch.library.custom_op("radiance::paroquant_mxfp4_linear_pre", mutates_args=())
def paroquant_mxfp4_linear_pre(hs: torch.Tensor, a: torch.Tensor, as_tok: torch.Tensor,
                               weight: torch.Tensor, ws_t: torch.Tensor, wref: torch.Tensor,
                               rec: torch.Tensor, cs: torch.Tensor, pb1: int, pb2: int) -> torch.Tensor:
    """Linear on the per-token rotation-stream tuple: (A [P, M, K], AS [P, M]) from a pqm_*
    producer below the tiled threshold; hs (bf16) for the tiled prefill path above it."""
    K = weight.shape[1] * 2
    out = _linear_impl(hs.reshape(-1, K), weight, ws_t, wref, rec, cs, pb1, pb2, pre=(a, as_tok))
    return out.view(*hs.shape[:-1], weight.shape[0])


@paroquant_mxfp4_linear_pre.register_fake
def _(hs, a, as_tok, weight, ws_t, wref, rec, cs, pb1, pb2):
    return torch.empty((*hs.shape[:-1], weight.shape[0]), device=hs.device, dtype=torch.bfloat16)


def _tiled(M: int) -> bool:
    """Producer and consumer take the SAME layout decision from M: fragment-tiled A in the A-tiled
    GEMM band, row-major below it."""
    return bool(_mx.A_TILED_MIN_M) and M >= _mx.A_TILED_MIN_M


@torch.library.custom_op("radiance::pqm_add_rms_rot", mutates_args=())
def pqm_add_rms_rot(y: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                    rec: torch.Tensor, cs: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add + Gemma RMSNorm + rotate + per-TOKEN e4m3 quant for the MXFP4 consumer ->
    (hs, residual_out, A, AS [P, M]). A is [P, M, K] row-major below the A-tiled band and the
    fragment-tiled [P, Mt*16*K] slab in it; the consumer takes the same decision from M."""
    y2 = y.reshape(-1, y.shape[-1])
    M, K = y2.shape
    P = rec.shape[0]
    res = residual.reshape(M, K)
    if not res.is_contiguous():
        res = res.contiguous()
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    hs = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    ro = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    tiled = _tiled(M)
    a = torch.empty((P, ((M + 15) // 16) * 16 * K) if tiled else (P, M, K), device=y.device, dtype=torch.uint8)
    as_tok = torch.empty((P, M), device=y.device, dtype=torch.float32)
    _pqk.launch_add_rms_rot_tok(y2.data_ptr(), res.data_ptr(), weight.data_ptr(), float(eps),
                                rec.data_ptr(), cs.data_ptr(), hs.data_ptr(), ro.data_ptr(),
                                a.data_ptr(), as_tok.data_ptr(), M, K, P, rec.shape[1],
                                torch.cuda.current_stream().cuda_stream, 1 if tiled else 0)
    return hs.view(y.shape), ro.view(residual.shape), a, as_tok


@pqm_add_rms_rot.register_fake
def _(y, residual, weight, eps, rec, cs):
    K = y.shape[-1]
    M = y.numel() // K
    P = rec.shape[0]
    return (torch.empty(y.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty(residual.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty((P, ((M + 15) // 16) * 16 * K) if _tiled(M) else (P, M, K), device=y.device, dtype=torch.uint8),
            torch.empty((P, M), device=y.device, dtype=torch.float32))


@torch.library.custom_op("radiance::pqm_ew_rot", mutates_args=())
def pqm_ew_rot(mode: int, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float,
               rec: torch.Tensor, cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Producer + rotate + per-token quant for the single-partition sites: mode 0 silu-mul
    (x = gate_up [M, 2N]), mode 1 attention gate (x * sigmoid(y)), mode 2 GDN gated rmsnorm
    (x, z = y, w [128]). Returns (hs, A [1, M, N], AS [1, M])."""
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    N = x2.shape[1] // 2 if mode == 0 else x2.shape[1]
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if mode == 0:
        y2, ys = x2, 0
    else:
        y2 = y.reshape(M, -1)
        if y2.stride(-1) != 1 or (y2.stride(0) & 7):
            y2 = y2.contiguous()
        ys = y2.stride(0)
    hs = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    tiled = _tiled(M)
    a = torch.empty((1, ((M + 15) // 16) * 16 * N) if tiled else (1, M, N), device=x.device, dtype=torch.uint8)
    as_tok = torch.empty((1, M), device=x.device, dtype=torch.float32)
    _pqk.launch_ew_rot_tok(mode, x2.data_ptr(), y2.data_ptr(), ys, w.data_ptr(), float(eps),
                           rec.data_ptr(), cs.data_ptr(), hs.data_ptr(), a.data_ptr(),
                           as_tok.data_ptr(), M, N, rec.shape[1],
                           torch.cuda.current_stream().cuda_stream, 1 if tiled else 0)
    return hs, a, as_tok


@pqm_ew_rot.register_fake
def _(mode, x, y, w, eps, rec, cs):
    Kx = x.shape[-1]
    M = x.numel() // Kx
    N = Kx // 2 if mode == 0 else Kx
    return (torch.empty((M, N), device=x.device, dtype=torch.bfloat16),
            torch.empty((1, ((M + 15) // 16) * 16 * N) if _tiled(M) else (1, M, N), device=x.device, dtype=torch.uint8),
            torch.empty((1, M), device=x.device, dtype=torch.float32))


@register_quantization_config("paroquant_mxfp4")
class ParoQuantMXFP4Config(QuantizationConfig):
    """MXFP4 weights + pairwise rotations, W4A8 through the radiance MXFP4 kernel."""

    def __init__(self, bits: int, group_size: int, krot: int, fp16_patterns: list[str]):
        super().__init__()
        if bits != 4:
            raise ValueError(f"paroquant_mxfp4 is 4-bit by definition, got {bits}")
        if group_size != GROUP:
            raise ValueError(f"rotation group must be {GROUP}, got {group_size}")
        if not (1 <= krot <= KROT_MAX):
            raise ValueError(f"krot={krot} outside the prologue's supported 1..{KROT_MAX}")
        self.bits, self.group_size, self.krot = bits, group_size, krot
        self.fp16_patterns = fp16_patterns
        self._fp16_re = [re.compile(p) for p in fp16_patterns]

    def __repr__(self):
        return f"ParoQuantMXFP4Config(krot={self.krot}, unquantized_patterns={len(self.fp16_patterns)})"

    @classmethod
    def get_name(cls):
        return "paroquant_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "ParoQuantMXFP4Config":
        bits = cls.get_from_keys_or(config, ["bits"], 4)
        group_size = cls.get_from_keys_or(config, ["group_size"], GROUP)
        krot = cls.get_from_keys_or(config, ["krot"], 8)
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
        return ParoQuantMXFP4LinearMethod(self)


class ParoQuantMXFP4LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: ParoQuantMXFP4Config):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size, output_size, params_dtype
        out_part = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        krot = self.quant_config.krot
        n_parts = len(output_partition_sizes)
        K = input_size_per_partition

        if K % GROUP:
            raise ValueError(f"K per partition ({K}) is not a multiple of the rotation group "
                             f"({GROUP}); a TP shard would straddle a group.")
        for b in output_partition_sizes[:-1]:
            if b % 128:
                raise ValueError(f"partition boundary {b} not a multiple of the 128 n-block")

        # Quark MXFP4 layout, as on disk. Row-parallel narrowing along input_dim uses the param's
        # own size (K/2 bytes, K/32 exponents), which is exactly the packed shard.
        weight = ModelWeightParameter(
            data=torch.empty(out_part, K // 2, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(out_part, K // MXBLOCK, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)

        # Rotation params and their TP-aware loader are the int4 module's, unchanged.
        for name, shape, dtype in [
            ("theta", (n_parts, krot, K // 2), torch.float16),
            ("pairs", (n_parts, krot, K), torch.int16),
            ("channel_scales", (n_parts, K), torch.float16),
        ]:
            init = torch.ones if name == "channel_scales" else torch.zeros
            p = torch.nn.Parameter(init(shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _pq._rotation_weight_loader
            layer.register_parameter(name, p)
        layer.pq_output_partition_sizes = list(output_partition_sizes)

    def process_weights_after_loading(self, layer) -> None:
        device = layer.weight.device
        N, K = layer.weight.shape[0], layer.weight.shape[1] * 2
        krot = self.quant_config.krot
        _ensure_mxfp4_decode_scratch(device)

        # ---- rotations: dedup identical adjacent partitions, build kernel records (as int4) ----
        sizes_all = layer.pq_output_partition_sizes
        keep, run_sizes = [0], [sizes_all[0]]
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
            raise ValueError(f"at most 3 distinct rotations per linear, got {len(keep)}")
        for b in run_sizes[:-1]:
            if b % 128:
                raise ValueError(f"distinct-rotation boundary {b} not a multiple of 128")
        pairs = layer.pairs.data[keep].to(torch.int64)
        if int(pairs.min()) < 0 or int(pairs.max()) >= GROUP:
            raise ValueError("pair indices not local to the 128 group")
        theta = layer.theta.data[keep].to(torch.float32)
        P = pairs.shape[0]
        ij = pairs[..., 0::2] | (pairs[..., 1::2] << 8)
        rec = torch.zeros((P, krot, K // 2, 4), dtype=torch.int16, device=device)
        rec[..., 0] = ij.to(torch.int16)
        rec[..., 1] = torch.cos(theta).to(torch.float16).view(torch.int16)
        rec[..., 2] = torch.sin(theta).to(torch.float16).view(torch.int16)
        cs = layer.channel_scales.data[keep].contiguous()
        layer.pq_pb1 = run_sizes[0] if len(run_sizes) > 1 else (1 << 30)
        layer.pq_pb2 = run_sizes[0] + run_sizes[1] if len(run_sizes) > 2 else (1 << 30)

        # ---- MXFP4 weight, prepared exactly as the MXFP4 kernel class prepares AMD's ----
        ws_t = layer.weight_scale.data.T.contiguous()                    # [K/32, N]
        wref = _mx.make_row_ref(ws_t)                                    # [N] e8m0 row max
        # the full [K/32, N] is what the single-launch GEMM reads (partition select in-kernel);
        # the per-partition A/B loop slices it on the fly
        w = layer.weight.data
        if _mx.WPERM:
            if N % 16 or K % 16:
                raise RuntimeError(f"RADIANCE_MXFP4_WPERM needs N,K divisible by 16, got {N},{K}")
            w = _mx.permute_w(w, N, K)       # 16-row tiles never straddle a 128-aligned boundary

        del layer.weight, layer.weight_scale, layer.theta, layer.pairs, layer.channel_scales
        layer.weight = torch.nn.Parameter(w.contiguous(), requires_grad=False)
        layer.ws_t = torch.nn.Parameter(ws_t, requires_grad=False)          # full [K/32, N]
        for b in (layer.pq_pb1, layer.pq_pb2):
            if b < N and b % 128:
                raise RuntimeError(f"paroquant_mxfp4: partition boundary {b} not 128-aligned (N={N})")
        layer.wref = torch.nn.Parameter(wref, requires_grad=False)
        layer.rec = torch.nn.Parameter(rec.contiguous(), requires_grad=False)
        layer.cs = torch.nn.Parameter(cs, requires_grad=False)

    # rotation stream (radiance_paroquant.install_stream): the norm / silu-mul / gate / gdn-norm
    # producers of THIS method build the per-token tuple (hs, A, AS) its GEMM consumes
    pq_stream_capable = True
    pq_ar_capable = False

    @staticmethod
    def stream_norm(y, residual, weight, eps, cons):
        hs, ro, a, as_tok = torch.ops.radiance.pqm_add_rms_rot(y, residual, weight, eps, cons.rec, cons.cs)
        return hs, ro, (a, as_tok)

    @staticmethod
    def stream_ew(mode, x, y, w, eps, cons):
        hs, a, as_tok = torch.ops.radiance.pqm_ew_rot(mode, x, y, w, eps, cons.rec, cons.cs)
        return hs, (a, as_tok)

    def apply(self, layer, x, bias: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(x, tuple):          # rotation stream: (hs, A, AS)
            hs, a, as_tok = x
            out = torch.ops.radiance.paroquant_mxfp4_linear_pre(hs, a, as_tok, layer.weight,
                                                                layer.ws_t, layer.wref, layer.rec,
                                                                layer.cs, layer.pq_pb1, layer.pq_pb2)
        else:
            out = torch.ops.radiance.paroquant_mxfp4_linear(x, layer.weight, layer.ws_t, layer.wref,
                                                            layer.rec, layer.cs, layer.pq_pb1,
                                                            layer.pq_pb2)
        if bias is not None:
            out = out + bias
        return out


if os.environ.get("RADIANCE_PAROQUANT", "0") == "1":
    sys.stderr.write("[radiance.paroquant_mxfp4] registered (e2m1+e8m0/32 weights, z-lab rotations, "
                     f"per-token W4A8 -> MXFP4 GEMM; fused prologue {'on' if FUSED_TOKQ else 'off'}, single launch {'on' if SINGLE_LAUNCH else 'off'}, "
                     f"rot stream {'on' if _pq.ROT_STREAM else 'off'}{'+2' if _pq.ROT_STREAM2 else ''}, WPERM={'on' if _mx.WPERM else 'off'}, "
                     f"decode band M<={_mx.DECODE_MAX_M})\n")
