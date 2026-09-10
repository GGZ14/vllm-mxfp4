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
if DECODE_MAX_M > _DEC_MAX_M:
    _DEC_MAX_M = DECODE_MAX_M          # split-K scratch must cover the widest decode row count
PTOK_ENABLED = os.environ.get("RADIANCE_PQ_PTOK", "1") == "1"
# Prefill GEMM on a fragment-tiled activation (pass C writes the tile layout, the GEMM reads A
# straight from global into the WMMA register; no A tile in LDS). Harness-gated 2026-09-02.
ATILED_ENABLED = os.environ.get("RADIANCE_PQ_ATILED", "1") == "1"
# Fused prefill prologue: pass A + pass C as one launch (pq_rotate_tokquant with row-sums), byte-exact
# (par_harness tokqrs). 0 = the two-pass path.
FUSED_TOKQ = os.environ.get("RADIANCE_PQ_FUSED_TOKQ", "1") == "1"
# Weight layout: fragment order (one 32-lane u32 slot per (n-tile, k-step)) rather than the
# loader's [N, K/8]. The kernels read RADIANCE_PQ_WPERM themselves; this flag and theirs MUST
# agree or the weight is read as garbage. Fragment order is what makes the decode kernel's
# streaming (NT) loads pay (RADIANCE_PQ_DECODE_NT, honoured only under WPERM).
WPERM = os.environ.get("RADIANCE_PQ_WPERM", "1") == "1"
# Rotation stream: the decode-band producers of the norm-fed linears (input_layernorm -> qkv /
# in_proj_qkvz, post_attention_layernorm -> gate_up) run residual-add + RMSNorm + rotate + quant
# as ONE kernel and hand the linear a (hs, A, ASG, RS) tuple; 96 rotation launches per step gone
# and the norm itself is one kernel instead of inductor's pair. Installed by install_stream()
# from the shared post-load hook (radiance_gdnmerge.merge_model). Changes the traced graph:
# the compile cache must be keyed (run_paroquant.sh adds -rs).
ROT_STREAM = os.environ.get("RADIANCE_PQ_ROT_STREAM", "0") == "1"
# Stream 2: the three single-partition producers (silu-mul -> down_proj, GDN gated norm ->
# out_proj, attention gate -> o_proj) fused with rotate + quant the same way. Needs ROT_STREAM.
ROT_STREAM2 = ROT_STREAM and os.environ.get("RADIANCE_PQ_ROT_STREAM2", "0") == "1"
_checked = set()


def permute_w(qweight: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """[N, K/8] int32 (row layout) -> fragment order, same shape. Byte i of a row holds codes
    2i (low nibble) and 2i+1; lane l of tile (nt, ks) takes the 4 bytes at k = ks*16 + (l>>4)*8
    of row nt*16 + (l & 15)."""
    nt, ks = N // 16, K // 16
    return (qweight.view(torch.uint8).view(nt, 16, ks, 2, 4)   # [n-tile][row][k-step][half][4 B]
                   .permute(0, 2, 3, 1, 4)                     # [n-tile][k-step][half][row][4 B]
                   .contiguous().view(N, K // 2).view(torch.int32).view(N, K // 8))


def unpermute_w(qweight: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of permute_w (the exact-reference path dequantizes the ROW layout)."""
    nt, ks = N // 16, K // 16
    return (qweight.view(torch.uint8).view(nt, ks, 2, 16, 4)
                   .permute(0, 3, 1, 2, 4)
                   .contiguous().view(N, K // 2).view(torch.int32).view(N, K // 8))


def untile_a(at: torch.Tensor, P: int, M: int, K: int) -> torch.Tensor:
    """[P, Mt*16*K] fragment-tiled codes -> [P, M, K] row-major (reference path only)."""
    Mt, ks = (M + 15) // 16, K // 16
    return (at.view(P, Mt, ks, 2, 16, 8)          # [p][m-tile][k-step][half][row][8 B]
              .permute(0, 1, 4, 2, 3, 5)          # [p][m-tile][row][k-step][half][8 B]
              .reshape(P, Mt * 16, K)[:, :M].contiguous())

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


def _linear_impl(x2, qweight, sz, rec, cs, pb1, pb2, pre=None):
    """The whole dispatch, opaque to dynamo. pre = (A, ASG, RS) already rotated+quantized by the
    fused norm producer (decode band only; ignored -- recomputed from x2 -- above the band)."""
    N, K = qweight.shape[0], qweight.shape[1] * PACK
    P, krot = rec.shape[0], rec.shape[1]
    G = K // GROUP
    x = x2
    M = x2.shape[0]
    _ensure_scratch(x.device)
    stream = torch.cuda.current_stream().cuda_stream
    ptok = PTOK_ENABLED and M > DECODE_MAX_M
    tiled = ptok and ATILED_ENABLED
    as_tok = None
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    if pre is not None and not ptok:
        a_codes, asg, rs = pre
        if a_codes.shape[0] != P:
            raise RuntimeError(f"paroquant: pre-quantized tuple has {a_codes.shape[0]} partition(s), "
                               f"layer has {P} -- a producer was hooked to the wrong linear")
    else:
        a_codes = torch.empty((P, M, K), device=x.device, dtype=torch.uint8)
        asg = torch.empty((P, M, G), device=x.device, dtype=torch.float32)
        rs = torch.empty((P, M, G), device=x.device, dtype=torch.float32)
    if pre is not None and not ptok:
        gemm_scale = asg
    elif ptok:
        # Prefill: pass A (rotate -> bf16 scratch + per-group scales), pass C (token scale +
        # encode + plain row-sums), PTOK GEMM (AutoRound-cost fold, As in the epilogue).
        # Tiled: pass C emits the fragment-tiled layout (16-row padded) and the A-direct GEMM
        # consumes it; row-major otherwise.
        as_tok = torch.empty((P, M), device=x.device, dtype=torch.float32)
        if tiled:
            Mt = (M + 15) // 16
            a_codes = torch.empty((P, Mt * 16 * K), device=x.device, dtype=torch.uint8)
        if FUSED_TOKQ:
            # one launch: rotate -> token amax -> encode (+ tiled layout) + plain row-sums
            _ext.launch_rotate_tokquant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(),
                                        a_codes.data_ptr(), as_tok.data_ptr(), M, K, P, krot,
                                        stream, 1 if tiled else 0, rs.data_ptr())
        else:
            xr = torch.empty((P, M, K), device=x.device, dtype=torch.bfloat16)
            _ext.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(), xr.data_ptr(),
                                     asg.data_ptr(), rs.data_ptr(), M, K, P, krot, 1, stream)
            _ext.launch_token_quant(xr.data_ptr(), asg.data_ptr(), a_codes.data_ptr(),
                                    as_tok.data_ptr(), rs.data_ptr(), M, K, P, stream,
                                    1 if tiled else 0)
        gemm_scale = as_tok
    else:
        _ext.launch_rotate_quant(x2.data_ptr(), rec.data_ptr(), cs.data_ptr(),
                                 a_codes.data_ptr(), asg.data_ptr(), rs.data_ptr(), M, K, P,
                                 krot, 0, stream)
        gemm_scale = asg
    if tiled:
        _ext.launch_gemm_at(a_codes.data_ptr(), qweight.data_ptr(), sz.data_ptr(),
                            gemm_scale.data_ptr(), rs.data_ptr(), out.data_ptr(), M, N, K,
                            pb1, pb2, stream)
    else:
        _ext.launch_gemm(a_codes.data_ptr(), qweight.data_ptr(), sz.data_ptr(),
                         gemm_scale.data_ptr(), rs.data_ptr(), out.data_ptr(), M, N, K, pb1,
                         pb2, 1 if ptok else 0, stream)
    if CHECK_ALL is not None and (N, K) in CHECK_ALL and M <= CHECK_MAX_M \
            and (N, K, M) not in _checked:
        _checked.add((N, K, M))
        a_ref = untile_a(a_codes, P, M, K) if tiled else a_codes
        w_ref = unpermute_w(qweight, N, K) if WPERM else qweight
        ref = _exact_ref(a_ref, asg, rs, w_ref, sz, N, K, pb1, pb2, as_tok=as_tok)
        num = (out.float() - ref.float()).pow(2).sum().sqrt()
        den = ref.float().pow(2).sum().sqrt().clamp_min(1e-30)
        sys.stderr.write(f"[radiance.paroquant] CHECKALL N={N} K={K} M={M} P={P} "
                         f"path={'tiled' if tiled else 'ptok' if ptok else 'decode'}"
                         f"{'+pre' if pre is not None and not ptok else ''} "
                         f"rel={float(num / den):.5f}\n")
    return out


@torch.library.custom_op("radiance::paroquant_linear", mutates_args=())
def paroquant_linear(x: torch.Tensor, qweight: torch.Tensor, sz: torch.Tensor,
                     rec: torch.Tensor, cs: torch.Tensor, pb1: int, pb2: int) -> torch.Tensor:
    """Owns the whole dispatch so no shape branch is visible to dynamo (see the AutoRound module
    for why: a data-dependent M branch in apply() splits the compiled graph at every linear)."""
    K = qweight.shape[1] * PACK
    out = _linear_impl(x.reshape(-1, K), qweight, sz, rec, cs, pb1, pb2)
    return out.view(*x.shape[:-1], qweight.shape[0])


@paroquant_linear.register_fake
def _(x, qweight, sz, rec, cs, pb1, pb2):
    return torch.empty((*x.shape[:-1], qweight.shape[0]), device=x.device, dtype=torch.bfloat16)


@torch.library.custom_op("radiance::paroquant_linear_pre", mutates_args=())
def paroquant_linear_pre(hs: torch.Tensor, a: torch.Tensor, asg: torch.Tensor, rs: torch.Tensor,
                         qweight: torch.Tensor, sz: torch.Tensor, rec: torch.Tensor,
                         cs: torch.Tensor, pb1: int, pb2: int) -> torch.Tensor:
    """Linear on the rotation-stream tuple: (A, ASG, RS) from pq_add_rms_rot in the decode band,
    hs (bf16) for the prefill path above it."""
    K = qweight.shape[1] * PACK
    out = _linear_impl(hs.reshape(-1, K), qweight, sz, rec, cs, pb1, pb2, pre=(a, asg, rs))
    return out.view(*hs.shape[:-1], qweight.shape[0])


@paroquant_linear_pre.register_fake
def _(hs, a, asg, rs, qweight, sz, rec, cs, pb1, pb2):
    return torch.empty((*hs.shape[:-1], qweight.shape[0]), device=hs.device, dtype=torch.bfloat16)


@torch.library.custom_op("radiance::pq_add_rms_rot", mutates_args=())
def pq_add_rms_rot(y: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                   rec: torch.Tensor, cs: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add + Gemma RMSNorm (+ rotate + per-group quant in the decode band) -> (hs,
    residual_out, A, ASG, RS). Above the band A/ASG/RS are allocated but untouched; the consumer
    recomputes from hs on the tiled prefill path. The M branch lives here, opaque to dynamo."""
    y2 = y.reshape(-1, y.shape[-1])
    M, K = y2.shape
    P = rec.shape[0]
    G = K // GROUP
    res = residual.reshape(M, K)
    if not res.is_contiguous():
        res = res.contiguous()
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    hs = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    ro = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    a = torch.empty((P, M, K), device=y.device, dtype=torch.uint8)
    asg = torch.empty((P, M, G), device=y.device, dtype=torch.float32)
    rs = torch.empty((P, M, G), device=y.device, dtype=torch.float32)
    fused = M <= DECODE_MAX_M
    _ext.launch_add_rms_rot(y2.data_ptr(), res.data_ptr(), weight.data_ptr(), float(eps),
                            rec.data_ptr(), cs.data_ptr(), hs.data_ptr(), ro.data_ptr(),
                            a.data_ptr(), asg.data_ptr(), rs.data_ptr(), M, K, P, rec.shape[1],
                            1 if fused else 0, torch.cuda.current_stream().cuda_stream)
    return hs.view(y.shape), ro.view(residual.shape), a, asg, rs


@pq_add_rms_rot.register_fake
def _(y, residual, weight, eps, rec, cs):
    K = y.shape[-1]
    M = y.numel() // K
    P = rec.shape[0]
    return (torch.empty(y.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty(residual.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty((P, M, K), device=y.device, dtype=torch.uint8),
            torch.empty((P, M, K // GROUP), device=y.device, dtype=torch.float32),
            torch.empty((P, M, K // GROUP), device=y.device, dtype=torch.float32))


@torch.library.custom_op("radiance::pq_ew_rot", mutates_args=())
def pq_ew_rot(mode: int, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float,
              rec: torch.Tensor, cs: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Producer + rotate + quant for the single-partition sites: mode 0 silu-mul (x = gate_up
    [M, 2N], y unused), mode 1 attention gate (x [M, N] * sigmoid(y)), mode 2 GDN gated rmsnorm
    (x [M, N], z = y, w [128]). Returns (hs, A, ASG, RS); above the decode band only hs is
    produced and the linear takes its tiled prefill path from it."""
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    N = x2.shape[1] // 2 if mode == 0 else x2.shape[1]
    G = N // GROUP
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
    a = torch.empty((1, M, N), device=x.device, dtype=torch.uint8)
    asg = torch.empty((1, M, G), device=x.device, dtype=torch.float32)
    rs = torch.empty((1, M, G), device=x.device, dtype=torch.float32)
    fused = M <= DECODE_MAX_M
    _ext.launch_ew_rot(mode, x2.data_ptr(), y2.data_ptr(), ys, w.data_ptr(), float(eps),
                       rec.data_ptr(), cs.data_ptr(), hs.data_ptr(), a.data_ptr(),
                       asg.data_ptr(), rs.data_ptr(), M, N, rec.shape[1], 1 if fused else 0,
                       torch.cuda.current_stream().cuda_stream)
    return hs, a, asg, rs


@pq_ew_rot.register_fake
def _(mode, x, y, w, eps, rec, cs):
    Kx = x.shape[-1]
    M = x.numel() // Kx
    N = Kx // 2 if mode == 0 else Kx
    return (torch.empty((M, N), device=x.device, dtype=torch.bfloat16),
            torch.empty((1, M, N), device=x.device, dtype=torch.uint8),
            torch.empty((1, M, N // GROUP), device=x.device, dtype=torch.float32),
            torch.empty((1, M, N // GROUP), device=x.device, dtype=torch.float32))


class _PqSiluMulRot(torch.nn.Module):
    """Drop-in for the MLP's SiluAndMul: (hs, A, ASG, RS) for the down_proj."""

    def __init__(self, down):
        super().__init__()
        self._down = [down]                       # not a submodule (no double registration)

    def forward(self, gu):
        d = self._down[0]
        hs, rest = d.quant_method.stream_ew(0, gu, gu, d.cs, 0.0, d)
        return (hs, *rest)


def _rot_gdn_output_projection(self, core_attn_out, z):
    """QwenGatedDeltaNetAttention._output_projection: gated norm + rotate + quant in one launch,
    tuple into out_proj."""
    T = core_attn_out.shape[0]
    x = core_attn_out.reshape(T, -1)
    zz = z.reshape(T, -1)
    op = self.out_proj
    hs, rest = op.quant_method.stream_ew(2, x, zz, self.norm.weight, float(self.norm.eps), op)
    output, _ = op((hs, *rest))
    return output


def _rot_attn_forward(self, positions, hidden_states):
    """Qwen3NextAttention.forward with the gate multiply + rotate + quant fused into o_proj's
    producer (mirrors the stock body)."""
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    attn_output = self.attn(q, k, v)
    if gate is not None:
        op = self.o_proj
        hs, rest = op.quant_method.stream_ew(1, attn_output, gate, op.cs, 0.0, op)
        output, _ = op((hs, *rest))
        return output
    output, _ = self.o_proj(attn_output)
    return output


def _gdn_norm_ok(la) -> str | None:
    n = getattr(la, "norm", None)
    if n is None or not _is_pq(getattr(la, "out_proj", None)):
        return "no norm / out_proj not paroquant"
    if getattr(n, "group_size", None) is not None or not getattr(n, "norm_before_gate", False):
        return "norm shape"
    if getattr(n, "activation", "swish") not in ("swish", "silu"):
        return f"activation {n.activation}"
    if n.weight.dtype != torch.bfloat16 or n.weight.numel() != 128:
        return "norm weight"
    if la.out_proj.input_size_per_partition % 128:
        return "N % 128"
    return None


# ---- rotation stream 3: the all-reduce fused into the norm+rotate producer ----------------------
# Contract (radiance_arnq's fp8 stream, adapted): a RowParallel linear whose consumer site is
# fused stops reducing (reduce_results=False) and hands its PARTIAL to pq_ar_add_rms_rot, which
# does the two-rank one-shot all-reduce + residual add + norm + rotate + quant in one launch in
# the decode band; above the band it all-reduces through vLLM and runs the plain norm kernel.
# Own IPC scratch/flags/counters (not the plain AR's -- slot parity must alternate per launch of
# THIS kernel family). The last layer keeps the stock contract (model.norm / drafter inputs).
ROT_STREAM3 = ROT_STREAM and os.environ.get("RADIANCE_PQ_ROT_STREAM3", "0") == "1"
AR_CHECK = int(os.environ.get("RADIANCE_PQ_AR_CHECK", "0"))
# Debug: keep the reduce-later contract but take the unfused path (vLLM AR + plain kernel)
# inside pq_ar_add_rms_rot -- separates a kernel/protocol fault from a contract fault.
AR_FALLBACK = os.environ.get("RADIANCE_PQ_AR_FALLBACK", "0") == "1"
_AR = {"ok": False}


def _ar_setup(device) -> bool:
    """Allocate this kernel family's IPC scratch/flags and exchange handles. Once per process."""
    if "tried" in _AR:
        return _AR["ok"]
    _AR["tried"] = True
    try:
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        comm = getattr(tp.device_communicator, "radiance_comm", None)
        if comm is None or comm.disabled or tp.world_size != 2:
            sys.stderr.write("[radiance.paroquant] rot stream3: no 2-rank radiance comm, off\n")
            return False
        ext = comm._ext
        slot_bytes = 128 * 5120 * 2 * 2           # M<=128 rows x K<=10240 bf16 per slot
        nflags = 1024                             # M*chunks (<= 128*5 at gpw 1)
        torch.cuda.set_device(device)
        sc, sc_h, _ = comm._alloc(2 * slot_bytes, True)
        fl, fl_h, _ = comm._alloc(nflags * 4, True)
        sc_handles = [None] * 2
        fl_handles = [None] * 2
        dist.all_gather_object(sc_handles, sc_h, group=comm.group)
        dist.all_gather_object(fl_handles, fl_h, group=comm.group)
        peer = 1 - comm.rank
        _AR.update(scratch=sc, flags=fl, peer_scratch=ext.ar_ipc_open(sc_handles[peer]),
                   peer_flags=ext.ar_ipc_open(fl_handles[peer]),
                   seq=torch.zeros(nflags, dtype=torch.int32, device=device),
                   slot_bytes=slot_bytes, nflags=nflags, drain=comm.drain, acq=comm.acq,
                   comm=comm, ok=True)
        sys.stderr.write(f"[radiance.paroquant] rot stream3: fused AR buffers ready (rank {comm.rank}, "
                         f"slot {slot_bytes >> 10} KiB, {nflags} flags, drain={comm.drain} acq={comm.acq})\n")
        return True
    except Exception as e:                                          # noqa: BLE001
        sys.stderr.write(f"[radiance.paroquant] rot stream3: setup failed, off: {e!r}\n")
        return False


@torch.library.custom_op("radiance::pq_ar_add_rms_rot", mutates_args=())
def pq_ar_add_rms_rot(y: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                      rec: torch.Tensor, cs: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """y is this rank's PARTIAL sum. Decode band: fused AR+add+norm+rotate+quant; above it:
    vLLM all-reduce then the plain norm kernel (same outputs as pq_add_rms_rot)."""
    y2 = y.reshape(-1, y.shape[-1])
    M, K = y2.shape
    P = rec.shape[0]
    G = K // GROUP
    res = residual.reshape(M, K)
    if not res.is_contiguous():
        res = res.contiguous()
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    hs = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    ro = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    a = torch.empty((P, M, K), device=y.device, dtype=torch.uint8)
    asg = torch.empty((P, M, G), device=y.device, dtype=torch.float32)
    rs = torch.empty((P, M, G), device=y.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    if M <= DECODE_MAX_M and _AR["ok"] and M * K * 2 <= _AR["slot_bytes"] and not AR_FALLBACK:
        _ext.launch_ar_add_rms_rot(y2.data_ptr(), _AR["peer_scratch"], _AR["scratch"],
                                   _AR["slot_bytes"], _AR["peer_flags"], _AR["flags"],
                                   _AR["seq"].data_ptr(), _AR["nflags"], res.data_ptr(),
                                   weight.data_ptr(), float(eps), rec.data_ptr(), cs.data_ptr(),
                                   hs.data_ptr(), ro.data_ptr(), a.data_ptr(), asg.data_ptr(),
                                   rs.data_ptr(), M, K, P, rec.shape[1], _AR["drain"], _AR["acq"],
                                   stream)
        if AR_CHECK and _AR.get("checks", 0) < AR_CHECK:
            # RADIANCE_PQ_AR_CHECK=N: for the first N decode-band calls also run the unfused
            # path (vLLM all-reduce + plain kernel) on the same inputs and report the divergence
            # per output. Eager only (the compare syncs) -- use with --enforce-eager.
            _AR["checks"] = _AR.get("checks", 0) + 1
            from vllm.distributed import tensor_model_parallel_all_reduce as _tpar
            yr = _tpar(y2).contiguous()
            hs2 = torch.empty_like(hs); ro2 = torch.empty_like(ro); a2 = torch.empty_like(a)
            asg2 = torch.empty_like(asg); rs2 = torch.empty_like(rs)
            _ext.launch_add_rms_rot(yr.data_ptr(), res.data_ptr(), weight.data_ptr(), float(eps),
                                    rec.data_ptr(), cs.data_ptr(), hs2.data_ptr(), ro2.data_ptr(),
                                    a2.data_ptr(), asg2.data_ptr(), rs2.data_ptr(), M, K, P,
                                    rec.shape[1], 1, stream)
            torch.cuda.synchronize()
            rows_bad = int((ro != ro2).view(M, -1).any(dim=1).sum())
            sys.stderr.write(f"[radiance.paroquant] AR_CHECK call {_AR['checks']} M={M} K={K} P={P}: "
                             f"ro diff elems {int((ro != ro2).sum())} (rows {rows_bad}/{M}), "
                             f"hs diff {int((hs != hs2).sum())}, codes diff {int((a != a2).sum())}, "
                             f"asg maxrel {float(((asg - asg2).abs() / asg2.abs().clamp_min(1e-12)).max()):.2e}\n")
        return hs.view(y.shape), ro.view(residual.shape), a, asg, rs
    from vllm.distributed import tensor_model_parallel_all_reduce as _tpar
    yr = _tpar(y2).contiguous()
    _ext.launch_add_rms_rot(yr.data_ptr(), res.data_ptr(), weight.data_ptr(), float(eps),
                            rec.data_ptr(), cs.data_ptr(), hs.data_ptr(), ro.data_ptr(),
                            a.data_ptr(), asg.data_ptr(), rs.data_ptr(), M, K, P, rec.shape[1],
                            1 if M <= DECODE_MAX_M else 0, stream)
    return hs.view(y.shape), ro.view(residual.shape), a, asg, rs


@pq_ar_add_rms_rot.register_fake
def _(y, residual, weight, eps, rec, cs):
    K = y.shape[-1]
    M = y.numel() // K
    P = rec.shape[0]
    return (torch.empty(y.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty(residual.shape, device=y.device, dtype=torch.bfloat16),
            torch.empty((P, M, K), device=y.device, dtype=torch.uint8),
            torch.empty((P, M, K // GROUP), device=y.device, dtype=torch.float32),
            torch.empty((P, M, K // GROUP), device=y.device, dtype=torch.float32))


# ---- rotation stream: patched forwards + installer --------------------------------------------
def _rot_layer_forward(self, hidden_states, residual, positions=None, **kwargs):
    """Decoder-layer forward under the rotation stream. Mirrors the stock body
    (qwen3_next.py Qwen3NextDecoderLayer.forward) minus the branches install_stream() proves dead
    (sequence parallel, layer_scale), exactly as radiance_arnq._stream_forward does."""
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hs = hidden_states
    elif self._pq_rot_in is not None:
        cons = self._pq_rot_in
        if self._pq_ar_in:
            hsb, residual, a, asg, rs = torch.ops.radiance.pq_ar_add_rms_rot(
                hidden_states, residual, self.input_layernorm.weight,
                float(self.input_layernorm.variance_epsilon), cons.rec, cons.cs)
            hs = (hsb, a, asg, rs)
        else:
            hsb, residual, rest = cons.quant_method.stream_norm(
                hidden_states, residual, self.input_layernorm.weight,
                float(self.input_layernorm.variance_epsilon), cons)
            hs = (hsb, *rest)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hs = hidden_states

    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn(hidden_states=hs)
    else:
        attn_out = self.self_attn(hidden_states=hs, positions=positions)

    if self._pq_rot_mid is not None:
        cons = self._pq_rot_mid
        if self._pq_ar_mid:
            hsb, residual, a, asg, rs = torch.ops.radiance.pq_ar_add_rms_rot(
                attn_out, residual, self.post_attention_layernorm.weight,
                float(self.post_attention_layernorm.variance_epsilon), cons.rec, cons.cs)
            hidden_states = self.mlp((hsb, a, asg, rs))
        else:
            hsb, residual, rest = cons.quant_method.stream_norm(
                attn_out, residual, self.post_attention_layernorm.weight,
                float(self.post_attention_layernorm.variance_epsilon), cons)
            hidden_states = self.mlp((hsb, *rest))
    else:
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


def _rot_gdn_forward_hip(self, hidden_states):
    """QwenGatedDeltaNetAttention.forward_hip (AITER-Triton branch) made tuple-aware: the
    quantized in_proj_qkvz takes the stream tuple, the fp16 in_proj_ba takes the bf16 hidden."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _g
    if isinstance(hidden_states, tuple):
        hsb, pre = hidden_states[0], hidden_states
    else:
        hsb, pre = hidden_states, hidden_states
    num_tokens = hsb.size(0)
    projected_states_qkvz, _ = self.in_proj_qkvz(pre)
    projected_states_ba, _ = self.in_proj_ba(hsb)
    projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
    projected_states_ba = projected_states_ba.view(num_tokens, -1)
    core_attn_out = torch.empty(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hsb.dtype, device=hsb.device)
    z = torch.empty(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=projected_states_qkvz.dtype, device=projected_states_qkvz.device)
    torch.ops.vllm.qwen_gdn_attention_core(
        projected_states_qkvz, projected_states_ba, z, core_attn_out,
        layer_name=_g._encode_layer_name(self.prefix), use_aiter=True)
    return self._output_projection(core_attn_out, z)


def _is_pq(lin) -> bool:
    """A linear whose quant method can consume a rotation-stream tuple (int4 per-group or
    paroquant_mxfp4 per-token): it exposes stream_norm/stream_ew that build ITS tuple."""
    return (lin is not None and getattr(lin, "rec", None) is not None
            and getattr(getattr(lin, "quant_method", None), "pq_stream_capable", False))


def _is_pq_ar(lin) -> bool:
    """Stream 3 (fused all-reduce) only exists for the int4 per-group tuple."""
    return _is_pq(lin) and getattr(lin.quant_method, "pq_ar_capable", False)


def install_stream(model) -> None:
    """Patch the decoder layers for the rotation stream. Best-effort and loud: any guard that
    fails leaves that layer stock."""
    if not ROT_STREAM:
        return
    import types
    core = None
    for m in model.modules():
        if hasattr(m, "layers") and hasattr(m, "aux_hidden_state_layers"):
            core = m
            break
    if core is None:
        sys.stderr.write("[radiance.paroquant] rot stream: no decoder core found, skipping\n")
        return
    if tuple(getattr(core, "aux_hidden_state_layers", ()) or ()):
        sys.stderr.write("[radiance.paroquant] rot stream: aux hidden taps set, skipping\n")
        return
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _g
        gdn_ok = bool(getattr(_g, "GDN_AITER_TRITON_AVAILABLE", False))
    except Exception as e:                                          # noqa: BLE001
        sys.stderr.write(f"[radiance.paroquant] rot stream: GDN module probe failed {e!r}\n")
        gdn_ok = False
    n_in = n_mid = n_gdn = n_act = n_gnq = n_attn = n_ar = 0
    ar_ok = ROT_STREAM3 and _ar_setup(next(model.parameters()).device)
    L = len(core.layers)
    # Flags are initialised in a pre-pass: layer i's iteration grants layer i+1's `_pq_ar_in`
    # (its down_proj goes partial), so the per-layer init must not run after that grant --
    # it did once, and every layer but the last normalised an un-reduced partial.
    for layer in core.layers:
        layer._pq_rot_in = None
        layer._pq_rot_mid = None
        layer._pq_ar_in = False
        layer._pq_ar_mid = False
    for i, layer in enumerate(core.layers):
        if getattr(layer, "layer_scale", False) or getattr(layer, "use_attn_reduce_scatter_for_moe", False):
            sys.stderr.write(f"[radiance.paroquant] rot stream: layer {i} layer_scale/SP, stock\n")
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "expert_gate", None) is not None:
            sys.stderr.write(f"[radiance.paroquant] rot stream: layer {i} mlp shape, stock\n")
            continue
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            cons_in = getattr(la, "in_proj_qkvz", None)
            if (gdn_ok and _is_pq(cons_in) and getattr(la, "in_proj_ba", None) is not None
                    and getattr(la, "_forward_method", None) == getattr(la, "forward_hip", None)):
                la.forward_hip = types.MethodType(_rot_gdn_forward_hip, la)
                la._forward_method = la.forward_hip
                layer._pq_rot_in = cons_in
                n_gdn += 1
        else:
            cons_in = getattr(layer.self_attn, "qkv_proj", None)
            if _is_pq(cons_in):
                layer._pq_rot_in = cons_in
        if layer._pq_rot_in is not None:
            n_in += 1
        cons_mid = getattr(mlp, "gate_up_proj", None)
        if _is_pq(cons_mid):
            layer._pq_rot_mid = cons_mid
            n_mid += 1
        layer.forward = types.MethodType(_rot_layer_forward, layer)
        if ar_ok:
            # mid: this layer's o_proj/out_proj stays partial, the mid epilogue reduces it
            row = (layer.linear_attn.out_proj if layer.layer_type == "linear_attention"
                   else layer.self_attn.o_proj)
            if layer._pq_rot_mid is not None and _is_pq_ar(row) and row.bias is None \
                    and getattr(row, "reduce_results", False):
                row.reduce_results = False
                layer._pq_ar_mid = True
                n_ar += 1
            # down stream: this layer's down_proj stays partial, the NEXT layer's input epilogue
            # reduces it. Never on the last layer (model.norm and the drafter read its output).
            nxt = core.layers[i + 1] if i + 1 < L else None
            down = getattr(mlp, "down_proj", None)
            if nxt is not None and _is_pq_ar(down) and down.bias is None \
                    and getattr(down, "reduce_results", False) \
                    and not getattr(nxt, "layer_scale", False) \
                    and not getattr(nxt, "use_attn_reduce_scatter_for_moe", False) \
                    and getattr(nxt.mlp, "expert_gate", None) is None:
                nin = (getattr(nxt.linear_attn, "in_proj_qkvz", None)
                       if nxt.layer_type == "linear_attention"
                       else getattr(nxt.self_attn, "qkv_proj", None))
                gdn_next_ok = (nxt.layer_type != "linear_attention"
                               or (gdn_ok and getattr(nxt.linear_attn, "in_proj_ba", None) is not None))
                if _is_pq_ar(nin) and gdn_next_ok:
                    down.reduce_results = False
                    nxt._pq_ar_in = True
                    n_ar += 1
        if ROT_STREAM2:
            down = getattr(mlp, "down_proj", None)
            if _is_pq(down) and down.input_size_per_partition % 128 == 0 \
                    and getattr(mlp, "act_fn", None) is not None:
                mlp.act_fn = _PqSiluMulRot(down)
                n_act += 1
            if layer.layer_type == "linear_attention":
                why = _gdn_norm_ok(layer.linear_attn)
                if why is None:
                    layer.linear_attn._output_projection = types.MethodType(
                        _rot_gdn_output_projection, layer.linear_attn)
                    n_gnq += 1
                else:
                    sys.stderr.write(f"[radiance.paroquant] rot stream2: layer {i} gdn norm stock ({why})\n")
            else:
                sa = layer.self_attn
                op = getattr(sa, "o_proj", None)
                if _is_pq(op) and op.input_size_per_partition % 128 == 0 and hasattr(sa, "_project_qkv_gate"):
                    sa.forward = types.MethodType(_rot_attn_forward, sa)
                    n_attn += 1
    for layer in core.layers:
        if getattr(layer, "_pq_ar_in", False) and layer._pq_rot_in is None:
            # the down stream was granted but the consumer did not install: undo it
            layer._pq_ar_in = False
            sys.stderr.write("[radiance.paroquant] rot stream3: a down stream had no consumer, reverted\n")
    sys.stderr.write(f"[radiance.paroquant] rot stream installed: {n_in} input epilogues "
                     f"({n_gdn} GDN), {n_mid} mid epilogues over {len(core.layers)} layers"
                     f"{f'; stream2: {n_act} silu-mul, {n_gnq} gdn-norm, {n_attn} attn-gate' if ROT_STREAM2 else ''}"
                     f"{f'; stream3: {n_ar} fused all-reduces' if ar_ok else ''}\n")


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
        if WPERM:
            if N % 16 or K % 16:
                raise ValueError(f"RADIANCE_PQ_WPERM needs N and K divisible by 16, got "
                                 f"N={N} K={K}; unset it for this checkpoint")
            qweight = permute_w(qweight, N, K)

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

    # rotation stream: this method's producers build the per-GROUP tuple its GEMM consumes
    pq_stream_capable = True
    pq_ar_capable = True

    @staticmethod
    def stream_norm(y, residual, weight, eps, cons):
        hs, ro, a, asg, rs = torch.ops.radiance.pq_add_rms_rot(y, residual, weight, eps, cons.rec, cons.cs)
        return hs, ro, (a, asg, rs)

    @staticmethod
    def stream_ew(mode, x, y, w, eps, cons):
        hs, a, asg, rs = torch.ops.radiance.pq_ew_rot(mode, x, y, w, eps, cons.rec, cons.cs)
        return hs, (a, asg, rs)

    def apply(self, layer, x, bias: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(x, tuple):          # rotation stream: (hs, A, ASG, RS)
            hs, a, asg, rs = x
            out = torch.ops.radiance.paroquant_linear_pre(hs, a, asg, rs, layer.qweight,
                                                          layer.sz, layer.rec, layer.cs,
                                                          layer.pq_pb1, layer.pq_pb2)
        else:
            out = torch.ops.radiance.paroquant_linear(x, layer.qweight, layer.sz, layer.rec,
                                                      layer.cs, layer.pq_pb1, layer.pq_pb2)
        if bias is not None:
            out = out + bias
        return out


if os.environ.get("RADIANCE_PAROQUANT", "0") == "1":
    sys.stderr.write("[radiance.paroquant] registered (int4 g128 asym + rotations, W4A8, "
                     f"gfx1201; weight layout {'FRAGMENT ORDER' if WPERM else 'row'}, "
                     f"prefill {'A-tiled' if ATILED_ENABLED else 'row'}, decode band M<="
                     f"{DECODE_MAX_M}, rot stream {'on' if ROT_STREAM else 'off'})\n")
