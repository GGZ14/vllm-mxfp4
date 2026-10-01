"""fp8 per-channel copies of the MoE lane's bf16 dense layers + a fused shared-expert gate (moe-densefp8/, MOE-GFX1201.md).

Two knobs, both off when unset (serve-moe-mxfp4.sh turns both on; read once at import):

  RADIANCE_MOE_DENSE_FP8=1   The TARGET model's bf16 linears that Quark excluded from MXFP4 and that dominate the
                             decode bytes are quantized at load to OCP e4m3fn with one fp32 scale per output row (max ->
                             448) and the bf16 tensors are freed, BEFORE vLLM sizes the KV cache (so the pool grows):
                               linear_attn.in_proj_qkvz [12288x2048] x30, linear_attn.out_proj [2048x4096] x30,
                               self_attn.qkv_proj [9216x2048] x10, self_attn.o_proj [2048x4096] x10,
                               mlp.shared_expert.gate_up_proj [1024x2048] x40, mlp.shared_expert.down_proj [2048x512] x40
                             (2.83 GB bf16 -> 1.42 GB). Kept exactly bf16, on purpose:
                               linear_attn.in_proj_ba [64x2048]: b and a feed the GDN beta (sigmoid) and the decay
                                 (exp(-exp(A_log) * softplus(a + dt_bias))), where a logit error compounds through the
                                 recurrence across every later token; 4 MB in all and latency-bound (3.9 us), no gain;
                               mlp.gate (router) [256x2048]: top-8-of-256 expert choice flips on small logit changes,
                                 vLLM itself builds it with quant_config=None; 1 MB, latency-bound;
                               mlp.shared_expert_gate [1x2048]: scalar gate (see the gate fix instead);
                               lm_head / embed_tokens (verify head stays exact; the draft head has its own int2 copy),
                               conv1d, norms, and the whole MTP drafter (its linears only cost acceptance, not output,
                                 and stay bf16 so the drafter sees the hidden states it was trained on as closely as
                                 the target allows).
                             Kernel per call (measured on gfx1201, in-graph microbenchmark, MOE-GFX1201.md):
                               M <= 16      the measured best per (N, K, M) of the HIP skinny W8A16 kernel
                                            (radiance_fp8w, fp8 -> bf16 by v_cvt_f32_fp8, v_dot2_f32_bf16) and the Triton
                                            W8A16 kernel (native float8e4nv -> bf16 cast, bf16 WMMA);
                               17..80       Triton W8A16, per-bucket tiles;
                               > 80         torch._scaled_mm rowwise W8A8: per-token dynamic fp8 activations (vLLM's
                                            scaled_fp8_quant) x per-channel weights. Prefill chunks only. This is the one
                                            place activations are quantized -- at M = 4096 it is 1.3-1.4x faster than the
                                            bf16 hipBLASLt GEMM it replaces, where every weight-only path is 15-50%
                                            slower; decode (M <= 80) stays weight-only. Same split the W4A8 experts use
                                            (fp8 activations only for calls of 1025+ tokens).
  RADIANCE_MOE_GATE_FIX=1    Qwen2MoeMLP's expert gate (mlp.shared_expert_gate, [1x2048], target AND drafter):
                             sigmoid(linear(x)) * out was three kernels (a hipBLASLt GEMV that alone takes ~20 us, sigmoid,
                             mul); one Triton kernel computes it in ~3 us and emulates the bf16 roundings of the three.
                             Only the dot product's summation order differs from hipBLASLt: identical on the bench's
                             real-weight checks, and in the serve 15/20 fixed greedy prompts stay identical over 256
                             tokens, the other 5 flip at exact near-ties (top-2 logprob margin <= 0.125).

Both are applied from vLLM's model_loader process_weights_after_loading (patch_moe_densefp8.py): convert(model) runs
for every loaded model (target, then the MTP drafter). Asked for and failing is fatal at load, never a silent fallback.

  w8a16_triton(x, w8, s, cfg)    Triton weight-only GEMM: W8 = OCP e4m3fn bytes [N, K], one fp32 scale per output
                                 row, x bf16, fp32 accumulate, bf16 out. DEC selects the fp8 -> 16-bit decode:
                                   0 = bit trick into fp16 (x cast to fp16: overflow risk above 65504),
                                   1 = bit trick into fp16 then exact cast to bf16 (bf16 dot),
                                   2 = Triton's native float8e4nv -> bf16 cast (bf16 dot).
                                 The bit trick places eeee.mmm into the fp16 exponent/mantissa: value * 2^-8 for
                                 normals and subnormals alike, so DEC 0/1 take the scale pre-multiplied by 256.
  hip_fp8w(x, w8, s, wv, sk)     the HIP skinny kernel (radiance_fp8w.hip, M <= 16), built at container start.
  gate_mul(x, wg, out)           out * sigmoid(x @ wg^T) in ONE kernel for Qwen2MoeMLP's expert_gate (shared_expert_gate,
                                 [1, 2048]); emulates the bf16 roundings of linear -> sigmoid -> mul.
"""
import os
import sys

import torch
import triton
import triton.language as tl


def _log(msg):
    sys.stderr.write(f"[radiance.densefp8] {msg}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------ Triton W8A16
@triton.jit
def _w8a16_kernel(X, W, S, Y, M, N, K, stride_xm, stride_ym,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DEC: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    xp = X + offs_m[:, None] * stride_xm + offs_k[None, :]
    wp = W + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k0 in range(0, K, BLOCK_K):
        x = tl.load(xp, mask=mask_m[:, None], other=0.0)
        b = tl.load(wp, mask=mask_n[None, :], other=0)
        if DEC == 0:
            b16 = b.to(tl.uint16)
            w = (((b16 & 0x80) << 8) | ((b16 & 0x7F) << 7)).to(tl.float16, bitcast=True)
            acc += tl.dot(x.to(tl.float16), w)
        elif DEC == 1:
            b16 = b.to(tl.uint16)
            w = (((b16 & 0x80) << 8) | ((b16 & 0x7F) << 7)).to(tl.float16, bitcast=True).to(tl.bfloat16)
            acc += tl.dot(x, w)
        else:
            w = b.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            acc += tl.dot(x, w)
        xp += BLOCK_K
        wp += BLOCK_K
    s = tl.load(S + offs_n, mask=mask_n, other=0.0)
    y = acc * s[None, :]
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


def w8a16_triton(x, w8, s, cfg, out=None):
    """x [M, K] bf16 (row stride may exceed K), w8 uint8 [N, K], s fp32 [N] (x256 for DEC 0/1)."""
    m, k = x.shape
    n = w8.shape[0]
    bm = cfg["BLOCK_M"]
    y = out if out is not None else torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, cfg["BLOCK_N"]), triton.cdiv(m, bm))
    _w8a16_kernel[grid](x, w8, s, y, m, n, k, x.stride(0), y.stride(0), BLOCK_M=bm, BLOCK_N=cfg["BLOCK_N"],
                        BLOCK_K=cfg["BLOCK_K"], DEC=cfg.get("DEC", 1), num_warps=cfg["num_warps"],
                        num_stages=cfg.get("num_stages", 2))
    return y


# ------------------------------------------------------------------------------------------------ Triton W8A8
@triton.jit
def _w8a8_kernel(XQ, XS, W, S, Y, M, N, K, stride_xm, stride_ym, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """y = (xq @ w8^T) * xs[m] * s[n]: fp8 x fp8 WMMA, per-token activation scale, per-channel weight scale."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_pid_n
    group_id = pid // width
    first_m = group_id * GROUP_M
    gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % gsz
    pid_n = (pid % width) // gsz
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    ap = XQ + offs_m[:, None] * stride_xm + offs_k[None, :]
    bp = W + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k0 in range(0, K, BLOCK_K):
        a = tl.load(ap, mask=mask_m[:, None], other=0.0)
        b = tl.load(bp, mask=mask_n[None, :], other=0).to(tl.float8e4nv, bitcast=True)
        acc += tl.dot(a, b)
        ap += BLOCK_K
        bp += BLOCK_K
    xs = tl.load(XS + offs_m, mask=mask_m, other=0.0)
    s = tl.load(S + offs_n, mask=mask_n, other=0.0)
    y = acc * xs[:, None] * s[None, :]
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


def w8a8_triton(xq, xs, w8, s, cfg, out=None):
    """xq fp8 e4m3fn [M, K] (per-token quantized), xs fp32 [M], w8 uint8 [N, K], s fp32 [N]."""
    m, k = xq.shape
    n = w8.shape[0]
    y = out if out is not None else torch.empty(m, n, dtype=torch.bfloat16, device=xq.device)
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _w8a8_kernel[grid](xq, xs, w8, s, y, m, n, k, xq.stride(0), y.stride(0), BLOCK_M=cfg["BLOCK_M"],
                       BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], GROUP_M=cfg.get("GROUP_M", 8),
                       num_warps=cfg["num_warps"], num_stages=cfg.get("num_stages", 2))
    return y


# ------------------------------------------------------------------------------------------------ HIP W8A16
_HIP = None


def hip_module():
    global _HIP
    if _HIP is None:
        import radiance_fp8w  # built by build_radiance_fp8w.sh at container start
        _HIP = radiance_fp8w
    return _HIP


def hip_fp8w(x, w8, s, wv, sk, out=None):
    m, k = x.shape
    n = w8.shape[0]
    y = out if out is not None else torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    hip_module().gemm(x.data_ptr(), w8.data_ptr(), s.data_ptr(), y.data_ptr(), m, k, n, wv, sk, x.stride(0),
                      torch.cuda.current_stream().cuda_stream)
    return y


# ------------------------------------------------------------------------------------------------ fused expert gate
@triton.jit
def _gate_mul_kernel(X, G, O, Y, K, H, stride_x, stride_o, stride_y, BLOCK_K: tl.constexpr,
                     BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = (k0 + offs) < K
        xv = tl.load(X + row * stride_x + k0 + offs, mask=km, other=0.0).to(tl.float32)
        gv = tl.load(G + k0 + offs, mask=km, other=0.0).to(tl.float32)
        acc += xv * gv
    logit = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)          # linear() returns bf16
    gate = tl.sigmoid(logit).to(tl.bfloat16).to(tl.float32)            # F.sigmoid on bf16 returns bf16
    offh = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        hm = (h0 + offh) < H
        o = tl.load(O + row * stride_o + h0 + offh, mask=hm, other=0.0).to(tl.float32)
        tl.store(Y + row * stride_y + h0 + offh, (o * gate).to(tl.bfloat16), mask=hm)


def gate_mul(x, wg, out):
    """out * sigmoid(x @ wg^T): x [T, K] bf16, wg [1, K] bf16, out [T, H] bf16 -> new [T, H] bf16."""
    t, k = x.shape
    h = out.shape[-1]
    y = torch.empty_like(out)
    if t == 0:
        return y
    bk = min(4096, triton.next_power_of_2(k))
    bh = min(4096, triton.next_power_of_2(h))
    _gate_mul_kernel[(t,)](x, wg, out, y, k, h, x.stride(0), out.stride(0), y.stride(0), BLOCK_K=bk, BLOCK_H=bh,
                           num_warps=4)
    return y


# ------------------------------------------------------------------------------------------------ quantizer
def quant_fp8_rows(w, q=None, s=None):
    """[N, K] bf16 -> (e4m3fn bytes uint8 [N, K], fp32 scale [N]) with the row max mapped to 448 (into q / s if given)."""
    n, k = w.shape
    q = torch.empty(n, k, dtype=torch.uint8, device=w.device) if q is None else q
    s = torch.empty(n, dtype=torch.float32, device=w.device) if s is None else s
    for i in range(0, n, 4096):
        wf = w[i:i + 4096].float()
        sc = (wf.abs().amax(dim=1) / 448.0).clamp(min=1e-30)
        q[i:i + 4096] = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        s[i:i + 4096] = sc
        del wf, sc
    return q, s


def dequant_fp8_rows(q, s):
    return q.view(torch.float8_e4m3fn).float() * s[:, None]


# ================================================================================================ serve integration
DENSE = os.environ.get("RADIANCE_MOE_DENSE_FP8", "0").strip() == "1"
GATE_FIX = os.environ.get("RADIANCE_MOE_GATE_FIX", "0").strip() == "1"

import re  # noqa: E402

_CONVERT_RE = re.compile(r"\.layers\.\d+\.(linear_attn\.(in_proj_qkvz|out_proj)|self_attn\.(qkv_proj|o_proj)|"
                         r"mlp\.shared_expert\.(gate_up_proj|down_proj))$")
_KEEP_RE = re.compile(r"\.layers\.\d+\.(linear_attn\.in_proj_ba|mlp\.gate|mlp\.shared_expert_gate)$")

# Measured dispatch (in-graph microbenchmark, gfx1201, per-call times). Keys: (N, K).
#   small: M -> ("hip", WV, SK) | ("tri", BLOCK_M, BLOCK_N, BLOCK_K, num_warps) for M = 1..16 (unmeasured M take the
#          nearest measured M above); tri: BLOCK_M bucket -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) for M = 17..80.
_PLAN = {
    (12288, 2048): {
        'small': {1: ('tri', 16, 64, 512, 4), 2: ('hip', 2, 2), 3: ('hip', 1, 2), 4: ('hip', 1, 2), 5: ('hip', 1, 2),
                  8: ('hip', 1, 1), 10: ('tri', 16, 64, 512, 4), 16: ('tri', 16, 64, 512, 4)},
        'tri': {32: (32, 32, 256, 4), 64: (64, 32, 128, 4), 128: (128, 64, 128, 4)},
    },
    (2048, 4096): {
        'small': {1: ('hip', 8, 4), 2: ('hip', 1, 8), 3: ('hip', 1, 4), 4: ('hip', 1, 4), 5: ('hip', 1, 4),
                  8: ('hip', 1, 4), 10: ('hip', 1, 2), 16: ('tri', 16, 16, 512, 2)},
        'tri': {32: (32, 32, 256, 4), 64: (64, 32, 256, 4), 128: (128, 32, 128, 8)},
    },
    (9216, 2048): {
        'small': {1: ('tri', 16, 64, 256, 4), 2: ('hip', 1, 2), 3: ('hip', 1, 2), 4: ('hip', 1, 2), 5: ('hip', 1, 2),
                  8: ('hip', 1, 1), 10: ('tri', 16, 64, 256, 4), 16: ('tri', 16, 64, 256, 4)},
        'tri': {32: (32, 32, 256, 4), 64: (64, 64, 128, 4), 128: (128, 64, 128, 4)},
    },
    (1024, 2048): {
        'small': {1: ('hip', 16, 2), 2: ('hip', 1, 1), 3: ('hip', 8, 1), 4: ('hip', 4, 1), 5: ('hip', 2, 1),
                  8: ('hip', 4, 1), 10: ('hip', 2, 1), 16: ('hip', 4, 2)},
        'tri': {32: (32, 32, 256, 4), 64: (64, 16, 256, 4), 128: (128, 16, 128, 4)},
    },
    (2048, 512): {
        'small': {1: ('tri', 16, 64, 512, 4), 2: ('tri', 16, 64, 512, 4), 3: ('tri', 16, 64, 512, 4),
                  4: ('tri', 16, 64, 512, 4), 5: ('tri', 16, 64, 512, 4), 8: ('tri', 16, 64, 512, 4),
                  10: ('tri', 16, 64, 512, 4), 16: ('tri', 16, 64, 512, 4)},
        'tri': {32: (32, 16, 256, 2), 64: (64, 32, 256, 4), 128: (128, 64, 128, 4)},
    },
}
_SMALL_MS = (1, 2, 3, 4, 5, 8, 10, 16)
W8A16_MAX_M = 80
_route_cache = {}
_ops = None


def _route(n, k, m):
    key = (n, k, m)
    r = _route_cache.get(key)
    if r is None:
        p = _PLAN[(n, k)]
        if m <= 16:
            mm = next(x for x in _SMALL_MS if m <= x)
            e = p['small'][mm]
            r = ("hip", (e[1], e[2])) if e[0] == "hip" else ("tri", dict(BLOCK_M=e[1], BLOCK_N=e[2], BLOCK_K=e[3],
                                                                      num_warps=e[4], num_stages=2, DEC=2))
        elif m <= W8A16_MAX_M:
            bm = 32 if m <= 32 else 64 if m <= 64 else 128
            e = p['tri'][bm]
            r = ("tri", dict(BLOCK_M=e[0], BLOCK_N=e[1], BLOCK_K=e[2], num_warps=e[3], num_stages=2, DEC=2))
        else:
            r = ("w8a8", None)
        _route_cache[key] = r
    return r


def _fp8_linear_impl(x: torch.Tensor, w8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """y = x @ (w8 * s[:, None])^T. x [..., K] bf16, w8 uint8 e4m3fn [N, K], s fp32 [N]."""
    global _ops
    n, k = w8.shape
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if m == 0:
        return x.new_empty((*x.shape[:-1], n))
    if x2.stride(-1) != 1 or x2.stride(0) % 8:
        x2 = x2.contiguous()
    kind, prm = _route(n, k, m)
    if kind == "hip":
        y = hip_fp8w(x2, w8, s, prm[0], prm[1])
    elif kind == "tri":
        y = w8a16_triton(x2, w8, s, prm)
    else:
        if _ops is None:
            import vllm._custom_ops as ops
            _ops = ops
        xq, xs = _ops.scaled_fp8_quant(x2.contiguous(), None, use_per_token_if_dynamic=True)
        y = torch._scaled_mm(xq, w8.view(torch.float8_e4m3fn).t(), scale_a=xs, scale_b=s.view(1, -1),
                             out_dtype=torch.bfloat16)
    return y.reshape(*x.shape[:-1], n)


def _fp8_linear_fake(x: torch.Tensor, w8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w8.shape[0]))


def _gate_mul_impl(x: torch.Tensor, wg: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return gate_mul(x, wg, out)


def _gate_mul_fake(x: torch.Tensor, wg: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


_REGISTERED = False


def register_ops():
    """torch.ops.vllm.radiance_fp8w_linear / radiance_gate_mul (opaque to dynamo, like rocm_unquantized_gemm)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    if not hasattr(torch.ops.vllm, "radiance_fp8w_linear"):
        direct_register_custom_op(op_name="radiance_fp8w_linear", op_func=_fp8_linear_impl,
                                  fake_impl=_fp8_linear_fake)
    if not hasattr(torch.ops.vllm, "radiance_gate_mul"):
        direct_register_custom_op(op_name="radiance_gate_mul", op_func=_gate_mul_impl, fake_impl=_gate_mul_fake)
    _REGISTERED = True


if DENSE or GATE_FIX:
    register_ops()


def _method_cls():
    from vllm.model_executor.layers.linear import LinearMethodBase

    class Fp8PerChannelLinearMethod(LinearMethodBase):
        """W8A16 (decode) / W8A8 (prefill) apply for a converted layer: layer.weight uint8 e4m3fn [N, K],
        layer.weight_scale fp32 [N]. Weights are created bf16 by UnquantizedLinearMethod and converted after load."""

        def create_weights(self, *args, **kwargs):
            raise RuntimeError("[radiance.densefp8] Fp8PerChannelLinearMethod is assigned after loading only")

        def process_weights_after_loading(self, layer):
            return

        def apply(self, layer, x, bias=None):
            y = torch.ops.vllm.radiance_fp8w_linear(x, layer.weight, layer.weight_scale)
            if bias is not None:
                y = y + bias
            return y

    return Fp8PerChannelLinearMethod


def _is_drafter(model):
    name = type(model).__name__
    return "MTP" in name or "Eagle" in name or "DFlash" in name or "Draft" in name


def _self_check_dense(samples):
    """Every distinct (N, K) at every dispatch band against x @ dequant(w8, s)^T; also compiles every Triton
    variant before vLLM's warm-up / graph capture."""
    worst = {}
    for (n, k), mod in samples.items():
        w8, s = mod.weight, mod.weight_scale
        ref_w = dequant_fp8_rows(w8, s)
        for m in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 20, 33, 40, 50, 65, 80, 81, 512):
            x = torch.randn(m, k, device=w8.device, dtype=torch.bfloat16)
            y = torch.ops.vllm.radiance_fp8w_linear(x, w8, s).float()
            ref = x.float() @ ref_w.t()
            rel = float((y - ref).norm() / ref.norm().clamp(min=1e-30))
            kind = _route(n, k, m)[0]
            tol = 6e-2 if kind == "w8a8" else 1e-2          # W8A8 adds the per-token activation quant error
            if not rel < tol:
                raise RuntimeError(f"[radiance.densefp8] self-check failed: ({n}, {k}) M={m} {kind}: rel. error "
                                   f"{rel:.4f} >= {tol}")
            worst[kind] = max(worst.get(kind, 0.0), rel)
        del ref_w
    return worst


def _convert_dense(model):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    method = _method_cls()()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    alloc0, resv0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    todo, kept = [], {}
    for pname, mod in model.named_modules():                    # pass 1: what converts, and checks
        if not isinstance(mod, LinearBase):
            continue
        prefix = getattr(mod, "prefix", "") or pname
        if not isinstance(getattr(mod, "quant_method", None), UnquantizedLinearMethod):
            continue
        if not _CONVERT_RE.search(prefix):
            km = _KEEP_RE.search(prefix)
            if km:
                kept[km.group(1)] = kept.get(km.group(1), 0) + 1
            continue
        w = mod.weight
        if w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"[radiance.densefp8] {prefix}: weight {tuple(w.shape)} {w.dtype}, expected 2-D bf16")
        if getattr(mod, "bias", None) is not None:
            raise RuntimeError(f"[radiance.densefp8] {prefix}: has a bias, not expected on this model")
        if getattr(mod, "tp_size", 1) != 1:
            raise RuntimeError("[radiance.densefp8] only TP=1 is supported")
        if tuple(w.shape) not in _PLAN:
            raise RuntimeError(f"[radiance.densefp8] {prefix}: no measured dispatch for {tuple(w.shape)}")
        todo.append((prefix, mod))
    if not todo:
        raise RuntimeError("[radiance.densefp8] RADIANCE_MOE_DENSE_FP8=1 but no layer matched: wrong model?")
    # One arena for every fp8 weight (and one for the scales), allocated while the bf16 weights are still alive: the
    # fp8 copies then never land in the holes the bf16 tensors leave, so those segments empty out completely and
    # empty_cache() can hand them back to the device before vLLM sizes the KV cache.
    al = 256
    offs, tot, rows = [], 0, 0
    for _, mod in todo:
        n, k = mod.weight.shape
        offs.append((tot, rows))
        tot += (n * k + al - 1) // al * al
        rows += n
    arena = torch.empty(tot, dtype=torch.uint8, device=todo[0][1].weight.device)
    sarena = torch.empty(rows, dtype=torch.float32, device=arena.device)
    nconv, b_bf16, wrel_max, counts, samples = 0, 0, 0.0, {}, {}
    for (prefix, mod), (o, r) in zip(todo, offs):                 # pass 2: quantize into the arena, drop bf16
        wd = mod.weight.data
        n, k = wd.shape
        if float(wd[:: max(1, n // 64)].abs().amax()) == 0.0:
            raise RuntimeError(f"[radiance.densefp8] {prefix}: weight is all-zero at conversion (not loaded?)")
        q, s = quant_fp8_rows(wd, arena[o:o + n * k].view(n, k), sarena[r:r + n])
        sel = torch.arange(0, n, max(1, n // 256), device=wd.device)
        wrel = float((dequant_fp8_rows(q[sel], s[sel]) - wd[sel].float()).norm() / wd[sel].float().norm())
        if not wrel < 0.05:
            raise RuntimeError(f"[radiance.densefp8] {prefix}: fp8 sample rel. error {wrel:.4f}")
        wrel_max = max(wrel_max, wrel)
        b_bf16 += wd.numel() * wd.element_size()
        del wd
        mod.weight = torch.nn.Parameter(q, requires_grad=False)          # drops the bf16 Parameter
        mod.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        mod.quant_method = method
        key = prefix.rsplit(".", 2)[-2] + "." + prefix.rsplit(".", 1)[-1]
        counts[key] = counts.get(key, 0) + 1
        samples.setdefault((n, k), mod)
        nconv += 1
    del todo
    hip_module()                                                       # fail loudly if the kernel is not built
    worst = _self_check_dense(samples)
    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    freed = torch.cuda.mem_get_info()[0] - free0
    alloc1, resv1 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    b_fp8 = arena.numel() + sarena.numel() * 4
    _log(f"MoE dense fp8 ON: {nconv} linears -> fp8 e4m3 per-channel ({b_bf16 / 2**30:.2f} GiB bf16 -> "
         f"{b_fp8 / 2**30:.2f} GiB; torch allocated {(alloc1 - alloc0) / 2**30:+.2f} GiB, reserved "
         f"{(resv1 - resv0) / 2**30:+.2f} GiB, device free memory {freed / 2**30:+.2f} GiB): "
         + ", ".join(f"{a} x{b}" for a, b in sorted(counts.items()))
         + "; kept bf16: " + ", ".join(f"{a} x{b}" for a, b in sorted(kept.items()))
         + f"; max weight sample rel. error {wrel_max:.4f}; self-check worst rel. error "
         + ", ".join(f"{a} {b:.4f}" for a, b in sorted(worst.items())))


def _check_gate(model):
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    n, w = 0, None
    for _, mod in model.named_modules():
        if isinstance(mod, Qwen2MoeMLP) and getattr(mod, "expert_gate", None) is not None:
            n += 1
            w = w if w is not None else mod.expert_gate.weight
    if n == 0:
        return 0, None
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] != 1:
        raise RuntimeError(f"[radiance.densefp8] gate fix: expert_gate weight {tuple(w.shape)} {w.dtype}")
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm  # noqa: F401
    worst = 0.0
    for m in (1, 5, 65):
        x = torch.randn(m, w.shape[1], device=w.device, dtype=torch.bfloat16)
        o = torch.randn(m, 2048, device=w.device, dtype=torch.bfloat16)
        ref = torch.sigmoid(torch.nn.functional.linear(x, w)) * o
        y = torch.ops.vllm.radiance_gate_mul(x, w, o)
        d = float((y.float() - ref.float()).abs().max() / ref.float().abs().max().clamp(min=1e-30))
        if not d < 1e-2:
            raise RuntimeError(f"[radiance.densefp8] gate fix self-check failed at M={m}: max rel. diff {d:.4f}")
        worst = max(worst, d)
    return n, worst


def convert(model):
    """Called at the end of vLLM's process_weights_after_loading for every loaded model (target, then drafter)."""
    drafter = _is_drafter(model)
    if DENSE and not drafter:
        _convert_dense(model)
    if GATE_FIX:
        n, worst = _check_gate(model)
        _log(f"MoE gate fix ON ({type(model).__name__}): {n} Qwen2MoeMLP expert gates -> fused sigmoid(x.w)*out "
             f"kernel; self-check max rel. diff {worst if worst is not None else float('nan'):.2e}")


def gate_ok(x, wg, out):
    """Guard used by the patched Qwen2MoeMLP.forward: anything unusual takes the original three-op path."""
    return (x.dim() == 2 and out.dim() == 2 and x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
            and wg.dtype == torch.bfloat16 and wg.dim() == 2 and wg.shape[0] == 1 and wg.shape[1] == x.shape[1]
            and x.stride(-1) == 1 and out.stride(-1) == 1 and wg.is_contiguous() and x.shape[0] == out.shape[0])
