#!/usr/bin/env python3
"""Quantize a DFlash2 drafter to Quark OCP-MXFP4 with AWQ input-channel scaling.

Usage (inside the serving image, GPU visible):
  python3 quantize_dflash_mxfp4.py SRC_BF16_DIR CALIB_PREFIX OUT_DIR

WHY THE SCALES HAVE TO BE FOLDED. AWQ divides a layer's input by a per-channel vector s and
multiplies the weight columns by s, so the product is unchanged but the large-activation channels
land in a friendlier part of the 4-bit grid. There is nowhere in a quark MXFP4 checkpoint to store
s -- the format is packed e2m1 plus an E8M0 exponent per 32 weights -- so s must be folded into
whatever produces the input.

DFlash2 makes that non-obvious, because a *dynamic* two-tap grouped conv sits between the norm and
the projections:

    h = input_layernorm(h)
    h, coeff = attention_conv.prepare(h)        # coeff = kernel_projection(h); then conv(h, coeff)
    h = self_attn(h)                            # qkv_proj
    h = attention_conv.finish(h, coeff)

The conv itself is per-channel and linear in h, so a per-channel 1/s commutes through it. The
coefficients do not: they come from kernel_projection(h), so scaling h changes them. Folding s into
the norm alone silently alters the conv and therefore the model. The fix is to compensate:

    input_layernorm.weight            /= s      # h' = h/s
    attention_conv.kernel_projection  *= s      # sees h again -> coefficients unchanged
    q_proj, k_proj, v_proj            *= s      # absorbs the 1/s the conv passed through

kernel_projection stays BF16 (the model builds it with quant_config=None), so multiplying it back
up costs only bf16 rounding. The MLP side is identical with post_attention_layernorm / mlp_conv.

down_proj folds into up_proj's OUTPUT rows instead: its input is silu(gate)*up, and scaling up's
rows by 1/s scales the product by 1/s with gate untouched. That is a different axis from the
gate_up column scale, so the two do not interact.

o_proj and fc are quantized round-to-nearest with no scaling. o_proj could fold into v_proj's rows,
but with GQA (32 q heads over 8 kv heads) s would have to be constant across each group of four,
and fc's input is the TARGET model's hidden states, which we do not own.

The alpha search uses a diagonal surrogate for the output error, ||(Wq/s - W) diag(a)||_F, where a
is the measured per-channel mean |x|. Real AWQ minimises against cached activations; we only kept
per-channel statistics, and this is the same objective under the assumption that input channels are
uncorrelated.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

# e2m1: 8 magnitudes, sign in the high bit of the nibble.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
GROUP = 32


def quantize_mxfp4(w: torch.Tensor, chunk: int = 2048):
    """[N, K] float -> (packed uint8 [N, K/2], e8m0 uint8 [N, K/32]). Round-half-even.

    Chunked over rows: the nearest-magnitude search materialises an [N, K/32, 32, 8] distance
    tensor, which for gate_up ([34816, 5120]) is 1.4 G elements and will not fit.
    """
    if w.shape[0] > chunk:
        ps, es = [], []
        for i in range(0, w.shape[0], chunk):
            a, b = quantize_mxfp4(w[i:i + chunk], chunk)
            ps.append(a); es.append(b)
        return torch.cat(ps), torch.cat(es)
    n, k = w.shape
    assert k % GROUP == 0, k
    wb = w.float().reshape(n, k // GROUP, GROUP)
    amax = wb.abs().amax(-1)
    # OCP: the block scale is a power of two chosen so the block maximum lands at the top of the
    # element range. e2m1's largest normal is 6.0 = 1.5 * 2^2, hence the -2.
    exp = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 2.0, torch.zeros_like(amax))
    exp = exp.clamp(-127, 127)
    scale = torch.exp2(exp)
    v = wb / scale.unsqueeze(-1)
    sign = torch.signbit(v)
    mag = v.abs().clamp(max=E2M1_MAX)
    grid = E2M1.to(w.device)
    # nearest representable magnitude, ties to even code
    d = (mag.unsqueeze(-1) - grid).abs()
    code = d.argmin(-1).to(torch.uint8)
    tie = (d.min(-1).values.unsqueeze(-1) == d).sum(-1) > 1
    if tie.any():
        even = (code // 2) * 2
        code = torch.where(tie, even.to(torch.uint8), code)
    code = code | (sign.to(torch.uint8) << 3)
    code = code.reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    e8m0 = (exp + 127).to(torch.uint8).contiguous()
    return packed, e8m0


def dequantize_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor, k: int):
    n = packed.shape[0]
    code = torch.empty(n, k, dtype=torch.uint8, device=packed.device)
    code[:, 0::2] = packed & 0xF
    code[:, 1::2] = packed >> 4
    grid = E2M1.to(packed.device)
    mag = grid[(code & 0x7).long()]
    val = torch.where((code & 0x8) > 0, -mag, mag)
    scale = torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    return (val.reshape(n, k // GROUP, GROUP) * scale).reshape(n, k)


def search_scale(W: torch.Tensor, a: torch.Tensor, grid=None, max_rows: int = 2048):
    """AWQ alpha search. W is [N, K] (K = input channels), a is [K] mean |x|.

    s is per-COLUMN and the objective averages over rows, so a strided row subsample estimates it
    to well under the spacing of the alpha grid while keeping the search affordable -- the full
    matrices here are up to 34816 x 5120 and every candidate alpha costs a quantize+dequantize.
    """
    if grid is None:
        grid = [i / 10 for i in range(11)]
    if W.shape[0] > max_rows:
        W = W[:: max(1, W.shape[0] // max_rows)][:max_rows]
    W = W.float()
    a = a.float().clamp_min(1e-8)
    an = a / a.mean()
    best, best_s, best_alpha = None, None, None
    for alpha in grid:
        s = an.pow(alpha).clamp(1e-2, 1e2)
        s = s / s.log().mean().exp()          # geometric mean 1: keeps the norm weights sane
        q, e = quantize_mxfp4(W * s)
        err = (dequantize_mxfp4(q, e, W.shape[1]) / s - W).mul(a).pow(2).sum()
        if best is None or err < best:
            best, best_s, best_alpha = err, s, alpha
    return best_s, best_alpha, float(best)


def main():
    src, calib_prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dev = "cuda"
    T = load_file(os.path.join(src, "model.safetensors"))
    T = {k: v.to(dev) for k, v in T.items()}
    cfg = json.load(open(os.path.join(src, "config.json")))
    nl = cfg["num_hidden_layers"]

    c0 = torch.load(f"{calib_prefix}.tp0.pt", map_location=dev)
    c1 = torch.load(f"{calib_prefix}.tp1.pt", map_location=dev)

    def act(name, sharded):
        if sharded:      # row-parallel: each rank saw half the input channels
            return torch.cat([c0[name]["absmean"].to(dev), c1[name]["absmean"].to(dev)])
        return c0[name]["absmean"].to(dev)

    report = []

    for l in range(nl):
        P = f"layers.{l}"
        # ---- attention: fold into input_layernorm, compensate the conv's kernel_projection ----
        qn, kn, vn = (f"{P}.self_attn.{x}_proj.weight" for x in "qkv")
        a = act(f"{P}.self_attn.qkv_proj", False)
        W = torch.cat([T[qn], T[kn], T[vn]], 0)
        s, alpha, err = search_scale(W, a)
        T[f"{P}.input_layernorm.weight"] = (T[f"{P}.input_layernorm.weight"].float() / s).to(torch.bfloat16)
        kp = f"{P}.attention_conv.kernel_projection.weight"
        T[kp] = (T[kp].float() * s).to(torch.bfloat16)
        for nme in (qn, kn, vn):
            T[nme] = (T[nme].float() * s).to(torch.bfloat16)
        report.append((f"{P}.qkv", alpha))

        # ---- mlp: fold into post_attention_layernorm, compensate mlp_conv ----
        gn, un, dn = (f"{P}.mlp.{x}_proj.weight" for x in ("gate", "up", "down"))
        a = act(f"{P}.mlp.gate_up_proj", False)
        W = torch.cat([T[gn], T[un]], 0)
        s, alpha, err = search_scale(W, a)
        T[f"{P}.post_attention_layernorm.weight"] = (
            T[f"{P}.post_attention_layernorm.weight"].float() / s).to(torch.bfloat16)
        kp = f"{P}.mlp_conv.kernel_projection.weight"
        T[kp] = (T[kp].float() * s).to(torch.bfloat16)
        for nme in (gn, un):
            T[nme] = (T[nme].float() * s).to(torch.bfloat16)
        report.append((f"{P}.gate_up", alpha))

        # ---- down_proj: fold into up_proj's output rows (silu(gate)*up scales with up) ----
        a = act(f"{P}.mlp.down_proj", True)
        s, alpha, err = search_scale(T[dn], a)
        T[un] = (T[un].float() / s.unsqueeze(1)).to(torch.bfloat16)
        T[dn] = (T[dn].float() * s).to(torch.bfloat16)
        report.append((f"{P}.down", alpha))

    # ---- quantize every linear the drafter serves quantized ----
    targets = []
    for l in range(nl):
        P = f"layers.{l}"
        targets += [f"{P}.self_attn.{x}_proj.weight" for x in ("q", "k", "v", "o")]
        targets += [f"{P}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
    targets.append("fc.weight")

    out_t, tot_err, nq = {}, [], 0
    for k, v in T.items():
        if k in targets:
            w = v.float()
            packed, e8m0 = quantize_mxfp4(w)
            deq = dequantize_mxfp4(packed, e8m0, w.shape[1])
            rel = (deq - w).norm() / w.norm()
            tot_err.append((k, float(rel)))
            out_t[k] = packed.cpu()
            out_t[k.replace(".weight", ".weight_scale")] = e8m0.cpu()
            nq += 1
        else:
            out_t[k] = v.to(torch.bfloat16).cpu()

    os.makedirs(out, exist_ok=True)
    save_file(out_t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    cfg["quantization_config"] = {
        "algo_config": None,
        "exclude": [],
        "export": {"kv_cache_group": [], "min_kv_scale": 0.0, "pack_method": "reorder",
                   "weight_format": "real_quantized", "weight_merge_groups": None},
        "global_quant_config": {
            "bias": None, "output_tensors": None, "target_device": None,
            "input_tensors": {"block_size": None, "ch_axis": -1, "dtype": "fp4",
                              "enable_buffer_reuse": False, "group_size": 32, "is_dynamic": True,
                              "is_scale_quant": False, "max_input_numel": 4194304,
                              "mx_element_dtype": None, "observer_cls": "PerBlockMXObserver",
                              "qscheme": "per_group", "round_method": "half_even",
                              "scale_calculation_mode": "even", "scale_format": "e8m0",
                              "scale_type": "float", "symmetric": None},
            "weight": {"block_size": None, "ch_axis": -1, "dtype": "fp4",
                       "enable_buffer_reuse": False, "group_size": 32, "is_dynamic": False,
                       "is_scale_quant": False, "max_input_numel": 4194304,
                       "mx_element_dtype": None, "observer_cls": "PerBlockMXObserver",
                       "qscheme": "per_group", "round_method": "half_even",
                       "scale_calculation_mode": "even", "scale_format": "e8m0",
                       "scale_type": "float", "symmetric": None}},
        "kv_cache_post_rope": False, "kv_cache_quant_config": {},
        "layer_quant_config": {}, "layer_type_quant_config": {},
        "quant_method": "quark", "quant_mode": "eager_mode",
        "softmax_quant_spec": None, "version": "0.12+radiance-awq-mxfp4",
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    tot_err.sort(key=lambda x: -x[1])
    print(f"quantized {nq} linears -> {out}")
    print("alpha chosen per group (0 = no scaling, 1 = full activation weighting):")
    for name, al in report:
        print(f"    {name:<28} alpha={al:.2f}")
    print("worst relative weight error:")
    for name, e in tot_err[:6]:
        print(f"    {name:<44} {e:.4f}")
    print(f"  mean {sum(e for _, e in tot_err)/len(tot_err):.4f} over {len(tot_err)} tensors")


if __name__ == "__main__":
    main()
