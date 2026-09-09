"""Kernel-level decode attribution: int4 PARO linear vs MXFP4-PARO linear, per real shape, TP=2 rank.

Loads the real on-disk buffers of one module per shape class from both checkpoints (z-lab int4 AWQ
and the hybrid MXFP4-ft), emulates TP=2 (column-parallel: first N/2 rows; row-parallel: first K/2
input channels -- exact because rotation pairs are group-local), preps them through the two serving
methods, and times method.apply(x) with events at decode-band M. For the MXFP4 method both
prologues are timed: two-pass (rotate -> token quant) and the fused single-launch kernel.
The int4 numbers are the streams-OFF path (separate rotate+quant launch), which int4 prod improves
on further with the fused norm producers, so int4 here is a slightly pessimistic stand-in.
Ends with a per-step estimate (16 attention + 48 GDN layers) for each variant.
"""
import os, sys, glob, json, time
import torch, torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, "/patches/paroquant")
import radiance_paroquant as I4        # noqa
import radiance_paroquant_mxfp4 as MX  # noqa

INT4 = os.environ.get("BL_INT4", "/models/Qwen3.8-27B-PARO")
MXF = os.environ.get("BL_MXFP4", "/models/Qwen3.8-27B-PARO-MXFP4-ft")
MS = [int(v) for v in (os.environ.get("BL_MS") or "1,8,16,64").split(",")]
ITERS = int(os.environ.get("BL_ITERS", "200"))
dev = "cuda"


def weight_map(ck):
    idx = f"{ck}/model.safetensors.index.json"
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for fn in glob.glob(f"{ck}/*.safetensors"):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                wm[k] = os.path.basename(fn)
    return wm


WM4, WMX = weight_map(INT4), weight_map(MXF)


def get(ck, wm, name):
    with safe_open(f"{ck}/{wm[name]}", "pt") as f:
        return f.get_tensor(name)


def leaves(wm, prefix):
    return sorted({k[len(prefix):].rsplit(".", 1)[0] for k in wm if k.startswith(prefix)})


# shape classes: (label, layer, [module names], row_parallel, count_per_step)
L0 = "model.language_model.layers.0."
L3 = "model.language_model.layers.3."
print("layer0 modules (int4):", leaves(WM4, L0))
print("layer3 modules (int4):", leaves(WM4, L3))
SHAPES = [
    ("gdn in_proj (P=2)", 0, ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], False, 48),
    ("gdn out_proj", 0, ["linear_attn.out_proj"], True, 48),
    ("mlp gate_up (P=2)", 0, ["mlp.gate_proj", "mlp.up_proj"], False, 64),
    ("mlp down", 0, ["mlp.down_proj"], True, 64),
    ("attn qkv (P=3)", 3, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], False, 16),
    ("attn o_proj", 3, ["self_attn.o_proj"], True, 16),
]


def build_int4(layer_idx, mods, rowpar):
    pre = f"model.language_model.layers.{layer_idx}."
    qw, sc, qz, th, pr, cs = [], [], [], [], [], []
    for m in mods:
        q = get(INT4, WM4, pre + m + ".qweight"); s = get(INT4, WM4, pre + m + ".scales"); z = get(INT4, WM4, pre + m + ".qzeros")
        t = get(INT4, WM4, pre + m + ".theta"); p = get(INT4, WM4, pre + m + ".pairs"); c = get(INT4, WM4, pre + m + ".channel_scales").reshape(-1)
        K, N8 = q.shape; G = s.shape[0]
        if rowpar:
            q = q[:K // 2]; s = s[:G // 2]; z = z[:G // 2]; t = t[:, :K // 4]; p = p[:, :K // 2]; c = c[:K // 2]
        else:
            q = q[:, :N8 // 2]; s = s[:, :s.shape[1] // 2]; z = z[:, :z.shape[1] // 2]
        qw.append(q.contiguous()); sc.append(s.contiguous()); qz.append(z.contiguous()); th.append(t); pr.append(p); cs.append(c)
    layer = nn.Module()
    layer.qweight = nn.Parameter(torch.cat(qw, 1), requires_grad=False)
    layer.scales = nn.Parameter(torch.cat(sc, 1), requires_grad=False)
    layer.qzeros = nn.Parameter(torch.cat(qz, 1), requires_grad=False)
    layer.theta = nn.Parameter(torch.stack(th), requires_grad=False)
    layer.pairs = nn.Parameter(torch.stack(pr), requires_grad=False)
    layer.channel_scales = nn.Parameter(torch.stack(cs), requires_grad=False)
    layer.pq_output_partition_sizes = [s.shape[1] for s in sc]
    layer.to(dev)
    cfg = I4.ParoQuantConfig.from_config({"bits": 4, "group_size": 128, "krot": 8})
    meth = I4.ParoQuantLinearMethod(cfg)
    meth.process_weights_after_loading(layer)
    return meth, layer, layer.qweight.shape[0] if False else layer.cs.shape[1], sum(layer.pq_output_partition_sizes)


def build_mx(layer_idx, mods, rowpar):
    pre = f"model.language_model.layers.{layer_idx}."
    w, ws, th, pr, cs = [], [], [], [], []
    for m in mods:
        a = get(MXF, WMX, pre + m + ".weight"); b = get(MXF, WMX, pre + m + ".weight_scale")
        t = get(MXF, WMX, pre + m + ".theta"); p = get(MXF, WMX, pre + m + ".pairs"); c = get(MXF, WMX, pre + m + ".channel_scales").reshape(-1)
        N, K2 = a.shape; K = K2 * 2
        if rowpar:
            a = a[:, :K2 // 2]; b = b[:, :b.shape[1] // 2]; t = t[:, :K // 4]; p = p[:, :K // 2]; c = c[:K // 2]
        else:
            a = a[:N // 2]; b = b[:N // 2]
        w.append(a.contiguous()); ws.append(b.contiguous()); th.append(t); pr.append(p); cs.append(c)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.cat(w), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.cat(ws), requires_grad=False)
    layer.theta = nn.Parameter(torch.stack(th), requires_grad=False)
    layer.pairs = nn.Parameter(torch.stack(pr), requires_grad=False)
    layer.channel_scales = nn.Parameter(torch.stack(cs), requires_grad=False)
    layer.pq_output_partition_sizes = [a.shape[0] for a in w]
    layer.to(dev)
    cfg = MX.ParoQuantMXFP4Config.from_config({"bits": 4, "group_size": 128, "krot": 8})
    meth = MX.ParoQuantMXFP4LinearMethod(cfg)
    meth.process_weights_after_loading(layer)
    return meth, layer, layer.cs.shape[1], sum(layer.pq_output_partition_sizes)


def time_us(fn, iters):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(3):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters): fn()
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000.0 / iters)
    return best


totals = {}   # variant -> {M: us per step}
print(f"\n{'shape':22s} {'K':>6s} {'N':>6s} {'M':>3s} | {'int4 us':>9s} | {'mx 2pass':>9s} {'mx fused':>9s} {'+single':>9s} | single/int4")
for label, li, mods, rowpar, cnt in SHAPES:
    m4, l4, K4, N4 = build_int4(li, mods, rowpar)
    mm, lm, Km, Nm = build_mx(li, mods, rowpar)
    assert (K4, N4) == (Km, Nm), (K4, N4, Km, Nm)
    for M in MS:
        x = (torch.randn(M, K4, device=dev) * 0.8).to(torch.bfloat16)
        t4 = time_us(lambda: m4.apply(l4, x), ITERS)
        MX.SINGLE_LAUNCH = False
        MX.FUSED_TOKQ = False; t2 = time_us(lambda: mm.apply(lm, x), ITERS)
        MX.FUSED_TOKQ = True;  tf = time_us(lambda: mm.apply(lm, x), ITERS)
        MX.SINGLE_LAUNCH = True; ts = time_us(lambda: mm.apply(lm, x), ITERS)
        print(f"{label:22s} {K4:6d} {N4:6d} {M:3d} | {t4:9.1f} | {t2:9.1f} {tf:9.1f} {ts:9.1f} | {ts / t4:5.2f}x")
        for name, t in (("int4", t4), ("mx two-pass", t2), ("mx fused", tf), ("mx fused+single", ts)):
            totals.setdefault(name, {}).setdefault(M, 0.0)
            totals[name][M] += t * cnt
    del m4, l4, mm, lm; torch.cuda.empty_cache()

print("\nper-step linear-only estimate (16 attn + 48 GDN layers), ms:")
print(f"{'variant':12s} " + " ".join(f"{'M=%d' % M:>8s}" for M in MS))
for name, d in totals.items():
    print(f"{name:12s} " + " ".join(f"{d[M] / 1000:8.2f}" for M in MS))
