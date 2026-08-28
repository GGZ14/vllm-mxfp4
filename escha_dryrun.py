"""Validate the escha quant method without starting a server.

Covers the three things that can silently be wrong in an integration and that the kernel gates
cannot see: which modules get claimed, whether the per-shard weight loader puts the right bytes in
the right place, and whether apply() reproduces the reference chain end to end on REAL weights.

Runs at TP=1 with the distributed helpers stubbed, so it needs no process group and only a few
hundred MB of VRAM -- it can run alongside a live server.
"""
import json, os, struct, sys
import numpy as np
import torch

sys.path.insert(0, "/esc")
MODEL = os.environ.get("ESCHA_MODEL", "/models/Qwen3.8-27B-Escha-W2")

# TP=1 stubs, installed before radiance_escha imports them.
import vllm.distributed as _d
_d.get_tensor_model_parallel_world_size = lambda: 1
_d.get_tensor_model_parallel_rank = lambda: 0

import radiance_escha as R
Cfg = R.register()

cfg_json = json.load(open(f"{MODEL}/config.json"))["quantization_config"]
cfg = Cfg.from_config(cfg_json)
print(cfg)

# ---- 1. routing -------------------------------------------------------------------------------
P = "model.language_model.layers.0"
checks = [(f"{P}.mlp.gate_up_proj", True), (f"{P}.mlp.down_proj", True),
          (f"{P}.linear_attn.in_proj_qkvz", True), (f"{P}.linear_attn.out_proj", True),
          (f"{P}.linear_attn.in_proj_ba", False), ("lm_head", False)]
bad = 0
for pref, want in checks:
    got = cfg.is_coded(pref)
    ok = got == want
    bad += not ok
    print(f"  route {pref.split('layers.')[-1]:<28} coded={got!s:<5} expect={want!s:<5} "
          f"{'ok' if ok else 'MISMATCH'}"
          + (f"  K={cfg.kbits_for(pref)}" if got else ""))

# ---- 2. per-shard weight loading --------------------------------------------------------------
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
_h = {}
def get(nm, dt):
    p = f"{MODEL}/{idx[nm]}"
    if p not in _h:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; _h[p] = (json.loads(f.read(n)), 8 + n)
    h, off = _h[p]; o = h[nm]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(off + o[0]); b = f.read(o[1] - o[0])
    return torch.from_numpy(np.frombuffer(b, dt).reshape(h[nm]["shape"]).copy())

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
class Fake(MergedColumnParallelLinear):
    def __init__(self):
        torch.nn.Module.__init__(self)

prefix = f"{P}.mlp.gate_up_proj"
method = R._linear_method_cls()(cfg, prefix)
layer = Fake()
IC, OCs = 5120, [17408, 17408]           # gate and up, TP=1
method.create_weights(layer, IC, OCs, IC, sum(OCs), torch.bfloat16, weight_loader=None)
print(f"\n  create_weights ok: kbits={layer.escha_kbits} row_parallel={layer.escha_row_parallel}")

DT = {"escha_code": np.int16, "escha_rin": np.float16, "escha_rout": np.float16,
      "escha_s_in": np.float32, "escha_s_out": np.float32, "escha_config": np.int32}
for shard, src in enumerate(("gate_proj", "up_proj")):
    for suf, dt in DT.items():
        t = get(f"{P}.mlp.{src}.{suf}", dt)
        getattr(layer, suf).weight_loader(getattr(layer, suf), t, shard)
method.process_weights_after_loading(layer)
print(f"  loaded: code {[tuple(c.shape) for c in layer.escha_code]} "
      f"rin {[tuple(r.shape) for r in layer.escha_rin]} rout {[tuple(r.shape) for r in layer.escha_rout]}")

# ---- 3. apply() against the reference chain ----------------------------------------------------
sys.path.insert(0, "/esc/escha")
import exl3_ref as REF

def to_e4m3(a):
    """Round to e4m3 exactly as the hardware does (round-to-nearest-even), via torch's own type.
    An earlier hand-rolled version rounded ties DOWN and the discrepancy only showed up on real
    trellis values, where ties are ~1% of elements."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return t.to(torch.float8_e4m3fn).float().numpy().astype(np.float64)

def fwht128(v):
    H = REF.hadamard(128) / np.sqrt(128.0)
    s = v.shape
    return (v.reshape(*s[:-1], s[-1] // 128, 128) @ H).reshape(s)

dev = torch.device("cuda")
for k in range(len(layer.escha_code)):
    layer.escha_code[k] = layer.escha_code[k].to(dev)
    layer.escha_rout[k] = layer.escha_rout[k].to(dev)
    layer.escha_s_out[k] = layer.escha_s_out[k].to(dev)
    layer.escha_rin[k] = layer.escha_rin[k].to(dev)
    layer.escha_s_in[k] = layer.escha_s_in[k].to(dev)

M, OCTEST = 8, 256
torch.manual_seed(0)
x = (torch.randn(M, IC, dtype=torch.bfloat16, device=dev) * 0.05)
y = method.apply(layer, x)
print(f"\n  apply() -> {tuple(y.shape)} {y.dtype}")

# reference for the FIRST shard's leading OCTEST columns
sh = 0
K = layer.escha_kbits[sh]
code = get(f"{P}.mlp.gate_proj.escha_code", np.int16).numpy()[:, :OCTEST // 16]
words = np.ascontiguousarray(code).view(np.uint16).view(np.uint32)
rin = layer.escha_rin[sh].float().cpu().numpy().astype(np.float64)
sin = layer.escha_s_in[sh].cpu().numpy().astype(np.float64)
rout = layer.escha_rout[sh][:OCTEST].float().cpu().numpy().astype(np.float64)
sout = layer.escha_s_out[sh][:OCTEST].cpu().numpy().astype(np.float64)
xr = x.float().cpu().numpy().astype(np.float64)
h = fwht128(xr * sin * rin)
As = np.abs(h).max(axis=1) / 448.0
A = to_e4m3(h / As[:, None])
W = np.zeros((IC, OCTEST), np.float64)
for i in range(words.shape[0]):
    for j in range(words.shape[1]):
        W[i*16:(i+1)*16, j*16:(j+1)*16] = REF.decode_tile(words[i, j], K)
C = (A @ to_e4m3(W)) * As[:, None]
ref = fwht128(C) * rout * sout
got = y[:, :OCTEST].float().cpu().numpy().astype(np.float64)
rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
ok = rel < 3e-2
bad += not ok
print(f"  apply vs reference chain: rel={rel:.3e}  {'OK' if ok else 'FAIL'}")
print(f"\ndryrun: {'FAIL' if bad else 'PASS'}")
sys.exit(1 if bad else 0)
