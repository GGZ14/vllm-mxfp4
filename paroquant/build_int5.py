"""int5 W5A8 ParoQuant checkpoint, round-to-nearest, from the bf16 base + z-lab's TRAINED rotations.

Grid: uniform asymmetric int5, group 128 along K, fp16 scale + integer zero point -- exactly
UniformAffineQuantizer.pseudo_quantize with n_bits=5 (scale = (max-min)/31, zero = clamp(-round(min/scale)),
codes = clamp(round(w/scale) + zero, 0, 31)), so a later stage-2 fine-tune (optimize --n-bit 5) starts
from the same grid. Rotations/channel scales are z-lab's (bit width independent), as build_hybrid.py.

Checkpoint (quant_method paroquant, bits 5, "int5-bitplane"):
  qweight     [K, N/8]  int32  AWQ packing of the LOW nibble (code & 15)   -- the int4 loader's layout
  qweight_hi  [K, N/32] int32  bit (n % 32) of word n // 32 = code >> 4      -- the fifth bit, one plane
  qzeros      [G, N/8]  int32  AWQ packing of zero & 15;  qzeros_hi [G, N/32] int32 the zero's fifth bit
  scales      [G, N]    fp16;  theta / pairs / channel_scales as z-lab ships them
The serving kernel feeds the fp8 WMMA the signed code (c - 16), which is exact in e4m3 for all 32 codes.
WRITE_PSEUDO=1 also writes the fp16 pseudo-quantized model (55 GiB; the stock-path accuracy gate).
"""
import json, os, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, "/src")
from paroquant.kernels.cuda import scaled_pairwise_rotation
from paroquant.optim.quantizer import _calc_scales_and_zero_points
from paroquant.cli.convert import _pack_awq

BITS, QMAX, GS, dev = 5, 31, 128, "cuda"
BASE, PARO = Path("/models/Qwen3.8-27B-bf16"), Path("/models/Qwen3.8-27B-PARO")
OUT_REAL = Path(os.environ.get("OUT_REAL", "/models/Qwen3.8-27B-PARO-int5-rtn"))
WRITE_PSEUDO = os.environ.get("WRITE_PSEUDO", "0") == "1"
OUT_PSEUDO = Path(os.environ.get("OUT_PSEUDO", "/models/Qwen3.8-27B-PARO-int5-rtn-pseudo"))
OUT_REAL.mkdir(parents=True, exist_ok=True)
if WRITE_PSEUDO: OUT_PSEUDO.mkdir(parents=True, exist_ok=True)

paro = safe_open(PARO / "model.safetensors", framework="pt")
quantized = {k[:-len(".theta")] for k in paro.keys() if k.endswith(".theta")}
print(f"z-lab quantized modules: {len(quantized)} | int{BITS} g{GS} asymmetric RTN", flush=True)
index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
shards = sorted(set(index.values()))
pseudo_map, real_map = {}, {}
n_q = 0; worst_rt = 0.0; t0 = time.time()

def rot(w, pairs, theta):   return scaled_pairwise_rotation(w, pairs, theta, None, GS)
def unrot(w, pairs, theta): return scaled_pairwise_rotation(w, torch.flip(pairs, [0]), -torch.flip(theta, [0]), None, GS)

def bitplane(v):            # [R, C] int codes -> [R, C/32] int32, bit (c % 32) of word c // 32 = v >> 4
    hi = ((v >> 4) & 1).to(torch.int64).view(v.shape[0], -1, 32)
    w = (hi << torch.arange(32, device=v.device, dtype=torch.int64)).sum(-1)
    return ((w + 2**31) % 2**32 - 2**31).to(torch.int32)

for si, shard in enumerate(shards):
    pseudo_t, real_t = {}, {}
    with safe_open(BASE / shard, framework="pt") as f:
        for name in f.keys():
            if name.startswith("mtp."): continue
            mod = name[:-len(".weight")] if name.endswith(".weight") else None
            if mod in quantized:
                w = f.get_tensor(name).to(dev, torch.float32)                       # [N, K]
                N, K = w.shape; G = K // GS
                pairs = paro.get_tensor(f"{mod}.pairs").to(dev)
                theta = paro.get_tensor(f"{mod}.theta").to(dev, torch.float32)
                cs_opt = (1.0 / paro.get_tensor(f"{mod}.channel_scales").to(dev, torch.float32)).view(1, -1)
                w_rot = rot(w * cs_opt, pairs, theta)
                scale, zpf = _calc_scales_and_zero_points(w_rot, GS, 0, QMAX)      # [N*G, 1]
                scale = scale.clamp(min=1e-5, max=1e5)
                zero = torch.clamp(-torch.round(zpf), 0, QMAX)
                codes = torch.clamp(torch.round(w_rot.reshape(-1, GS) / scale) + zero, 0, QMAX)
                w_rot_q = ((codes - zero) * scale).reshape(N, K)
                codes = codes.reshape(N, G, GS).to(torch.int32).reshape(N, K)        # [N, K]
                zero = zero.reshape(N, G).to(torch.int32)                             # [N, G]
                scale = scale.reshape(N, G)
                real_t[f"{mod}.qweight"] = _pack_awq((codes & 15).T.contiguous()).cpu()            # [K, N/8]
                real_t[f"{mod}.qweight_hi"] = bitplane(codes.T.contiguous()).cpu()                 # [K, N/32]
                real_t[f"{mod}.qzeros"] = _pack_awq((zero & 15).T.contiguous()).cpu()              # [G, N/8]
                real_t[f"{mod}.qzeros_hi"] = bitplane(zero.T.contiguous()).cpu()                   # [G, N/32]
                real_t[f"{mod}.scales"] = scale.T.contiguous().to(torch.float16).cpu()             # [G, N]
                for leaf in ("theta", "pairs", "channel_scales"):
                    real_t[f"{mod}.{leaf}"] = paro.get_tensor(f"{mod}.{leaf}")
                if WRITE_PSEUDO:
                    w_pseudo = unrot(w_rot_q, pairs, theta) / cs_opt
                    rt = ((rot(w_pseudo * cs_opt, pairs, theta) - w_rot_q).norm() / w_rot_q.norm()).item()
                    worst_rt = max(worst_rt, rt)
                    pseudo_t[name] = w_pseudo.to(torch.float16).cpu()
                n_q += 1
            else:
                t = f.get_tensor(name); t = t.to(torch.float16) if t.is_floating_point() else t
                real_t[name] = t
                if WRITE_PSEUDO: pseudo_t[name] = t
    save_file(real_t, OUT_REAL / shard, metadata={"format": "pt"})
    for k in real_t: real_map[k] = shard
    if WRITE_PSEUDO:
        save_file(pseudo_t, OUT_PSEUDO / shard, metadata={"format": "pt"})
        for k in pseudo_t: pseudo_map[k] = shard
    del pseudo_t, real_t; torch.cuda.empty_cache()
    print(f"  shard {si+1}/{len(shards)} {shard}  quantized so far {n_q}  worst round-trip {worst_rt:.2e}  {time.time()-t0:.0f}s", flush=True)

assert n_q == len(quantized), (n_q, len(quantized))
cfg = json.load(open(PARO / "config.json"))
outs = [(OUT_REAL, real_map)] + ([(OUT_PSEUDO, pseudo_map)] if WRITE_PSEUDO else [])
for out, wmap in outs:
    json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
    for fn in PARO.iterdir():
        if fn.suffix in (".json", ".jinja", ".txt") and fn.name != "model.safetensors.index.json":
            shutil.copy(fn, out / fn.name)
qc = dict(cfg["quantization_config"]); qc.update({"quant_method": "paroquant", "bits": BITS, "group_size": GS,
        "format": "int5-bitplane", "rotations": "z-lab/Qwen3.8-27B-PARO", "codes": "rtn"})
cfg_r = dict(cfg); cfg_r["quantization_config"] = qc
json.dump(cfg_r, open(OUT_REAL / "config.json", "w"), indent=2)
if WRITE_PSEUDO:
    cfg_p = dict(cfg); cfg_p.pop("quantization_config", None); cfg_p.pop("torch_dtype", None)
    json.dump(cfg_p, open(OUT_PSEUDO / "config.json", "w"), indent=2)
print(f"DONE: {n_q} modules -> {OUT_REAL}, {time.time()-t0:.0f}s", flush=True)
