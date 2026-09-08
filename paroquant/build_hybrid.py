"""Tier 1 hybrid: bf16 base weights + z-lab's TRAINED rotations -> MXFP4, one shot, no optimizer.

Why: our own calibration initializes rotations at identity and, with the samples/epochs this box
can afford, leaves them there on most layers (layer 28: 100% zero angles). z-lab's were trained on
2048 samples x 5 epochs and average 1.7-3.1 rad on every layer. A rotation that tames per-group
outliers for int4 does the same for e2m1, so we reuse theirs and only change the weight grid.

Verified beforehand (convention_check.py): z-lab's stored pairs/theta/channel_scales reproduce
the optimizer's rotate(W_bf16 * cs) to within int4 noise (0.104 vs 0.1025 expected).

Writes two checkpoints, mirroring the base's shard layout:
  pseudo : fp16 weights already through rotate -> MXFP4 -> inverse-rotate; loads on the stock
           path and carries the scheme's exact error. This is the accuracy gate.
  real   : packed e2m1 [N,K/2] u8 + e8m0 [N,K/32] u8 + z-lab's theta/pairs/channel_scales per
           quantized module; quant_method paroquant_mxfp4. This is what the serving kernel eats.
"""
import json, os, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, "/src")
from paroquant.kernels.cuda import scaled_pairwise_rotation
from paroquant.optim.mxfp4 import quantize_to_codes, dequantize_from_codes, scale_rule

BASE, PARO = Path("/models/Qwen3.8-27B-bf16"), Path("/models/Qwen3.8-27B-PARO")
OUT_PSEUDO = Path(os.environ.get("OUT_PSEUDO", "/models/Qwen3.8-27B-PARO-MXFP4-pseudo"))
OUT_REAL = Path(os.environ.get("OUT_REAL", "/models/Qwen3.8-27B-PARO-MXFP4"))
GS, dev = 128, "cuda"
for d in (OUT_PSEUDO, OUT_REAL):
    d.mkdir(parents=True, exist_ok=True)

paro = safe_open(PARO / "model.safetensors", framework="pt")
paro_keys = set(paro.keys())
quantized = {k[:-len(".theta")] for k in paro_keys if k.endswith(".theta")}
print(f"z-lab quantized modules: {len(quantized)} | mxfp4 scale rule: {scale_rule()}", flush=True)

index = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
shards = sorted(set(index.values()))
pseudo_map, real_map = {}, {}
n_q = 0; worst_rt = 0.0; t0 = time.time()

def rot(w, pairs, theta):        return scaled_pairwise_rotation(w, pairs, theta, None, GS)
def unrot(w, pairs, theta):      return scaled_pairwise_rotation(w, torch.flip(pairs, [0]), -torch.flip(theta, [0]), None, GS)

for si, shard in enumerate(shards):
    pseudo_t, real_t = {}, {}
    with safe_open(BASE / shard, framework="pt") as f:
        for name in f.keys():
            if name.startswith("mtp."):            # z-lab ships none; the drafter is external
                continue
            mod = name[:-len(".weight")] if name.endswith(".weight") else None
            if mod in quantized:
                w = f.get_tensor(name).to(dev, torch.float32)                     # [N, K]
                pairs = paro.get_tensor(f"{mod}.pairs").to(dev)
                theta = paro.get_tensor(f"{mod}.theta").to(dev, torch.float32)
                cs_stored = paro.get_tensor(f"{mod}.channel_scales")             # pre-inverted
                cs_opt = (1.0 / cs_stored.to(dev, torch.float32)).view(1, -1)
                w_rot = rot(w * cs_opt, pairs, theta)
                packed, e8m0 = quantize_to_codes(w_rot)                           # OCP rule
                w_rot_q = dequantize_from_codes(packed, e8m0)
                w_pseudo = (unrot(w_rot_q, pairs, theta) / cs_opt)
                # self-check: pseudo must re-rotate onto the quantized weight (fp32 round trip)
                rt = ((rot(w_pseudo * cs_opt, pairs, theta) - w_rot_q).norm() / w_rot_q.norm()).item()
                worst_rt = max(worst_rt, rt)
                pseudo_t[name] = w_pseudo.to(torch.float16).cpu()
                real_t[f"{mod}.weight"] = packed.cpu()
                real_t[f"{mod}.weight_scale"] = e8m0.cpu()
                for leaf in ("theta", "pairs", "channel_scales"):
                    real_t[f"{mod}.{leaf}"] = paro.get_tensor(f"{mod}.{leaf}")
                n_q += 1
            else:
                t = f.get_tensor(name)
                t = t.to(torch.float16) if t.is_floating_point() else t
                pseudo_t[name] = t; real_t[name] = t
    save_file(pseudo_t, OUT_PSEUDO / shard, metadata={"format": "pt"})
    save_file(real_t, OUT_REAL / shard, metadata={"format": "pt"})
    for k in pseudo_t: pseudo_map[k] = shard
    for k in real_t: real_map[k] = shard
    del pseudo_t, real_t; torch.cuda.empty_cache()
    print(f"  shard {si+1}/{len(shards)} {shard}  quantized so far {n_q}  worst round-trip {worst_rt:.2e}  {time.time()-t0:.0f}s", flush=True)

assert n_q == len(quantized), (n_q, len(quantized))
for out, wmap in ((OUT_PSEUDO, pseudo_map), (OUT_REAL, real_map)):
    json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
    for fn in PARO.iterdir():                       # tokenizer, template, config: the proven set
        if fn.suffix in (".json", ".jinja", ".txt") and fn.name != "model.safetensors.index.json":
            shutil.copy(fn, out / fn.name)
cfg = json.load(open(PARO / "config.json"))
cfg_p = dict(cfg); cfg_p.pop("quantization_config", None); cfg_p["torch_dtype"] = "float16"
json.dump(cfg_p, open(OUT_PSEUDO / "config.json", "w"), indent=2)
cfg_r = dict(cfg); cfg_r["quantization_config"] = {"quant_method": "paroquant_mxfp4", "format": "mxfp4",
    "bits": 4, "group_size": GS, "mxfp4_block": 32, "krot": int(cfg["quantization_config"]["krot"]),
    "scale_rule": scale_rule(), "rotations": "z-lab/Qwen3.8-27B-PARO"}
json.dump(cfg_r, open(OUT_REAL / "config.json", "w"), indent=2)
print(f"DONE: {n_q} modules, worst pseudo round-trip rel {worst_rt:.2e}, {time.time()-t0:.0f}s", flush=True)
