"""Requant error on REAL tensors of unsloth/Qwen3.8-27B-NVFP4, against the NVFP4 dequant and against
the bf16 original (Qwen3.8-27B-bf16). Also the e8m0 dynamic range per row (the folded kernel flushes
blocks more than 12 binades below the row max)."""
import json, sys, torch
sys.path.insert(0, "/patches")
import radiance_nvfp4 as rn
from safetensors import safe_open
NV = "/models/Qwen3.8-27B-NVFP4"; BF = "/models/Qwen3.8-27B-bf16"
bfmap = json.load(open(f"{BF}/model.safetensors.index.json"))["weight_map"]
dev = "cpu"; torch.set_num_threads(16)
def rel(a, b): return float(((a - b) ** 2).sum().sqrt() / (b ** 2).sum().sqrt())
def sqnr(a, b): return 10 * torch.log10(((b ** 2).sum()) / (((a - b) ** 2).sum())).item()
layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,7,20,40,55").split(",")]
print(f"{'layer':>18} | nv/bf16 | ocp/nv noclip/nv mse/nv | ocp/bf16 mse/bf16 direct/bf16 | SQNR nv mse | maxd>12 rows")
tot = {}
with safe_open(f"{NV}/model.safetensors", "pt", device="cpu") as f:
    for L in layers:
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            base = f"model.language_model.layers.{L}.mlp.{proj}"
            p = f.get_tensor(f"{base}.weight_packed").to(dev)
            s = f.get_tensor(f"{base}.weight_scale").to(dev)
            g = float(f.get_tensor(f"{base}.weight_global_scale").float())
            w_nv = rn.dequant_nvfp4(p, s, g)
            with safe_open(f"{BF}/{bfmap[base + '.weight']}", "pt", device="cpu") as fb:
                w_bf = fb.get_tensor(base + ".weight").to(dev).float()
            out = {}
            for m in ["ocp", "noclip", "mse"]:
                pk, e8, _ = rn.quant_mxfp4(w_nv, m); out[m] = rn.dequant_mxfp4(pk, e8)
                if m == "mse":
                    ref = e8.max(dim=1).values.unsqueeze(1).float(); d = ref - e8.float()
                    nz = (pk.reshape(pk.shape[0], -1, 16) != 0).any(-1)   # blocks with any nonzero code
                    flush_rows = int(((d > 12) & nz).any(1).sum())
            pk, e8, _ = rn.quant_mxfp4(w_bf, "mse"); direct = rn.dequant_mxfp4(pk, e8)
            print(f"{L:>3} {proj:<14} | {rel(w_nv, w_bf):7.4f} | {rel(out['ocp'], w_nv):6.4f} {rel(out['noclip'], w_nv):9.4f} {rel(out['mse'], w_nv):6.4f} | "
                  f"{rel(out['ocp'], w_bf):8.4f} {rel(out['mse'], w_bf):8.4f} {rel(direct, w_bf):11.4f} | {sqnr(w_nv, w_bf):5.1f} {sqnr(out['mse'], w_bf):5.1f} | {flush_rows}", flush=True)
            for k, v in [("nv/bf16", rel(w_nv, w_bf)), ("mse/nv", rel(out['mse'], w_nv)), ("mse/bf16", rel(out['mse'], w_bf)), ("direct/bf16", rel(direct, w_bf))]:
                tot.setdefault(k, []).append(v)
print({k: round(sum(v) / len(v), 4) for k, v in tot.items()})
print("DONE")
