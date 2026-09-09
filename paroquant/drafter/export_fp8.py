"""Quantize a bf16 DFlash2 drafter to FP8 in tcclaviger's layout: e4m3 weights with 128x128
block-wise `weight_scale_inv` (bf16) on the attention/MLP projections and `fc`; norms, conv
kernels, candidate selector and codebooks stay bf16. Copies config.json (quantization_config
from the reference FP8 drafter) so vLLM loads it exactly like the original."""
import json, os, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
ref_cfg = json.load(open(os.path.join(ref, "config.json")))
skip = set(ref_cfg["quantization_config"]["modules_to_not_convert"])
B = ref_cfg["quantization_config"]["weight_block_size"][0]
sd = {}
with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        mod = k[: -len(".weight")] if k.endswith(".weight") else k
        if not k.endswith(".weight") or t.dim() != 2 or mod in skip or mod.startswith("candidate_selector"):
            sd[k] = t.to(torch.bfloat16).contiguous(); continue
        w = t.float(); N, K = w.shape
        nb, kb = (N + B - 1) // B, (K + B - 1) // B
        wp = torch.zeros(nb * B, kb * B); wp[:N, :K] = w
        amax = wp.view(nb, B, kb, B).abs().amax(dim=(1, 3)).clamp_min(1e-12)          # [nb, kb]
        scale = (amax / 448.0)
        q = (wp.view(nb, B, kb, B) / scale[:, None, :, None]).view(nb * B, kb * B)[:N, :K]
        sd[k] = q.to(torch.float8_e4m3fn).contiguous()
        sd[mod + ".weight_scale_inv"] = scale.to(torch.bfloat16).contiguous()
os.makedirs(out, exist_ok=True)
save_file(sd, os.path.join(out, "model.safetensors"))
cfg = json.load(open(os.path.join(src, "config.json"))); cfg["quantization_config"] = ref_cfg["quantization_config"]
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for extra in ("README.md",):
    if os.path.exists(os.path.join(ref, extra)): shutil.copy(os.path.join(ref, extra), out)
n8 = sum(1 for v in sd.values() if v.dtype == torch.float8_e4m3fn)
print(f"exported {len(sd)} tensors ({n8} fp8) -> {out}")
