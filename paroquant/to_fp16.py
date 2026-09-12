"""bf16 safetensors -> fp16 safetensors, shard by shard on the CPU. The optimizer loads the base with
dtype=float16; from a bf16 checkpoint that conversion materializes the whole model (55 GB) in anonymous
RAM, while an fp16 checkpoint of matching dtype stays memory-mapped (page cache, reclaimable per layer)."""
import json, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
idx = json.load(open(src / "model.safetensors.index.json")); t0 = time.time()
for i, shard in enumerate(sorted(set(idx["weight_map"].values()))):
    if (dst / shard).exists(): continue
    out = {}
    with safe_open(src / shard, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k); out[k] = t.to(torch.float16) if t.dtype == torch.bfloat16 else t
    save_file(out, dst / shard, metadata={"format": "pt"}); del out
    print(f"  {i+1} {shard} {time.time()-t0:.0f}s", flush=True)
json.dump(idx, open(dst / "model.safetensors.index.json", "w"), indent=1)
for fn in src.iterdir():
    if fn.suffix in (".json", ".jinja", ".txt", ".py") and fn.name != "model.safetensors.index.json":
        shutil.copy(fn, dst / fn.name)
cfg = json.load(open(dst / "config.json")); cfg["torch_dtype"] = "float16"; json.dump(cfg, open(dst / "config.json", "w"), indent=2)
print("DONE", flush=True)
