"""int5-bitplane serving checkpoint from a paroquant optimize/finetune result dir (one .pt per module:
"{layer}.{name}.pt" with the trained weight, quantizer scale / zero_point_float, and the frozen rotation
tensors). Quantization is z-lab's own routine (paroquant.cli.convert._quantize_rotated_weight via
_quantize_layer); only the packing is ours: qweight/qzeros = AWQ packing of the low nibble, qweight_hi /
qzeros_hi = [K, N/32] / [G, N/32] int32 fifth-bit planes (see build_int5.py). Shards mirror the base.
Usage: convert_int5.py --model /models/<base> --result-dir /out/<base> --output-path /models/<out>"""
import argparse, json, shutil, sys, time, types
from pathlib import Path
import torch
sys.path.insert(0, "/src")
for name in ("awq", "awq.modules", "awq.modules.linear", "awq.modules.linear.gemm"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["awq.modules.linear.gemm"].WQLinearMMFunction = object
from paroquant.cli import convert as C
from safetensors import safe_open
from safetensors.torch import save_file

def _bitplane(v):                     # [R, C] int -> [R, C/32] int32, bit (c % 32) of word c // 32 = v >> 4
    hi = ((v.to(torch.int64) >> 4) & 1).view(v.shape[0], -1, 32)
    w = (hi << torch.arange(32, device=v.device, dtype=torch.int64)).sum(-1)
    return ((w + 2**31) % 2**32 - 2**31).to(torch.int32)

def _to_int5_buffers(quantized, scales_2d, zeros_2d):
    q = quantized.to(torch.int32); z = zeros_2d.to(torch.int32)
    assert int(q.max()) <= 31 and int(z.max()) <= 31, "int5 converter got codes above 31"
    return {"qweight": C._pack_awq((q & 15).T.contiguous()).cpu(), "qweight_hi": _bitplane(q.T.contiguous()).cpu(),
            "qzeros": C._pack_awq((z & 15).T.contiguous()).cpu(), "qzeros_hi": _bitplane(z.T.contiguous()).cpu(),
            "scales": scales_2d.T.contiguous().to(torch.float16).cpu()}
C._to_awq_buffers = _to_int5_buffers

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--result-dir", required=True)
    ap.add_argument("--output-path", required=True); a = ap.parse_args()
    base, res, out = Path(a.model), Path(a.result_dir), Path(a.output_path); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in res.iterdir() if p.suffix == ".pt" and p.name[0].isdigit())
    print(f"{len(files)} module results", flush=True); t0 = time.time()
    quant = {}                                                   # full module name -> buffers
    bits = group = krot = None
    for i, f in enumerate(files):
        layer, name = f.name.split(".", 1); name = name[:-3]
        sd = torch.load(f, weights_only=False, map_location="cuda")
        bufs, bits, group, krot = C._quantize_layer(sd, "cuda")
        quant[f"model.language_model.layers.{layer}.{name}"] = bufs
        del sd; torch.cuda.empty_cache()
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    idx = json.load(open(base / "model.safetensors.index.json"))["weight_map"]; wmap = {}
    for shard in sorted(set(idx.values())):
        tens = {}
        with safe_open(base / shard, framework="pt") as fh:
            for k in fh.keys():
                if k.startswith("mtp."): continue
                mod = k[:-len(".weight")] if k.endswith(".weight") else None
                if mod in quant:
                    for leaf, t in quant.pop(mod).items(): tens[f"{mod}.{leaf}"] = t
                else:
                    t = fh.get_tensor(k); tens[k] = t.to(torch.float16) if t.is_floating_point() else t
        save_file(tens, out / shard, metadata={"format": "pt"})
        for k in tens: wmap[k] = shard
        print(f"  wrote {shard}", flush=True)
    assert not quant, f"unplaced quantized modules: {list(quant)[:3]}"
    json.dump({"metadata": {}, "weight_map": wmap}, open(out / "model.safetensors.index.json", "w"), indent=1)
    paro = Path("/models/Qwen3.8-27B-PARO")
    for fn in paro.iterdir():
        if fn.suffix in (".json", ".jinja", ".txt") and fn.name != "model.safetensors.index.json": shutil.copy(fn, out / fn.name)
    cfg = json.load(open(paro / "config.json")); qc = dict(cfg["quantization_config"])
    qc.update({"quant_method": "paroquant", "bits": bits, "group_size": group, "krot": krot, "format": "int5-bitplane",
               "rotations": "z-lab/Qwen3.8-27B-PARO", "codes": "stage-2 finetune (weights + scales under the int5 grid)"})
    cfg["quantization_config"] = qc; json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"DONE {len(files)} modules -> {out} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
