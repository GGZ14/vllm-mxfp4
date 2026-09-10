"""Write the int5-bitplane serving checkpoint from a paroquant optimize/finetune result dir.
Reuses paroquant.cli.convert --mode real end to end (rotations, codes, scales, zero points, shard
layout, config) and only swaps the AWQ buffer writer for the int5 layout that build_int5.py documents:
qweight = AWQ packing of code & 15, qweight_hi = fifth-bit plane [K, N/32], same pair for the zeros.
Usage (in the radiance image, /src mounted): python3 convert_int5.py --model /models/<base> \
  --result-dir /out/<base> --output-path /models/<out>"""
import json, sys
import torch
sys.path.insert(0, "/src")
from paroquant.cli import convert as C

def _bitplane(v):                     # [R, C] int -> [R, C/32] int32, bit (c % 32) of word c // 32 = v >> 4
    hi = ((v.to(torch.int64) >> 4) & 1).view(v.shape[0], -1, 32)
    w = (hi << torch.arange(32, device=v.device, dtype=torch.int64)).sum(-1)
    return ((w + 2**31) % 2**32 - 2**31).to(torch.int32)

def _to_int5_buffers(quantized, scales_2d, zeros_2d):
    q = quantized.to(torch.int32); z = zeros_2d.to(torch.int32)
    assert int(q.max()) <= 31 and int(z.max()) <= 31, "int5 converter got codes above 31"
    return {
        "qweight": C._pack_awq((q & 15).T.contiguous()).cpu(),
        "qweight_hi": _bitplane(q.T.contiguous()).cpu(),
        "qzeros": C._pack_awq((z & 15).T.contiguous()).cpu(),
        "qzeros_hi": _bitplane(z.T.contiguous()).cpu(),
        "scales": scales_2d.T.contiguous().to(torch.float16).cpu(),
    }

C._to_awq_buffers = _to_int5_buffers
if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--output-path") + 1]
    sys.argv = ["convert"] + args + (["--mode", "real"] if "--mode" not in args else [])
    C.main()
    cfg_p = f"{out}/config.json"; cfg = json.load(open(cfg_p))
    qc = cfg.get("quantization_config", {}); qc.update({"quant_method": "paroquant", "bits": 5, "format": "int5-bitplane"})
    cfg["quantization_config"] = qc; json.dump(cfg, open(cfg_p, "w"), indent=2)
    print("int5-bitplane checkpoint written:", out)
