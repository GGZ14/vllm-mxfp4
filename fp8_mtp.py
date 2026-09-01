#!/usr/bin/env python3
"""MXFP4 body + FP8 drafter, in one checkpoint.

Run this once against amd/Qwen3.8-27B-Quark-AWQ-MXFP4 to produce the checkpoint serve-mxfp4.sh
serves (./setup-mxfp4.sh drives it for you). It is not optional: AMD ships the MTP head bf16 but names it in neither `exclude` nor
`layer_quant_config`, so vLLM's quark config falls through to `global_quant_config` (mxfp4) for
`mtp.*`, builds a packed uint8 weight of half the input width, and dies loading the full-width
bf16 tensor into it --

    AssertionError: Attempted to load weight (torch.Size([5120, 10240]))
                    into parameter (torch.Size([5120, 5120]))

-- before any of the quality argument below comes into play. Writing an explicit
`layer_quant_config` for the eight MTP projections is what makes the head loadable at all.

MXFP4 on the drafter failed twice: plain RTN cost acceptance 2.5 -> 2.21, and AWQ calibration did
not rescue it (0-5% error improvement; the alpha search chose a=0.1-0.2, and a=0.0 for mtp.fc,
because MXFP4's per-32 E8M0 block exponent already does most of what per-channel scaling would).
The error is intrinsic to 4 bits, and for a drafter accuracy IS throughput.

FP8 trades a smaller bandwidth win for a much smaller error: e4m3 per-channel is ~2-3% relative
versus MXFP4's ~11.6%. The drafter is 34% of decode weight traffic at n=8, so fp8 removes ~17% of
total decode traffic instead of 25% -- but should actually hold acceptance.

vLLM's quark config supports this natively: `layer_quant_config` is matched with fnmatch, and
QuarkW8A8Fp8 wants weight fp8_e4m3 static per_channel + input_tensors fp8_e4m3 dynamic (so no
input_scale is stored). Explicit layer names are used rather than a `*q_proj` glob, which would
also match the body's 64 layers.
"""
import json, shutil, struct, sys, pathlib

# Checked before importing torch, so a bare invocation on a host without it still explains itself.
if len(sys.argv) != 3:
    sys.exit(f"usage: {pathlib.Path(sys.argv[0]).name} <src-checkpoint> <dst-checkpoint>\n"
             "  src  a Quark AWQ MXFP4 snapshot, e.g. the directory under\n"
             "       ~/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/\n"
             "  dst  where to write it, e.g. $MODELS/Qwen3.8-27B-MXFP4-mtpfp8\n"
             "\n"
             "Needs torch. If the host has none, run it inside the image -- see the README.")

import torch

MTP = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
       "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.k_proj",
       "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
       "mtp.layers.0.self_attn.v_proj"]
FP8_MAX = 448.0
TDT = {"BF16": torch.bfloat16, "U8": torch.uint8, "F32": torch.float32}
SDT = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16",
       torch.float8_e4m3fn: "F8_E4M3"}

def spec(dtype, dynamic, qscheme, ch_axis):
    return {"block_size": None, "ch_axis": ch_axis, "dtype": dtype, "enable_buffer_reuse": False,
            "group_size": None, "is_dynamic": dynamic, "is_scale_quant": False,
            "max_input_numel": 4194304, "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver" if qscheme == "per_channel"
                            else "PerTensorMinMaxObserver",
            "qscheme": qscheme, "round_method": "half_even", "scale_calculation_mode": None,
            "scale_format": None, "scale_type": "float", "symmetric": True}

FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
           "weight": spec("fp8_e4m3", False, "per_channel", 0),
           "input_tensors": spec("fp8_e4m3", True, "per_tensor", -1)}

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def load(p, hdr, base, name):
    m = hdr[name]; s, e = m["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + s); buf = bytearray(f.read(e - s))
    return torch.frombuffer(buf, dtype=TDT[m["dtype"]]).reshape(m["shape"])

def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()

def main(src_dir, dst_dir):
    src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    sf = src / "model.safetensors"
    hdr, base = read_header(sf)

    new = {}
    print(f"{'tensor':34s} {'shape':>18} {'rel err':>9}   (MXFP4 was ~0.116)")
    for name in MTP:
        w = load(sf, hdr, base, name + ".weight").float()
        amax = w.abs().amax(dim=1).clamp(min=1e-12)          # per output channel
        s = (amax / FP8_MAX).float()
        q = (w / s.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        rel = ((q.float() * s.unsqueeze(1) - w).norm() / w.norm()).item()
        print(f"{name:34s} {str(tuple(w.shape)):>18} {rel:9.4f}")
        new[name + ".weight"] = q
        new[name + ".weight_scale"] = s

    names = [k for k in hdr if k != "__metadata__"]
    out_hdr, off, plan = {}, 0, []
    for k in names:
        if k in new:
            t = new[k]; nb = t.numel() * t.element_size()
            out_hdr[k] = {"dtype": SDT[t.dtype], "shape": list(t.shape),
                          "data_offsets": [off, off + nb]}
            plan.append(("new", k, nb)); off += nb
        else:
            m = hdr[k]; nb = m["data_offsets"][1] - m["data_offsets"][0]
            out_hdr[k] = {"dtype": m["dtype"], "shape": m["shape"],
                          "data_offsets": [off, off + nb]}
            plan.append(("copy", k, nb)); off += nb
    for k, t in new.items():
        if k not in out_hdr:
            nb = t.numel() * t.element_size()
            out_hdr[k] = {"dtype": SDT[t.dtype], "shape": list(t.shape),
                          "data_offsets": [off, off + nb]}
            plan.append(("new", k, nb)); off += nb
    out_hdr["__metadata__"] = {"format": "pt"}
    blob = json.dumps(out_hdr).encode()
    blob += b" " * ((8 - (len(blob) % 8)) % 8)
    outf = dst / "model.safetensors"
    with open(sf, "rb") as fin, open(outf, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob))); fout.write(blob)
        for kind, k, nb in plan:
            if kind == "new":
                fout.write(raw(new[k]))
            else:
                s0, e0 = hdr[k]["data_offsets"]
                fin.seek(base + s0); left = e0 - s0
                while left:
                    c = fin.read(min(left, 32 << 20)); fout.write(c); left -= len(c)
    print(f"\nwrote {outf} ({outf.stat().st_size / 2**30:.2f} GiB)")

    cfg = json.loads((src / "config.json").read_text())
    qc = cfg["quantization_config"]
    qc["exclude"] = [e for e in qc["exclude"] if e not in MTP]
    qc["layer_quant_config"] = {name: FP8_CFG for name in MTP}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"exclude {len(qc['exclude'])} entries; layer_quant_config {len(qc['layer_quant_config'])} fp8 layers")
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json", "chat_template.jinja"):
        if (src / n).exists():
            shutil.copy(src / n, dst / n)

main(sys.argv[1], sys.argv[2])
