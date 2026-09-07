#!/usr/bin/env python3
"""GPU-free self-test for radiance_tp3pad: run every padding rule against the REAL checkpoint
shapes (read from the safetensors headers, no tensor data) and assert the geometry tables in
TP3_PADDING_PLAN.md. Catches checkpoint / tensor-name drift before a serve does.

    ./tp3pad_selftest.py                       both checkpoints under $MODELS (~/models)
    ./tp3pad_selftest.py --torch               also pad one real tensor per rule and check fills
                                               (needs torch + safetensors: run it in the image)
    MODELS=/x ./tp3pad_selftest.py             another checkpoint directory
    SNAP=... DRAFTER=... ./tp3pad_selftest.py  explicit paths (same names serve-mxfp4.sh uses)

Exit status 0 = every table holds. Runs with torch absent (the shape planner is torch-free)."""
import glob
import json
import os
import re
import struct
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RADIANCE_TP_PAD", "3")
import radiance_tp3pad as tp  # noqa: E402

MODELS = os.path.expanduser(os.environ.get("MODELS", "~/models"))
SNAP = os.environ.get("SNAP") or os.path.join(MODELS, "Qwen3.8-27B-MXFP4-mtpfp8")
DRAFTER = os.environ.get("DRAFTER") or os.path.join(MODELS, "Qwen3.8-27B-DFlash2-FP8")
TP = tp.PAD

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")


def headers(path):
    """name -> (dtype, shape) across every *.safetensors in path."""
    out = OrderedDict()
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors under {path}")
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], v["shape"])
    return out


# Expected padded shapes per rule, in tensor units, straight from the plan's tables. A rule that
# pads a tensor not listed here, or to a different shape, is a failure.
TARGET_EXPECT = {
    # label: {(dtype, old_shape): new_shape}
    "q_proj": {("U8", (12288, 2560)): (18432, 2560), ("U8", (12288, 160)): (18432, 160),
               ("F8_E4M3", (12288, 5120)): (18432, 5120), ("F32", (12288,)): (18432,)},
    "k_proj": {("U8", (1024, 2560)): (1536, 2560), ("U8", (1024, 160)): (1536, 160),
               ("F8_E4M3", (1024, 5120)): (1536, 5120), ("F32", (1024,)): (1536,)},
    "v_proj": {("U8", (1024, 2560)): (1536, 2560), ("U8", (1024, 160)): (1536, 160),
               ("F8_E4M3", (1024, 5120)): (1536, 5120), ("F32", (1024,)): (1536,)},
    "o_proj": {("U8", (5120, 3072)): (5120, 4608), ("U8", (5120, 192)): (5120, 288),
               ("F8_E4M3", (5120, 6144)): (5120, 9216)},
    "in_proj_qkv": {("U8", (10240, 2560)): (11520, 2560), ("U8", (10240, 160)): (11520, 160)},
    "in_proj_z": {("U8", (6144, 2560)): (6912, 2560), ("U8", (6144, 160)): (6912, 160)},
    "in_proj_b": {("U8", (48, 2560)): (54, 2560), ("U8", (48, 160)): (54, 160)},
    "in_proj_a": {("U8", (48, 2560)): (54, 2560), ("U8", (48, 160)): (54, 160)},
    "out_proj": {("U8", (5120, 3072)): (5120, 3456), ("U8", (5120, 192)): (5120, 216)},
    "conv1d": {("BF16", (10240, 1, 4)): (11520, 1, 4)},
    "A_log": {("BF16", (48,)): (54,)},
    "dt_bias": {("BF16", (48,)): (54,)},
}


def target_mlp_expect(inter):
    return {
        "gate_proj": {("U8", (17408, 2560)): (inter, 2560), ("U8", (17408, 160)): (inter, 160),
                      ("F8_E4M3", (17408, 5120)): (inter, 5120), ("F32", (17408,)): (inter,)},
        "up_proj": {("U8", (17408, 2560)): (inter, 2560), ("U8", (17408, 160)): (inter, 160),
                    ("F8_E4M3", (17408, 5120)): (inter, 5120), ("F32", (17408,)): (inter,)},
        "down_proj": {("U8", (5120, 8704)): (5120, inter // 2), ("U8", (5120, 544)): (5120, inter // 32),
                      ("F8_E4M3", (5120, 17408)): (5120, inter)},
    }


DRAFTER_EXPECT = {
    "q_proj": {("F8_E4M3", (4096, 5120)): (6144, 5120), ("BF16", (32, 40)): (48, 40)},
    "k_proj": {("F8_E4M3", (1024, 5120)): (1536, 5120), ("BF16", (8, 40)): (12, 40)},
    "v_proj": {("F8_E4M3", (1024, 5120)): (1536, 5120), ("BF16", (8, 40)): (12, 40)},
    "o_proj": {("F8_E4M3", (5120, 4096)): (5120, 6144), ("BF16", (40, 32)): (40, 48)},
    "gate_proj": {("F8_E4M3", (17408, 5120)): (17664, 5120), ("BF16", (136, 40)): (138, 40)},
    "up_proj": {("F8_E4M3", (17408, 5120)): (17664, 5120), ("BF16", (136, 40)): (138, 40)},
    "down_proj": {("F8_E4M3", (5120, 17408)): (5120, 17664), ("BF16", (40, 136)): (40, 138)},
}

# Names the padder must NOT touch (a rule regex that over-matches shows up here).
UNTOUCHED_RE = re.compile(r"(q_norm|k_norm|layernorm|\.norm\.|embed_tokens|lm_head|mtp\.fc\.|"
                          r"pre_fc_norm|visual|candidate_selector|attention_conv|mlp_conv|"
                          r"hidden_norm|^fc\.|^norm\.)")


def run_spec(label, path, spec, expect, stock, padded, intermediate=None, tps=(1, 3)):
    """tps: the tensor-parallel sizes this geometry is meant to serve; every padded dim must split
    into whole units at each of them (TP=3 for the real thing, TP=2 for the heads-only gate)."""
    tp_div = max(tps)
    print(f"== {label}: {path}")
    cfg = json.load(open(os.path.join(path, "config.json")))
    text = cfg.get("text_config", cfg)
    for k, v in stock.items():
        check(text.get(k) == v, f"{label} config {k}={text.get(k)} != stock {v}")
    hdr = headers(path)
    tally = OrderedDict()
    counts = {}
    mtp = set()
    for name, (dt, shape) in hdr.items():
        m = re.match(r"^mtp\.layers\.(\d+)\.", name)
        if m:
            mtp.add(m.group(1))
        plan = tp.plan_shape(name, shape, spec, intermediate)
        if plan is None:
            continue
        lab, kind, new_shape, sections = plan
        check(not UNTOUCHED_RE.search(name), f"rule {lab} matched a tensor it must leave alone: {name}")
        exp = expect.get(lab, {}).get((dt, tuple(shape)))
        check(exp is not None and tuple(new_shape) == tuple(exp),
              f"{name} {dt} {shape} -> {new_shape}, table says {exp}")
        # Sharding: dim 0 pads (rows / sections) are ColumnParallel outputs, kcols are RowParallel
        # inputs; both must split into TP whole units of this tensor.
        d = 1 if kind == "kcols" else 0
        check(new_shape[d] % tp_div == 0, f"{name}: padded dim{d} {new_shape[d]} does not divide by {tp_div}")
        if kind == "sections":
            olds, news = sections
            check(all(n % tp_div == 0 for n in news), f"{name}: a section of {news} does not divide by {tp_div}")
        key = (lab, dt, tuple(shape), tuple(new_shape))
        tally[key] = tally.get(key, 0) + 1
        counts[lab] = counts.get(lab, 0) + 1
    for (lab, dt, old, new), n in tally.items():
        print(f"  {n:4d}x {lab:12s} {dt:8s} {list(old)} -> {list(new)}")

    class _Cfg:  # what _expected_counts reads
        layer_types = text.get("layer_types", [])
        num_hidden_layers = text.get("num_hidden_layers", 0)
        intermediate_size = padded["intermediate_size"]

    exp_counts = tp._expected_counts(spec, _Cfg, len(mtp))
    for lab, n in exp_counts.items():
        check(counts.get(lab, 0) == n, f"{label} {lab}: {counts.get(lab, 0)} tensors padded, expected {n}")
    check(set(counts) <= set(exp_counts), f"{label} unexpected labels {sorted(set(counts) - set(exp_counts))}")
    print(f"  {sum(counts.values())} tensors padded across {len(counts)} rules"
          f"{' (MTP layers ' + str(sorted(mtp)) + ')' if mtp else ''}")

    # Per-rank geometry the kernels see.
    if spec == "target":
        inter = intermediate or tp.padded_intermediate()
        for t in tps:
            check(not any(v % t for v in padded.values()), f"TP{t}: geometry {dict(padded)} does not divide")
            q, kv = padded["num_attention_heads"] // t, padded["num_key_value_heads"] // t
            hg, h = padded["linear_num_key_heads"] // t, padded["linear_num_value_heads"] // t
            k_o, k_out, k_down = q * 256, h * 128, inter // t
            print(f"  TP{t}: {q} q / {kv} kv (GQA {q // kv}), GDN H {h} / Hg {hg}, "
                  f"K o_proj {k_o} out_proj {k_out} down_proj {k_down}")
            check(q % kv == 0 and q // kv == 6, f"TP{t}: GQA {q}/{kv} is not 6 (R4D attention needs 6)")
            check(h % hg == 0 and h // hg == 3, f"TP{t}: GDN V/K ratio {h}/{hg} is not 3")
            for nm, kk in (("o_proj", k_o), ("out_proj", k_out), ("down_proj", k_down)):
                check(kk % 64 == 0, f"TP{t}: {nm} K={kk} fails the MXFP4 kernel's K % 64")
            ba = 2 * h
            print(f"       in_proj_ba per-rank N={ba} -> WPERM {'ok' if ba % 16 == 0 else 'must be 0'}, "
                  f"gdnmerge {'merges' if ba % 16 == 0 else 'skips'}")
    else:
        for t in tps:
            q, kv = padded["num_attention_heads"] // t, padded["num_key_value_heads"] // t
            print(f"  TP{t}: {q} q / {kv} kv (GQA {q // kv}), MLP {padded['intermediate_size'] // t}")
            # the drafter's KV bytes/token must equal the target's, or the two cache groups get
            # different block sizes and the sliding-window prefix-cache lookup asserts
            tk = tp.target_padded()["num_key_value_heads"] // t if tp.target_padded()["num_key_value_heads"] % t == 0 else None
            if tk is not None:
                check(kv * 128 == tk * 256, f"drafter TP{t}: {kv} kv x 128 != target {tk} kv x 256 (cache group block sizes differ)")
            # GQA is baked into the trained weights (q head h -> kv head h // GQA): it must not change
            check(q // kv == tp.DRAFTER_STOCK["num_attention_heads"] // tp.DRAFTER_STOCK["num_key_value_heads"],
                  f"drafter TP{t}: GQA {q // kv} != stock 4 -- real q heads would read the wrong kv head")
            check((q * 128) % 128 == 0 and (padded["intermediate_size"] // t) % 128 == 0,
                  f"drafter TP{t}: a per-rank shard is not whole 128-blocks")
    return hdr


def run_vocab():
    print("== vocab")
    mult = tp.vocab_pad_multiple(64)
    check(mult == 192, f"vocab_pad_multiple(64)={mult}, expected 192")
    check(tp.vocab_pad_multiple(256) == 768, "lcm with LoRA's 256 should be 768")
    padded = (248320 + mult - 1) // mult * mult
    check(padded == 248448 and padded % TP == 0, f"padded vocab {padded}")
    print(f"  248320 -> {padded}, {padded // TP} per rank at TP{TP}; "
          f"rank {TP - 1} holds {248320 - (TP - 1) * (padded // TP)} real + "
          f"{padded - 248320} zero rows")
    os.environ["RADIANCE_TP_PAD"] = "0"
    check(tp.vocab_pad_multiple(64) == 64, "vocab multiple must be untouched with the env off")
    os.environ["RADIANCE_TP_PAD"] = "3"


def run_torch(path, spec, intermediate=None):
    """Pad one real tensor per rule and check the real region is unchanged and the pad is the
    documented fill."""
    import torch
    from safetensors import safe_open
    print(f"== torch fills: {path}")
    done = set()
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt", device="cpu") as st:
            for name in st.keys():
                sl = st.get_slice(name)
                plan = tp.plan_shape(name, sl.get_shape(), spec, intermediate)
                if plan is None:
                    continue
                key = (plan[0], sl.get_dtype(), name.rsplit(".", 1)[-1])   # weight AND its scale
                if key in done:
                    continue
                done.add(key)
                t = st.get_tensor(name)
                out = tp.pad_tensor(name, t, plan)
                lab, kind, new_shape, sections = plan
                check(list(out.shape) == list(new_shape), f"{name}: got {list(out.shape)} vs {new_shape}")
                fill = tp.fill_value(name, str(t.dtype), kind)
                if t.dtype == torch.uint8 and name.endswith("weight_scale"):
                    # e8m0 pads: 2^0 on dummy rows, 2^-127 on K columns of real rows (never the row max)
                    check(fill == (0x00 if kind == "kcols" else 0x7F), f"{name}: e8m0 pad {fill:#x} for {kind}")
                    if kind == "kcols":
                        # the fold reference is the per-row max: the pad must not move it on any row
                        same_max = torch.equal(out.max(dim=1).values, t.max(dim=1).values)
                        check(same_max, f"{name}: K pad raised a row's max exponent (fold reference)")
                tb = t.view(torch.uint8) if "float8" in str(t.dtype) else t
                ob = out.view(torch.uint8) if "float8" in str(t.dtype) else out
                if kind == "rows":
                    same = torch.equal(ob[: t.shape[0]], tb)
                    pad = ob[t.shape[0]:]
                elif kind == "kcols":
                    same = torch.equal(ob[:, : t.shape[1]], tb)
                    pad = ob[:, t.shape[1]:]
                else:
                    olds, news = sections
                    same, s, d, pads = True, 0, 0, []
                    for o, n in zip(olds, news):
                        same &= torch.equal(ob.narrow(0, d, o), tb.narrow(0, s, o))
                        pads.append(ob.narrow(0, d + o, n - o))
                        s += o
                        d += n
                    pad = torch.cat(pads)
                check(same, f"{name}: real region changed")
                if "float8" in str(t.dtype):
                    ok = bool((pad == 0).all())
                else:
                    ok = bool((pad.float() == float(fill)).all())
                check(ok, f"{name}: pad region is not {fill}")
                print(f"  ok {lab:12s} {str(t.dtype):22s} {list(t.shape)} -> {list(out.shape)} fill={fill}")


def main():
    torch_mode = "--torch" in sys.argv
    inter_env = int(os.environ.get("RADIANCE_TP_PAD_INTERMEDIATE", "") or 17472)
    exp = dict(TARGET_EXPECT)
    exp.update(target_mlp_expect(inter_env))
    run_spec("target", SNAP, "target", exp, tp.TARGET_STOCK, tp.target_padded(inter_env), inter_env,
             tps=(1, 3) if inter_env % 3 == 0 else (1, 2))
    if inter_env != 17408:
        # The TP=2 heads-only gate leaves the MLP stock (17472/2 = 8736 fails K % 64; 8704 passes);
        # make sure that variant plans too, and that it splits over 2 ranks.
        exp2 = dict(TARGET_EXPECT)
        os.environ["RADIANCE_TP_PAD_INTERMEDIATE"] = "17408"
        run_spec("target (RADIANCE_TP_PAD_INTERMEDIATE=17408, TP=2 heads-only gate)", SNAP, "target",
                 exp2, tp.TARGET_STOCK, tp.target_padded(17408), 17408, tps=(1, 2))
        os.environ.pop("RADIANCE_TP_PAD_INTERMEDIATE")
    if os.path.isfile(os.path.join(DRAFTER, "config.json")):
        run_spec("drafter", DRAFTER, "drafter", DRAFTER_EXPECT, tp.DRAFTER_STOCK, tp.drafter_padded())
    else:
        print(f"== drafter: {DRAFTER} not present, skipped")
    run_vocab()
    if torch_mode:
        run_torch(SNAP, "target", inter_env)
        if os.path.isfile(os.path.join(DRAFTER, "config.json")):
            run_torch(DRAFTER, "drafter")
    if FAILS:
        print(f"\n{len(FAILS)} FAILURE(S)")
        sys.exit(1)
    print("\nall tables hold")


if __name__ == "__main__":
    main()
