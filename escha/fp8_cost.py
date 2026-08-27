"""Does rounding the decoded trellis weights to e4m3 cost anything worth caring about?

Prefill is compute-bound and gfx1201's fp8 WMMA runs at 2x the f16 rate, so the fp8 pipe is worth
a clean 2x -- but only if e4m3's 3 mantissa bits do not meaningfully add to an error budget that
already contains 2-3 bit trellis quantisation. Measured against the fp8 base, not asserted.
"""
import sys, json, struct
import numpy as np
sys.path.insert(0, "/home/brian/mxfp4_work/escha"); sys.path.insert(0, "/home/brian/mxfp4_work")
import exl3_ref as R
from rot_probe import index, load_fp32, FP8

ESCHA = "/home/brian/models/Qwen3.8-27B-Escha-W2"
_h = {}
def get(nm, dt):
    idx = json.load(open(f"{ESCHA}/model.safetensors.index.json"))["weight_map"]
    p = f"{ESCHA}/{idx[nm]}"
    if p not in _h:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; _h[p] = (json.loads(f.read(n)), 8 + n)
    h, off = _h[p]; o = h[nm]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(off + o[0]); b = f.read(o[1] - o[0])
    return np.frombuffer(b, dt).reshape(h[nm]["shape"])

def e4m3_pairs():
    """Representable e4m3 values with their byte patterns, sorted by value."""
    out = {}
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 0xF and m == 7:      # NaN
            continue
        v = s * (m * 2.0**-9 if e == 0 else (1 + m / 8.0) * 2.0**(e - 7))
        # keep the even-mantissa pattern when two bytes share a value (+-0)
        if v not in out or (b & 1) < (out[v] & 1):
            out[v] = b
    vs = np.array(sorted(out))
    bs = np.array([out[v] for v in vs], dtype=np.uint8)
    return vs, bs


TBL, TBL_B = e4m3_pairs()


def to_e4m3(x, want_bytes=False):
    """Round to nearest e4m3, TIES TO EVEN.

    Round-half-down is the obvious implementation and it is wrong: gfx1201's
    cvt_pk_fp8_f32 rounds half to even, and the decoded trellis values -- sums of two fp16, so
    coarsely quantised already -- land on ties about 1% of the time. A continuous test ramp never
    sees this (ties are measure-zero there); real weights do. Caught by a one-hot GEMM, where every
    disagreement was exactly one e4m3 step.
    """
    x = np.asarray(x, dtype=np.float64)
    i = np.clip(np.searchsorted(TBL, x), 1, len(TBL) - 1)
    lo, hi = TBL[i - 1], TBL[i]
    dlo, dhi = np.abs(x - lo), np.abs(hi - x)
    pick_hi = dhi < dlo
    tie = dlo == dhi
    if tie.any():                                  # ties -> even mantissa
        pick_hi = np.where(tie, (TBL_B[i - 1] & 1) != 0, pick_hi)
    idx = np.where(pick_hi, i, i - 1)
    return (TBL[idx], TBL_B[idx]) if want_bytes else TBL[idx]


H = R.hadamard(128) / np.sqrt(128.0)
print(f"{'projection':<26} {'K':>2} {'rel(trellis)':>13} {'rel(+e4m3)':>11} {'added':>8}")
for base in ("model.language_model.layers.0.mlp.gate_proj",
             "model.language_model.layers.0.mlp.up_proj",
             "model.language_model.layers.31.mlp.gate_proj"):
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    rin = get(base + ".escha_rin", np.float16).astype(np.float64)
    rout = get(base + ".escha_rout", np.float16).astype(np.float64)
    ref = load_fp32(index(FP8), base).astype(np.float64)
    Wi = np.zeros((128, 128))
    for i in range(8):
        for j in range(8):
            Wi[i*16:(i+1)*16, j*16:(j+1)*16] = R.decode_tile(code[i, j], K)
    def chain(W):
        return ((H @ W) * rin[0:128, None]) @ H * rout[0:128][None, :]
    r = ref[0:128, 0:128].T
    a = chain(Wi)
    b = chain(to_e4m3(Wi))                     # quantise the DECODED weights, then transform
    ra = np.linalg.norm(a - r) / np.linalg.norm(r)
    rb = np.linalg.norm(b - r) / np.linalg.norm(r)
    print(f"{base.split('layers.')[-1]:<26} {K:>2} {ra:>13.4f} {rb:>11.4f} {100*(rb/ra-1):>7.2f}%")
print("\n  e4m3 on top of a 2-3 bit trellis: the two errors add in quadrature, so a ~3.6% RMS")
print("  rounding sits far inside a 0.22-0.45 quantisation error.")
