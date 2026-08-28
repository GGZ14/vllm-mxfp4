"""Gate the NumPy reference against the fp8 base, on both the K=2 and K=3 paths.

K=3 uses the generic dq8<bits,cb,align=4> reader rather than the 2-bit fast path, so it is a
different code path and needs its own evidence.
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

CASES = [
    "model.language_model.layers.0.mlp.gate_proj",      # K=2
    "model.language_model.layers.0.mlp.up_proj",        # K=3
    "model.language_model.layers.0.mlp.down_proj",      # K=3
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.31.mlp.gate_proj",
    "model.language_model.layers.31.mlp.down_proj",
]
H = R.hadamard(128) / np.sqrt(128.0)
print(f"{'projection':<40} {'K':>2} {'corr':>8} {'rel':>7} {'std ratio':>10}  verdict")
bad = 0
for base in CASES:
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    rin = get(base + ".escha_rin", np.float16).astype(np.float64)
    rout = get(base + ".escha_rout", np.float16).astype(np.float64)
    ref = load_fp32(index(FP8), base).astype(np.float64)
    W = np.zeros((128, 128))
    for i in range(8):
        for j in range(8):
            W[i*16:(i+1)*16, j*16:(j+1)*16] = R.decode_tile(code[i, j], K)
    W = ((H @ W) * rin[0:128, None]) @ H * rout[0:128][None, :]
    r = ref[0:128, 0:128].T
    c = np.corrcoef(W.ravel(), r.ravel())[0, 1]
    rel = np.linalg.norm(W - r) / np.linalg.norm(r)
    sr = W.std() / r.std()
    # a correct 2-3 bit decode: corr well above 0.8, std within a few %. A wrong one gives ~0.
    ok = c > 0.80 and 0.9 < sr < 1.1
    bad += (not ok)
    print(f"{base.split('layers.')[-1]:<40} {K:>2} {c:>+8.4f} {rel:>7.3f} {sr:>10.4f}  {'OK' if ok else 'FAIL'}")
print(f"\nreference gate: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
