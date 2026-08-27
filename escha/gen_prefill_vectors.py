"""Reference vectors for the fp8 prefill GEMM.

The reference mirrors what the kernel actually computes: activations quantised to e4m3 per token,
decoded weights rounded to e4m3, accumulate in float64, scale by the per-token scale. Comparing
against an un-quantised reference would just re-measure e4m3, which fp8_cost.py already priced.
"""
import sys, json, struct
import numpy as np
sys.path.insert(0, "/home/brian/mxfp4_work/escha")
import exl3_ref as R
from fp8_cost import to_e4m3

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

def e4m3_bytes(x):
    """Round to e4m3 and return (raw bytes, quantised values), ties to even -- see fp8_cost."""
    q, b = to_e4m3(x, want_bytes=True)
    return b, q


CASES = [("model.language_model.layers.0.mlp.gate_proj", 256, 128, 256),
         ("model.language_model.layers.0.mlp.up_proj", 512, 128, 128),
         ("model.language_model.layers.0.mlp.gate_proj", 300, 256, 128),
         ("model.language_model.layers.31.mlp.gate_proj", 1024, 256, 256),
         # Decode-shaped rows for the fp8 decode kernel, which shares this reference: M values
         # spanning the real speculative band (5 = SPEC+1 single stream, 8/40/64 batched), with
         # Kdim=128 chosen so a KB=4 block covers only half the k-range and the ragged tail runs.
         ("model.language_model.layers.0.mlp.gate_proj", 5, 128, 256),
         ("model.language_model.layers.0.mlp.up_proj", 8, 128, 128),
         ("model.language_model.layers.0.mlp.gate_proj", 40, 256, 128),
         ("model.language_model.layers.31.mlp.gate_proj", 64, 256, 256)]
rng = np.random.default_rng(1)
blobs = [struct.pack("<I", 0xE5C7A002), struct.pack("<I", len(CASES))]
for base, M, N, Kdim in CASES:
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    kt, nt = Kdim // 16, N // 16
    words = np.ascontiguousarray(code[:kt, :nt]).view(np.uint16).view(np.uint32)
    W = np.zeros((Kdim, N))
    for i in range(kt):
        for j in range(nt):
            W[i*16:(i+1)*16, j*16:(j+1)*16] = R.decode_tile(code[i, j], K)
    _, Wq = e4m3_bytes(W)                                  # kernel converts decoded -> e4m3
    Ax = rng.standard_normal((M, Kdim)) * 0.3
    As = np.abs(Ax).max(1) / 448.0
    Ab, Aq = e4m3_bytes(Ax / As[:, None])
    C = (Aq @ Wq) * As[:, None]
    blobs.append(struct.pack("<4i", K, M, N, Kdim))
    blobs.append(words.astype("<u4").tobytes())
    blobs.append(np.ascontiguousarray(Ab).tobytes())
    blobs.append(As.astype("<f4").tobytes())
    blobs.append(C.astype("<f4").tobytes())
    print(f"  {base.split('layers.')[-1]:<18} K={K} M={M} N={N} Kdim={Kdim}  |C|std={C.std():.4f}")
open("prefill_vectors.bin", "wb").write(b"".join(blobs))
print(f"wrote prefill_vectors.bin ({sum(len(b) for b in blobs)} bytes)")
