"""Dump real escha tiles plus their reference decode, as the gate for the HIP decoder.

Real tiles, not synthetic ones: the trellis is a sliding window over a circular bitstream, so
synthetic words would not exercise the wrap the way the checkpoint's own data does.

Layout: [magic u32][n_k2 u32][n_k3 u32] then, per case, K u32, then 256*K/32 u32 of packed words,
then 256 fp32 of the expected tile in ROW-MAJOR order (row = K index, col = N index).
"""
import sys, json, struct
import numpy as np
sys.path.insert(0, "/home/brian/mxfp4_work/escha")
import exl3_ref as R

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

CASES = [("model.language_model.layers.0.mlp.gate_proj", 2),
         ("model.language_model.layers.0.mlp.up_proj", 3),
         ("model.language_model.layers.31.mlp.down_proj", 3),
         ("model.language_model.layers.31.self_attn.o_proj", 2)]
TILES = [(0, 0), (1, 0), (0, 1), (3, 5), (7, 7), (2, 9)]

out = [struct.pack("<I", 0xE5C7A000)]
n2 = n3 = 0
body = []
for base, _ in CASES:
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    for (i, j) in TILES:
        if i >= code.shape[0] or j >= code.shape[1]:
            continue
        words = np.ascontiguousarray(code[i, j])
        tile = R.decode_tile(words, K).astype(np.float32)      # [16,16] row-major
        body.append(struct.pack("<I", K))
        body.append(words.view(np.uint16).view(np.uint32).astype("<u4").tobytes())
        body.append(tile.astype("<f4").tobytes())
        n2 += (K == 2); n3 += (K == 3)
out.append(struct.pack("<II", n2, n3))
out.extend(body)
open("vectors.bin", "wb").write(b"".join(out))
print(f"wrote vectors.bin: {n2} K=2 tiles, {n3} K=3 tiles, {sum(len(b) for b in out)} bytes")
