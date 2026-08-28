"""Reference vectors for the decode GEMM: real escha weights, random fp16 activations, exact C.

Uses REAL code tiles so the trellis wrap and both bit rates are exercised. The reference C is
computed in float64 from the reference decode, so the harness compares the kernel against the
verified codec rather than against another copy of itself.

Layout: [magic][K][M][N][Kdim] then code words, then A (fp16), then C_ref (fp32).
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

CASES = [("model.language_model.layers.0.mlp.gate_proj", 8, 128, 256),
         ("model.language_model.layers.0.mlp.gate_proj", 5, 256, 512),
         ("model.language_model.layers.0.mlp.up_proj", 8, 128, 256),
         ("model.language_model.layers.0.mlp.up_proj", 40, 128, 128),
         ("model.language_model.layers.31.mlp.gate_proj", 64, 256, 256)]

rng = np.random.default_rng(0)
blobs = [struct.pack("<I", 0xE5C7A001), struct.pack("<I", len(CASES))]
for base, M, N, Kdim in CASES:
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    kt, nt = Kdim // 16, N // 16
    words = np.ascontiguousarray(code[:kt, :nt]).view(np.uint16).view(np.uint32)
    # decode to W[Kdim, N] using the verified reference
    W = np.zeros((Kdim, N), dtype=np.float64)
    for i in range(kt):
        for j in range(nt):
            W[i*16:(i+1)*16, j*16:(j+1)*16] = R.decode_tile(code[i, j], K)
    A = (rng.standard_normal((M, Kdim)) * 0.3).astype(np.float16)
    C = A.astype(np.float64) @ W
    blobs.append(struct.pack("<4i", K, M, N, Kdim))
    blobs.append(words.astype("<u4").tobytes())
    blobs.append(A.astype("<f2").tobytes())
    blobs.append(C.astype("<f4").tobytes())
    print(f"  {base.split('layers.')[-1]:<20} K={K} M={M} N={N} Kdim={Kdim}  |C|std={C.std():.4f}")
open("gemm_vectors.bin", "wb").write(b"".join(blobs))
print(f"wrote gemm_vectors.bin ({sum(len(b) for b in blobs)} bytes)")
