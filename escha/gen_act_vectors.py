"""Reference vectors for the FULL escha linear layer: x -> rotate+quantize -> GEMM -> rotate.

This is the gate that matters for integration. The GEMM gates already prove the codec; what is
unproven is the chain around it, and that chain is where an integration silently produces
plausible-but-wrong output. The reference here follows the reference RUNTIME exactly
(escha/linear.py::_forward_runtime_had, sglang .../quantization/escha.py::_prefill_recon):

    y = Had128( (x * s_in) * rin ) @ decode(code)  ->  Had128  ->  * rout  ->  * s_out

and it models our own lossy steps too -- the e4m3 activation quantization and the e4m3 rounding of
decoded weights -- so a mismatch means a real bug rather than expected precision loss.

Layout: [magic][ncase] then per case [K][M][IC][OC] code words, x(bf16), s_in, rin, rout, s_out,
y_ref(fp32).
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

def bf16(a):
    """Round-to-nearest-even float32 -> bfloat16, returned as float32."""
    u = np.asarray(a, np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)

def fwht128(v):
    """Blockwise normalized Hadamard along the last axis; v[..., n], n % 128 == 0."""
    H = R.hadamard(128) / np.sqrt(128.0)
    s = v.shape
    return (v.reshape(*s[:-1], s[-1] // 128, 128) @ H).reshape(s)

# IC is never sliced (the in-axis Hadamard spans it); OC slices stay 128-aligned so the out-axis
# Hadamard blocks stay whole. M spans the decode band and a prefill-sized row count.
CASES = [("model.language_model.layers.0.mlp.gate_proj", 8, 256),
         ("model.language_model.layers.0.mlp.gate_proj", 5, 128),
         ("model.language_model.layers.0.mlp.up_proj", 64, 256),
         ("model.language_model.layers.31.mlp.gate_proj", 512, 128)]

rng = np.random.default_rng(7)
blobs = [struct.pack("<I", 0xE5C7A003), struct.pack("<I", len(CASES))]
for base, M, OC in CASES:
    K = int(get(base + ".escha_config", np.int32)[1])
    code = get(base + ".escha_code", np.int16)
    IC = code.shape[0] * 16
    nt = OC // 16
    words = np.ascontiguousarray(code[:, :nt]).view(np.uint16).view(np.uint32)
    rin   = get(base + ".escha_rin",  np.float16)[:IC]
    rout  = get(base + ".escha_rout", np.float16)[:OC]
    s_in  = get(base + ".escha_s_in",  np.float32)[:IC]
    s_out = get(base + ".escha_s_out", np.float32)[:OC]

    x = bf16(rng.standard_normal((M, IC), dtype=np.float32) * 0.05)

    # --- the reference chain, with our lossy steps modelled ---
    v = x.astype(np.float64) * s_in.astype(np.float64) * rin.astype(np.float64)
    h = fwht128(v)
    As = np.abs(h).max(axis=1) / 448.0
    As = np.where(As > 0, As, 1.0 / 448.0)
    A = to_e4m3(h / As[:, None])                       # what the GPU actually feeds the WMMA
    W = R.reconstruct_inner(words, K) if hasattr(R, "reconstruct_inner") else None
    if W is None:                                      # decode the bare code tiles, no rotations
        kt = words.shape[0]
        W = np.zeros((kt * 16, nt * 16), np.float64)
        for i in range(kt):
            for j in range(nt):
                W[i*16:(i+1)*16, j*16:(j+1)*16] = R.decode_tile(words[i, j], K)
    W8 = to_e4m3(W)                                    # our GEMMs round decoded weights to e4m3
    C = (A @ W8) * As[:, None]
    y = fwht128(bf16(C).astype(np.float64)) * rout.astype(np.float64) * s_out.astype(np.float64)

    print(f"  {base.split('layers.')[1]:<24} K={K} M={M} IC={IC} OC={OC}  |y|std={y.std():.5f}")
    blobs += [struct.pack("<4i", K, M, IC, OC),
              np.ascontiguousarray(words, np.uint32).tobytes(),
              np.ascontiguousarray(x.view(np.uint32) >> 16, np.uint32).astype(np.uint16).tobytes(),
              np.ascontiguousarray(s_in,  np.float32).tobytes(),
              np.ascontiguousarray(rin,   np.float16).tobytes(),
              np.ascontiguousarray(rout,  np.float16).tobytes(),
              np.ascontiguousarray(s_out, np.float32).tobytes(),
              np.ascontiguousarray(y, np.float32).tobytes()]
open("act_vectors.bin", "wb").write(b"".join(blobs))
print(f"wrote act_vectors.bin ({sum(len(b) for b in blobs)} bytes)")
