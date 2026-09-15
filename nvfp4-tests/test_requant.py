"""Offline checks for radiance_nvfp4: (1) fast e2m1 encoder == argmin reference (codes + scales),
(2) dequant(quant(w)) error sanity, (3) NVFP4 unpack/dequant agrees with vLLM's reference."""
import sys, torch
sys.path.insert(0, "/patches")
import radiance_nvfp4 as rn
from quantize_dflash_mxfp4 import quantize_mxfp4 as ref_quant, dequantize_mxfp4 as ref_deq
dev = "cuda"
torch.manual_seed(0)
# (1) encoder equivalence in OCP mode (the reference implements only that rule)
for shape in [(256, 512), (1000, 1024), (64, 17408)]:
    w = torch.randn(*shape, device=dev) * torch.rand(shape[0], 1, device=dev) * 3
    # sprinkle exact ties and zeros
    w[0, :64] = 0; w[1, :32] = 0.25 * 2.0 ** torch.arange(-3, 29, device=dev).float()
    p1, s1, _ = rn.quant_mxfp4(w, "ocp"); p2, s2 = ref_quant(w)
    x = p1 ^ p2
    # only difference allowed: the reference keeps a sign bit on zero-magnitude codes (-0), we canonicalise it
    mag0 = ((p2 & 0x07) == 0) & ((x & 0x0F) != 0) | ((p2 & 0x70) == 0) & ((x & 0xF0) != 0)
    bad = int(((x & 0x77) != 0).sum()) + int(((x != 0) & ~mag0).sum())
    print(f"ocp {shape}: code-diff={int((x != 0).sum())} (non-negzero diffs={bad}) scale-diff={int((s1 != s2).sum())}")
    d1 = rn.dequant_mxfp4(p1, s1); d2 = ref_deq(p2, s2, shape[1])
    print(f"    dequant bit-diff={int((d1 != d2).sum())}")
# (2) mode comparison on gaussian rows
w = torch.randn(2048, 5120, device=dev) * 0.02
for m in ["ocp", "noclip", "mse"]:
    p, s, e = rn.quant_mxfp4(w, m); d = rn.dequant_mxfp4(p, s)
    rel = ((d - w) ** 2).sum().sqrt() / (w ** 2).sum().sqrt()
    print(f"gaussian {m}: relRMS {rel:.4f}  (sq err {float(e):.4g})")
# (3) NVFP4 dequant vs vLLM reference
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
n, k = 512, 1024
packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = (torch.rand(n, k // 16, device=dev) * 4 + 0.5).to(torch.float8_e4m3fn)
g = 137.5
mine = rn.dequant_nvfp4(packed, sc, g)
ref = dequantize_to_dtype(packed, sc, torch.tensor(1.0 / g, device=dev), torch.float32, 16, False)
print(f"nvfp4 dequant max|diff| = {float((mine - ref).abs().max()):.3g} (ref max {float(ref.abs().max()):.3g})")
print("DONE")
