"""Does z-lab's stored rotation reproduce the optimizer's rotate(W_bf16 * cs) convention?

Dequantize z-lab's int4 for one module (that is the ROTATED, channel-scaled weight) and compare
it to rotating the bf16 base weight with their pairs/theta/channel_scales through the same
kernel the optimizer uses. Agreement to ~int4 noise confirms direction, sign, and the
pre-inverted channel_scales convention all at once.
"""
import json, sys, torch
from safetensors import safe_open
sys.path.insert(0, "/src")
from paroquant.kernels.cuda import scaled_pairwise_rotation

name = "model.language_model.layers.0.mlp.down_proj"
ORDER = (0, 2, 4, 6, 1, 3, 5, 7)

with safe_open("/models/Qwen3.8-27B-PARO/model.safetensors", framework="pt") as f:
    qw, qz, sc = (f.get_tensor(f"{name}.{k}") for k in ("qweight", "qzeros", "scales"))
    pairs, theta, cs = (f.get_tensor(f"{name}.{k}") for k in ("pairs", "theta", "channel_scales"))
idx = json.load(open("/models/Qwen3.8-27B-bf16/model.safetensors.index.json"))["weight_map"]
with safe_open("/models/Qwen3.8-27B-bf16/" + idx[f"{name}.weight"], framework="pt") as f:
    w = f.get_tensor(f"{name}.weight")           # [N, K] bf16

K, N8 = qw.shape; N = N8 * 8
def unpack(p):                                    # AWQ int32 [rows, N/8] -> [rows, N] codes
    out = torch.empty(p.shape[0], N, dtype=torch.int32)
    for j, col in enumerate(ORDER):
        out[:, col::8] = (p >> (4 * j)) & 0xF
    return out
codes = unpack(qw).float()                        # [K, N]
zeros = unpack(qz).float().repeat_interleave(128, dim=0)   # [K, N]
scale = sc.float().repeat_interleave(128, dim=0)           # [K, N]
w_rot_dq = ((codes - zeros) * scale).T.contiguous().cuda() # [N, K]

cs_opt = (1.0 / cs.float()).cuda()                # stored pre-inverted -> optimizer's multiplier
w_ref = scaled_pairwise_rotation(w.float().cuda() * cs_opt, pairs.cuda(), theta.float().cuda(), None, 128)

rel = ((w_rot_dq - w_ref).norm() / w_ref.norm()).item()
# sanity anchors: what an int4-g128 re-quant of w_ref itself costs, and what a WRONG rotation looks like
def int4(x):
    xs = x.reshape(-1, 128); mn, mx = xs.amin(1, keepdim=True), xs.amax(1, keepdim=True)
    s = (mx - mn).clamp_min(1e-5) / 15; zp = torch.round(-mn / s)
    return ((torch.clamp(torch.round(xs / s) + zp, 0, 15) - zp) * s).reshape(x.shape)
rel_int4 = ((int4(w_ref) - w_ref).norm() / w_ref.norm()).item()
w_wrong = scaled_pairwise_rotation(w.float().cuda() * cs_opt, pairs.cuda(), -theta.float().cuda(), None, 128)
rel_wrong = ((w_rot_dq - w_wrong).norm() / w_ref.norm()).item()
print(f"{name}  N={N} K={K}")
print(f"  rel err  z-lab dequant vs rotate(W_bf16*cs)      : {rel:.4f}")
print(f"  rel err  int4-g128 re-quant of the same (expected): {rel_int4:.4f}")
print(f"  rel err  vs INVERSE rotation (what a mismatch is) : {rel_wrong:.4f}")
print("CONVENTION MATCHES" if rel < 2 * rel_int4 else "CONVENTION MISMATCH")
