"""radiance_drafthead with an FP8 per-channel lm_head: (1) the in-kernel e4m3 decode rerank equals the
bf16-dequantized reference bitwise-close, (2) _head_matrix undoes the compressed-tensors transposed view,
(3) the int2 packing from fp8 rows equals packing from the dequantized bf16 rows."""
import sys, types, torch
sys.path.insert(0, "/patches")
import radiance_drafthead as dh
dev = "cuda"; torch.manual_seed(1)
N, K = 4096, 5120
w_bf = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
sc = w_bf.float().abs().amax(1, keepdim=True) / 448.0                      # per-channel fp8 scale [N,1]
w8 = (w_bf.float() / sc).clamp(-448, 448).to(torch.float8_e4m3fn)          # [N,K] storage
class Head(torch.nn.Module):
    pass
h = Head(); h.weight = torch.nn.Parameter(w8.t(), requires_grad=False)      # CT exposes the transposed view
h.weight_scale = torch.nn.Parameter(sc, requires_grad=False)
rows, s = dh._head_matrix(h)
print("head_matrix:", tuple(rows.shape), rows.is_contiguous(), None if s is None else tuple(s.shape))
assert rows.shape == (N, K) and rows.is_contiguous() and s.shape == (N,)
w_deq = w8.float() * sc                                                      # exact dequant reference
# (1) rerank kernel
M, R = 16, 80
x = torch.randn(M, K, device=dev).to(torch.bfloat16)
idx = torch.randint(0, N, (M, R), device=dev, dtype=torch.int32)
out8 = torch.empty(M, R, dtype=torch.float32, device=dev)
dh._rerank_exact[(M, R)](x, rows.view(torch.uint8), s, idx, out8, K, rows.stride(0), R=R, BLOCK_K=512, FP8=True, num_warps=4)
ref = torch.einsum("mk,mrk->mr", x.float(), w_deq[idx.long()])
print("rerank fp8 vs dequant ref: max|diff| =", float((out8 - ref).abs().max()), " ref max", float(ref.abs().max()))
assert float((out8 - ref).abs().max()) < 1e-3 * float(ref.abs().max())
outb = torch.empty(M, R, dtype=torch.float32, device=dev)
wb = w_deq.to(torch.bfloat16)
dh._rerank_exact[(M, R)](x, wb, x, idx, outb, K, wb.stride(0), R=R, BLOCK_K=512, FP8=False, num_warps=4)
print("bf16 path still runs: max|diff| vs its ref =", float((outb - torch.einsum("mk,mrk->mr", x.float(), wb.float()[idx.long()])).abs().max()))
# (2)+(3) packing from fp8 rows == packing from dequantized rows
class LP: pass
lp8 = LP(); st8 = dh._quantize_head_now(lp8, h); print(st8)
hb = Head(); hb.weight = torch.nn.Parameter(w_deq.to(torch.bfloat16).contiguous(), requires_grad=False)
dh._HEAD_CACHE.clear()
lpb = LP(); stb = dh._quantize_head_now(lpb, hb); print(stb)
print("packed equal:", torch.equal(lp8._radiance_wq, lpb._radiance_wq), " scale equal:", torch.equal(lp8._radiance_scale, lpb._radiance_scale), " zs equal:", torch.equal(lp8._radiance_zs, lpb._radiance_zs))
print("empty check on real head:", dh._head_is_empty(rows, s), " on zeros:", dh._head_is_empty(torch.zeros(64, K, dtype=torch.float8_e4m3fn, device=dev), s[:64]))
print("DONE")
