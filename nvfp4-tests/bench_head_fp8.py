"""Time the int2 draft head path with an FP8 per-channel lm_head vs a bf16 one (same shape as the
27B head per rank: 62080 x 5120), and the rerank kernels alone."""
import sys, time, types, torch
sys.path.insert(0, "/patches")
import radiance_drafthead as dh
dev = "cuda"; torch.manual_seed(0)
N, K = 62080, 5120
w_bf = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
sc = (w_bf.float().abs().amax(1, keepdim=True) / 448.0)
w8 = (w_bf.float() / sc).clamp(-448, 448).to(torch.float8_e4m3fn)
class Head(torch.nn.Module): pass
h8 = Head(); h8.weight = torch.nn.Parameter(w8.t(), requires_grad=False); h8.weight_scale = torch.nn.Parameter(sc, requires_grad=False)
hb = Head(); hb.weight = torch.nn.Parameter(w_bf, requires_grad=False)
class LP:
    head_dtype = None
def timeit(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters * 1000
for name, head in [("fp8", h8), ("bf16", hb)]:
    dh._HEAD_CACHE.clear(); lp = LP(); print(name, dh._quantize_head_now(lp, head)[:60])
    for M in (8, 16, 64):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ms = timeit(lambda: lp._apply_head(head, x, None))
        print(f"  {name} apply_head M={M}: {ms:.3f} ms")
    # rerank kernel alone
    M = 16; x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    idx = torch.randint(0, N, (M, dh.RERANK), device=dev, dtype=torch.int32); ex = torch.empty(M, dh.RERANK, device=dev)
    rows, s = dh._head_matrix(head)
    if s is not None:
        f = lambda: dh._rerank_exact[(M, dh.RERANK)](x, rows.view(torch.uint8), s, idx, ex, K, rows.stride(0), R=dh.RERANK, BLOCK_K=512, FP8=True, num_warps=4)
    else:
        f = lambda: dh._rerank_exact[(M, dh.RERANK)](x, rows, x, idx, ex, K, rows.stride(0), R=dh.RERANK, BLOCK_K=512, FP8=False, num_warps=4)
    print(f"  {name} rerank kernel M=16: {timeit(f):.3f} ms")
    f2 = lambda: dh._head_matrix(head)
    print(f"  {name} _head_matrix: {timeit(f2, 200):.4f} ms")
print("DONE")
