"""Fine-tune the DFlash2 drafter on captured target data (self-distillation, DFlash recipe).

Data: radiance_dflash_capture files -- per request: tokens, positions, aux hidden states (e4m3 +
per-token scale, [n, 5*5120]). Model: DFlash2Qwen3 re-implemented in plain torch from vLLM's
qwen3_dflash2.py / qwen3_dflash.py (same math: fc -> hidden_norm -> per-layer K/V injection with
k_norm + RoPE, queries = [anchor token, 7 masks] with q/k norms + RoPE, non-causal block, sliding
window 2048, two-tap grouped convs around attention and MLP, final norm, target lm_head).
Loss (paper): CE at the 7 mask positions, weights exp(-(k-1)/gamma), gamma = 4 for block 8.
Frozen: target embed_tokens / lm_head, candidate selector. Trainable: fc, hidden_norm, layers, norm
(MLPs optional). Optimizer: AdamW with fp32 master weights + states on the CPU (ZeRO-offload style)
so the 1.8B-parameter drafter trains on one 32 GB card; GPU holds bf16 params + grads.
Eval: weighted CE and per-position top-1 accuracy on held-out sequences (acceptance proxy).
"""
import argparse, glob, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("--capture", required=True); p.add_argument("--drafter", required=True)
p.add_argument("--target", required=True, help="bf16 target dir (embed_tokens / lm_head)")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=float, default=2.0); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--anchors", type=int, default=64, help="anchors per sequence per step")
p.add_argument("--seqs", type=int, default=4, help="sequences per step")
p.add_argument("--max-len", type=int, default=3072); p.add_argument("--min-ctx", type=int, default=32)
p.add_argument("--val", type=int, default=96); p.add_argument("--warmup", type=float, default=0.04)
p.add_argument("--gamma", type=float, default=4.0); p.add_argument("--freeze-mlp", action="store_true")
p.add_argument("--eval-only", action="store_true"); p.add_argument("--eval-every", type=int, default=150)
p.add_argument("--save-every", type=int, default=300); p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-files", type=int, default=0); p.add_argument("--device", default="cuda")
args = p.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)
dev = torch.device(args.device)
cfg = json.load(open(os.path.join(args.drafter, "config.json")))
dc = cfg["dflash_config"]; H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]; L = cfg["num_hidden_layers"]; I = cfg["intermediate_size"]; EPS = cfg["rms_norm_eps"]
BLOCK = int(dc["block_size"]); MASK_ID = int(dc["mask_token_id"]); TAPS = int(dc["conv_kernel_size"]); GS = int(dc["conv_group_size"])
NAUX = len(dc["target_layer_ids"]); THETA = float(cfg["rope_parameters"]["rope_theta"]); WINDOW = int(cfg.get("sliding_window") or 0)
EMB_SCALE = float(dc.get("input_embedding_scale", 1.0)); NQ = BLOCK; NDRAFT = BLOCK - 1
assert cfg.get("is_causal") is False, "trainer assumes the non-causal DFlash2 block"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ model (plain torch)
def rms(x, w, eps=EPS):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope_cos_sin(pos, dim=HD, theta=THETA):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    f = pos.float()[:, None] * inv[None, :]                       # [n, dim/2]
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):                                      # x [n, heads, HD], neox (rotate half)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


def grouped_conv(h, delta, base, block_size, groups, gs, taps):    # mirrors _grouped_conv
    blocks = h.unflatten(-1, (groups, gs))                         # [n, groups, gs]
    coef = base.view(1, taps, groups, gs) + delta.unsqueeze(-1)    # [n, taps, groups, gs]
    out = coef[:, 0] * blocks
    position = torch.arange(h.shape[0], device=h.device) & (block_size - 1)
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = H // GS
        self.base_kernel = nn.Parameter(torch.zeros(2, TAPS, H, dtype=torch.bfloat16))
        self.kernel_projection = nn.Linear(H, 2 * TAPS * self.groups, bias=False, dtype=torch.bfloat16)

    def prepare(self, h):
        coef = self.kernel_projection(h).reshape(h.shape[0], 2, TAPS, self.groups)
        return grouped_conv(h, coef[:, 0], self.base_kernel[0], NQ, self.groups, GS, TAPS), coef[:, 1]

    def finish(self, h, coef):
        return grouped_conv(h, coef, self.base_kernel[1], NQ, self.groups, GS, TAPS)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        bf = dict(dtype=torch.bfloat16)
        self.input_layernorm = nn.Parameter(torch.ones(H, **bf)); self.post_attention_layernorm = nn.Parameter(torch.ones(H, **bf))
        self.q_proj = nn.Linear(H, NH * HD, bias=False, **bf); self.k_proj = nn.Linear(H, NKV * HD, bias=False, **bf)
        self.v_proj = nn.Linear(H, NKV * HD, bias=False, **bf); self.o_proj = nn.Linear(NH * HD, H, bias=False, **bf)
        self.q_norm = nn.Parameter(torch.ones(HD, **bf)); self.k_norm = nn.Parameter(torch.ones(HD, **bf))
        self.gate_proj = nn.Linear(H, I, bias=False, **bf); self.up_proj = nn.Linear(H, I, bias=False, **bf)
        self.down_proj = nn.Linear(I, H, bias=False, **bf)
        self.attention_conv = Conv(); self.mlp_conv = Conv()

    def context_kv(self, ctx_normed, cos, sin):                    # ctx_normed: hidden_norm(fc(aux)) [n, H]
        k = self.k_proj(ctx_normed).view(-1, NKV, HD); v = self.v_proj(ctx_normed).view(-1, NKV, HD)
        k = apply_rope(rms(k, self.k_norm), cos, sin)
        return k, v

    def forward(self, h, residual, qpos_cos, qpos_sin, ctx_k, ctx_v, mask):
        # h: [Q, H] queries (block-aligned, NQ per anchor); ctx_k/v: [n, NKV, HD]; mask: [Q, n + Q] bool (True = attend)
        if residual is None:
            residual = h; h = rms(h, self.input_layernorm)
        else:
            residual = residual + h; h = rms(residual, self.input_layernorm)
        h, coef = self.attention_conv.prepare(h)
        q = self.q_proj(h).view(-1, NH, HD); k = self.k_proj(h).view(-1, NKV, HD); v = self.v_proj(h).view(-1, NKV, HD)
        q = apply_rope(rms(q, self.q_norm), qpos_cos, qpos_sin); k = apply_rope(rms(k, self.k_norm), qpos_cos, qpos_sin)
        K = torch.cat([ctx_k, k], 0); V = torch.cat([ctx_v, v], 0)                 # [n+Q, NKV, HD]
        rep = NH // NKV
        K = K.repeat_interleave(rep, dim=1); V = V.repeat_interleave(rep, dim=1)    # [n+Q, NH, HD]
        qh = q.transpose(0, 1)[None]; kh = K.transpose(0, 1)[None]; vh = V.transpose(0, 1)[None]  # [1, NH, *, HD]
        att = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None], scale=HD ** -0.5)
        h = self.o_proj(att[0].transpose(0, 1).reshape(-1, NH * HD))
        h = self.attention_conv.finish(h, coef)
        residual = residual + h; h = rms(residual, self.post_attention_layernorm)
        h, coef = self.mlp_conv.prepare(h)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        h = self.mlp_conv.finish(h, coef)
        return h, residual


class Drafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NAUX * H, H, bias=False, dtype=torch.bfloat16)
        self.hidden_norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16)); self.norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.layers = nn.ModuleList([Layer() for _ in range(L)])

    def forward(self, aux, ctx_pos, q_ids, q_pos, mask, embed):
        ctx = rms(self.fc(aux), self.hidden_norm)
        ccos, csin = rope_cos_sin(ctx_pos); qcos, qsin = rope_cos_sin(q_pos)
        h = embed[q_ids] * EMB_SCALE if EMB_SCALE != 1.0 else embed[q_ids]
        residual = None
        for lyr in self.layers:
            ck, cv = lyr.context_kv(ctx, ccos, csin)
            h, residual = lyr(h, residual, qcos, qsin, ck, cv, mask)
        return rms(residual + h, self.norm)


def load_drafter(m, path):
    """Load z-lab's bf16 drafter (or dequantize tcclaviger's FP8 block-128 one)."""
    sd = {}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith("weight_scale_inv"): continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                s = f.get_tensor(k.replace(".weight", ".weight_scale_inv")).float()   # [N/128, K/128]
                N, K = t.shape
                s = s.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
                t = (t.float() * s).to(torch.bfloat16)
            sd[k] = t
    own = m.state_dict(); loaded = 0
    for k, v in sd.items():
        kk = k.replace("mlp.", "").replace("self_attn.", "") if k.startswith("layers.") else k
        if kk.startswith("candidate_selector."): continue
        if kk.endswith("_layernorm.weight") or kk.endswith("_norm.weight") or kk in ("hidden_norm.weight", "norm.weight"):
            kk = kk[: -len(".weight")]
        if kk not in own:
            raise KeyError(f"unmapped drafter tensor {k} -> {kk}")
        assert own[kk].shape == v.shape, (k, own[kk].shape, v.shape)
        own[kk].copy_(v.to(own[kk].dtype)); loaded += 1
    log(f"drafter: {loaded} tensors loaded from {path}")
    return {k: v for k, v in sd.items() if k.startswith("candidate_selector.")}


def load_target_embed_head(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(os.path.join(path, idx[name]), "pt") as f:
            out[name] = f.get_tensor(name).to(torch.bfloat16)
    return out["model.language_model.embed_tokens.weight"], out["lm_head.weight"]


# ------------------------------------------------------------------ data
def load_seq(fn):
    d = torch.load(fn)
    pos = d["positions"].long(); n = pos.numel()
    if n < args.min_ctx + NQ + 1 or not bool((pos == torch.arange(n)).all()):
        return None
    n = min(n, args.max_len)
    return d["tokens"][:n].long(), d["hid_e4m3"][:n], d["scale"][:n]


def dequant(hid_u8, scale):
    return (hid_u8.to(dev).view(torch.float8_e4m3fn).float() * scale.to(dev).float()[:, None]).to(torch.bfloat16)


def make_batch(tokens, n, anchors):
    """Anchor a: context = positions 0..a (hidden states known), anchor token t[a+1] at a+1, masks at
    a+2..a+BLOCK; targets t[a+2..a+BLOCK]. Returns q_ids, q_pos, targets [A, NDRAFT], mask [A*NQ, n + A*NQ]."""
    A = anchors.numel()
    off = torch.arange(NQ, device=dev)
    q_pos = (anchors[:, None] + 1 + off[None, :])                                 # [A, NQ]
    q_ids = torch.full((A, NQ), MASK_ID, dtype=torch.long, device=dev)
    q_ids[:, 0] = tokens[anchors + 1]
    targets = tokens[(anchors[:, None] + 2 + off[None, :NDRAFT])]                 # [A, NDRAFT]
    ctx_pos = torch.arange(n, device=dev)
    qp = q_pos.reshape(-1)                                                          # [A*NQ]
    ctx_ok = ctx_pos[None, :] <= anchors.repeat_interleave(NQ)[:, None]            # context up to the anchor
    if WINDOW:
        ctx_ok &= (qp[:, None] - ctx_pos[None, :]) < WINDOW
    blk = torch.arange(A, device=dev).repeat_interleave(NQ)
    blk_ok = blk[:, None] == blk[None, :]                                          # own block, non-causal
    mask = torch.cat([ctx_ok, blk_ok], 1)
    return q_ids.reshape(-1), qp, targets, mask


WEIGHTS = torch.exp(-(torch.arange(NDRAFT, dtype=torch.float32)) / args.gamma).to(dev)  # k = 1..NDRAFT -> exp(-(k-1)/gamma)


def seq_loss(model, embed, head, tokens, aux, anchors, want_acc=False):
    n = aux.shape[0]
    q_ids, q_pos, targets, mask = make_batch(tokens, n, anchors)
    hs = model(aux, torch.arange(n, device=dev), q_ids, q_pos, mask, embed)      # [A*NQ, H]
    hs = hs.view(-1, NQ, H)[:, 1:, :]                                            # mask positions only
    logits = F.linear(hs.reshape(-1, H), head).float()                           # [A*NDRAFT, V]
    ce = F.cross_entropy(logits, targets.reshape(-1), reduction="none").view(-1, NDRAFT)
    loss = (ce * WEIGHTS[None, :]).sum(1).mean() / WEIGHTS.sum()
    acc = None
    if want_acc:
        acc = (logits.argmax(-1).view(-1, NDRAFT) == targets).float().mean(0)   # per position
    return loss, ce.mean(0).detach(), acc


@torch.no_grad()
def evaluate(model, embed, head, files, tag):
    model.eval(); tot = 0.0; cnt = 0; ce_pos = torch.zeros(NDRAFT, device=dev); acc_pos = torch.zeros(NDRAFT, device=dev)
    g = torch.Generator(device="cpu").manual_seed(1234)
    for fn in files:
        s = load_seq(fn)
        if s is None: continue
        tokens, hid, sc = s; n = tokens.numel(); tokens = tokens.to(dev)
        lo = min(args.min_ctx, n - NQ - 2); anchors = torch.randint(lo, n - NQ - 1, (min(args.anchors, 48),), generator=g).to(dev)
        loss, cep, acc = seq_loss(model, embed, head, tokens, dequant(hid, sc), anchors, want_acc=True)
        tot += loss.item(); ce_pos += cep; acc_pos += acc; cnt += 1
    model.train()
    acc_pos /= max(cnt, 1); ce_pos /= max(cnt, 1)
    # expected accepted tokens per block if acceptance were exactly position-wise top-1 hits with prefix semantics
    exp_acc = 0.0; prob = 1.0
    for k in range(NDRAFT): prob *= acc_pos[k].item(); exp_acc += prob
    log(f"[eval {tag}] weighted CE {tot / max(cnt, 1):.4f} | top1 by position " + " ".join(f"{a:.3f}" for a in acc_pos.tolist())
        + f" | prefix-expected accepted/block {exp_acc:.3f} | {cnt} seqs")
    return tot / max(cnt, 1), exp_acc


# ------------------------------------------------------------------ main
files = sorted(f for f in glob.glob(os.path.join(args.capture, "*.pt")) if os.path.getsize(f) > (args.min_ctx + 16) * 25700)  # drop warmup/short captures
if args.max_files: files = files[: args.max_files]
random.Random(args.seed).shuffle(files)
val_files, train_files = files[: args.val], files[args.val:]
log(f"{len(files)} capture files: {len(train_files)} train / {len(val_files)} val")
model = Drafter().to(dev)
selector_sd = load_drafter(model, args.drafter)
embed, head = load_target_embed_head(args.target); embed = embed.to(dev); head = head.to(dev)
log(f"target embed {tuple(embed.shape)} / head {tuple(head.shape)} on {dev}")
if args.freeze_mlp:
    for lyr in model.layers:
        for m in (lyr.gate_proj, lyr.up_proj, lyr.down_proj):
            for q in m.parameters(): q.requires_grad_(False)
trainable = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
log(f"trainable params: {sum(q.numel() for _, q in trainable) / 1e9:.2f} B in {len(trainable)} tensors")

evaluate(model, embed, head, val_files, "before")
if args.eval_only: sys.exit(0)

# CPU fp32 master + AdamW (offloaded optimizer); GPU keeps bf16 params/grads
def _pin(t): return t.pin_memory() if dev.type == "cuda" else t
master = [_pin(q.detach().float().cpu()).requires_grad_(True) for _, q in trainable]
opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
steps_per_epoch = math.ceil(len(train_files) / args.seqs); total = int(steps_per_epoch * args.epochs)
warm = max(1, int(total * args.warmup))
sched = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log(f"training: {total} steps ({args.seqs} seqs x {args.anchors} anchors each), lr {args.lr}, warmup {warm}")
step = 0; t0 = time.time(); ema = None
order = []
while len(order) < total * args.seqs:
    ep = train_files[:]; random.shuffle(ep); order += ep
os.makedirs(args.out, exist_ok=True)


def save(tag):
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("layers."):
            parts = k.split(".")
            name = parts[2]
            if name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                parts.insert(2, "self_attn")
            elif name in ("gate_proj", "up_proj", "down_proj"):
                parts.insert(2, "mlp")
            k = ".".join(parts)
        if k.endswith("input_layernorm") or k.endswith("post_attention_layernorm") or k.endswith("q_norm") or k.endswith("k_norm") or k in ("hidden_norm", "norm"):
            k = k + ".weight"
        sd[k] = v.detach().to(torch.bfloat16).cpu().contiguous()
    sd.update({k: v.cpu().contiguous() for k, v in selector_sd.items()})
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log(f"saved {len(sd)} tensors -> {args.out} ({tag})")


for step in range(total):
    lr = args.lr * sched(step)
    for g in opt.param_groups: g["lr"] = lr
    model.zero_grad(set_to_none=True)
    batch = order[step * args.seqs:(step + 1) * args.seqs]; used = 0; loss_acc = 0.0
    for fn in batch:
        s = load_seq(fn)
        if s is None: continue
        tokens, hid, sc = s; n = tokens.numel(); tokens = tokens.to(dev)
        lo = min(args.min_ctx, n - NQ - 2)
        anchors = torch.randint(lo, n - NQ - 1, (args.anchors,), device=dev)
        loss, _, _ = seq_loss(model, embed, head, tokens, dequant(hid, sc), anchors)
        (loss / len(batch)).backward(); loss_acc += loss.item(); used += 1
    if used == 0: continue
    # grads -> CPU master, clip, step, params back
    gn2 = 0.0
    for (n_, q), m in zip(trainable, master):
        g = q.grad.detach().float().cpu() if q.grad is not None else torch.zeros_like(m)
        m.grad = g; gn2 += float(g.pow(2).sum())
    gn = math.sqrt(gn2); clip = min(1.0, 1.0 / (gn + 1e-6))
    if clip < 1.0:
        for m in master: m.grad.mul_(clip)
    opt.step()
    with torch.no_grad():
        for (n_, q), m in zip(trainable, master): q.copy_(m.to(torch.bfloat16), non_blocking=True)
    l = loss_acc / used; ema = l if ema is None else 0.95 * ema + 0.05 * l
    if step % 10 == 0 or step == total - 1:
        el = time.time() - t0
        log(f"step {step + 1}/{total} loss {l:.4f} ema {ema:.4f} gnorm {gn:.2f} lr {lr:.2e} | {el / (step + 1):.1f} s/step, eta {(total - step - 1) * el / (step + 1) / 60:.0f} min")
    if (step + 1) % args.eval_every == 0 and step + 1 < total:
        evaluate(model, embed, head, val_files, f"step {step + 1}")
    if (step + 1) % args.save_every == 0 and step + 1 < total:
        save(f"step {step + 1}")
evaluate(model, embed, head, val_files, "after")
save("final")
