"""GPU unit test for radiance_paroquant_mxfp4: load -> prep -> linear vs an fp32 reference.

Runs inside the radiance image (needs vllm's parameter classes and both kernel .so's), no engine.
Builds a fake merged linear with P distinct z-lab-style rotations, feeds the real on-disk buffers
of a real module from the hybrid checkpoint, and checks the served output against a
straightforward fp32 dequant of the same codes: rotate+scale x, quantize to e4m3 per token,
matmul against e2m1*2^(E-127). Tolerance is the e4m3 activation quantization itself (~2-3e-2
relative), the same class the int4 module's CHECKALL reports against exact activations.

    podman run --rm --privileged --ipc=host --device /dev/kfd --device /dev/dri \\
      --group-add keep-groups -e HIP_VISIBLE_DEVICES=0 -e RADIANCE_PAROQUANT=1 \\
      -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_MXFP4_DECODE_MAX_M=64 \\
      -v ~/deadcode-vllm:/patches -v ~/models:/models --entrypoint bash \\
      stilldeadcode/vllm-radiance:0.9.3 -lc 'cd /patches && <build both .so into site-packages as
      run_paroquant.sh does> && python3 paroquant/test_mxfp4_loader.py'
"""
import json
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, "/patches/paroquant")
import radiance_paroquant_mxfp4 as M   # noqa: E402  (registers the op + config)

import os
CKPT = os.environ.get("PQM_CKPT") or "/models/Qwen3.8-27B-PARO-MXFP4"
TP = int(os.environ.get("PQM_TP", "1") or 1)
GRID = torch.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6.])
E4M3_MAX = 448.0


def _weight_map():
    """tensor name -> file, for sharded (index.json) or single-file checkpoints alike."""
    import glob, os
    idx = f"{CKPT}/model.safetensors.index.json"
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for fn in glob.glob(f"{CKPT}/*.safetensors"):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                wm[k] = os.path.basename(fn)
    return wm


def load_module(name):
    idx = _weight_map()
    out = {}
    for leaf in ("weight", "weight_scale", "theta", "pairs", "channel_scales"):
        with safe_open(f"{CKPT}/{idx[f'{name}.{leaf}']}", framework="pt") as f:
            out[leaf] = f.get_tensor(f"{name}.{leaf}")
    return out


def dequant_w(packed, e8m0):                              # [N,K/2] u8, [N,K/32] u8 -> [N,K] f32
    lo, hi = packed & 0xF, (packed >> 4) & 0xF
    codes = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    mag = GRID.to(codes.device)[(codes & 7).long()]
    val = torch.where((codes & 8) != 0, -mag, mag)
    sc = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return val * sc


def rotate_ref(x, pairs, theta, cs_stored):
    """Givens sweep on activations, group-local pairs, with the pre-inverted channel scale."""
    x = (x.float() * cs_stored.float().view(1, -1)).clone()
    K = x.shape[1]
    for r in range(pairs.shape[0]):
        for g in range(K // 128):
            b = g * 128
            idx = pairs[r, b:b + 128].long()
            th = theta[r, b // 2:(b + 128) // 2].float()
            i, j = idx[0::2] + b, idx[1::2] + b
            c, s = torch.cos(th), torch.sin(th)
            xi, xj = x[:, i].clone(), x[:, j].clone()
            x[:, i] = xi * c + xj * s
            x[:, j] = -xi * s + xj * c
    return x


def main():
    torch.manual_seed(0)
    dev = "cuda"
    # A GDN layer's in_proj_qkv + in_proj_z: two DISTINCT rotations merged (P=2), like vLLM does.
    mods_env = os.environ.get("PQM_MODULES") or "linear_attn.in_proj_qkv,linear_attn.in_proj_z"
    names = [f"model.language_model.layers.0.{m}" for m in mods_env.split(",")]
    mods = [load_module(n) for n in names]
    if TP > 1:   # emulate rank 0 of a column-parallel layer: first N/TP rows of each module
        for m in mods:
            rows = m["weight"].shape[0] // TP
            m["weight"] = m["weight"][:rows].contiguous(); m["weight_scale"] = m["weight_scale"][:rows].contiguous()
    print(f"checkpoint {CKPT} | modules {mods_env} | TP-emulation {TP}")
    K = mods[0]["weight"].shape[1] * 2
    sizes = [m["weight"].shape[0] for m in mods]
    N = sum(sizes)
    print(f"merged linear: K={K} N={N} partitions={sizes}")

    # Build the layer's raw parameters directly (shapes/names as create_weights registers them).
    # vLLM's parameter classes need an engine config context, and they are not what carries risk
    # here -- weight prep, partitioning and the kernel composition are, and those run below.
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.cat([m["weight"] for m in mods]), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.cat([m["weight_scale"] for m in mods]), requires_grad=False)
    layer.theta = nn.Parameter(torch.stack([m["theta"] for m in mods]), requires_grad=False)
    layer.pairs = nn.Parameter(torch.stack([m["pairs"] for m in mods]), requires_grad=False)
    layer.channel_scales = nn.Parameter(torch.stack([m["channel_scales"].reshape(-1) for m in mods]),
                                        requires_grad=False)
    layer.pq_output_partition_sizes = list(sizes)
    cfg = M.ParoQuantMXFP4Config.from_config({"bits": 4, "group_size": 128, "krot": 8})
    method = M.ParoQuantMXFP4LinearMethod(cfg)
    layer.to(dev)
    method.process_weights_after_loading(layer)
    print(f"prepared: P={layer.rec.shape[0]} pb1={layer.pq_pb1} pb2={layer.pq_pb2} "
          f"weight={tuple(layer.weight.shape)} ws_t={tuple(layer.ws_t.shape)} wref={tuple(layer.wref.shape)}")
    assert layer.rec.shape[0] == len(mods), "distinct rotations must NOT dedup"

    bounds = [0, sizes[0], N]
    for Mrows in [int(x) for x in (os.environ.get("PQM_MS") or "1,5,40,64,200,600,2048").split(",")]:
        x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        y = method.apply(layer, x).float()
        # reference
        ref = torch.empty(Mrows, N, device=dev)
        for p, m in enumerate(mods):
            xr = rotate_ref(x, m["pairs"].to(dev), m["theta"].to(dev), m["channel_scales"].to(dev))
            s = xr.abs().amax(1, keepdim=True).clamp_min(1e-12) / E4M3_MAX
            xq = (xr / s).to(torch.float8_e4m3fn).float() * s
            w = dequant_w(m["weight"].to(dev), m["weight_scale"].to(dev))
            ref[:, bounds[p]:bounds[p + 1]] = xq @ w.T
        rel = ((y - ref).norm() / ref.norm()).item()
        band = ("decode" if Mrows <= M._mx.DECODE_MAX_M else
                "A-tiled" if M._mx.A_TILED_MIN_M and Mrows >= M._mx.A_TILED_MIN_M else "prefill")
        print(f"  M={Mrows:4d} {band:7s} rel={rel:.4f}")
        assert rel < 4e-2, f"M={Mrows}: rel {rel} too large -- layout or scale mismatch"
    print("PASS: paroquant_mxfp4 linear matches fp32 reference at every band")
    # single-launch merged GEMM (partition select in the kernel) vs the per-partition loop + cat
    print("single launch vs per-partition loop (P=2):")
    for Mrows in [1, 5, 8, 40, 64, 200, 600, 2048]:
        x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        M.SINGLE_LAUNCH = True;  y1 = method.apply(layer, x)
        M.SINGLE_LAUNCH = False; y0 = method.apply(layer, x)
        M.SINGLE_LAUNCH = True
        # not bit-identical by design: the decode band picks split-K from the launch's N, so the
        # merged launch may reassociate the fp32 partial sums differently from the per-partition
        # loop. Equivalent to bf16 rounding: a handful of 1-ulp flips, rel ~1e-4.
        rel = ((y1.float() - y0.float()).norm() / y0.float().norm()).item()
        nd = int((y0 != y1).sum())
        print(f"  M={Mrows:4d}: single vs loop rel={rel:.2e}  differing elems {nd}/{y0.numel()}")
        assert rel < 1e-3, f"M={Mrows}: single-launch output differs from the per-partition loop (rel {rel})"
    print("PASS: single-launch merged GEMM matches the per-partition loop to bf16 rounding")

    # ---- rotation stream producers: the fused (norm|silu|gate|gdn-norm) + rotate + token quant
    # producers must give the SAME GEMM output as feeding their own hs through the plain path
    # (which runs pq_rotate_tokquant on hs) -- bit-exact, at every M below the tiled threshold.
    print("rotation stream (per-token producers) vs plain path on the producer's own hs:")
    wn = (torch.randn(K, device=dev) * 0.1).to(torch.bfloat16)          # Gemma-style (1 + w)
    for Mrows in [1, 5, 8, 40, 64, 200, 600]:
        y = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        res = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
        hs, ro, a, as_tok = torch.ops.radiance.pqm_add_rms_rot(y, res, wn, 1e-6, layer.rec, layer.cs)
        # residual out is exact bf16(y + res) in fp32
        ro_ref = (y.float() + res.float()).to(torch.bfloat16)
        assert torch.equal(ro, ro_ref), f"M={Mrows}: residual out differs"
        # hs vs a torch Gemma RMSNorm (reduction order differs -> allow ulps)
        v = (y.float() + res.float())
        hs_ref = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + wn.float())).to(torch.bfloat16)
        hs_rel = ((hs.float() - hs_ref.float()).norm() / hs_ref.float().norm()).item()
        assert hs_rel < 2e-3, f"M={Mrows}: hs rel {hs_rel} vs torch rmsnorm"
        y_pre = method.apply(layer, (hs, a, as_tok))
        y_plain = method.apply(layer, hs)
        same = torch.equal(y_pre, y_plain)
        print(f"  norm site M={Mrows:4d} {'tiled' if M._tiled(Mrows) else 'row-major'}: pre == plain {same}  hs rel {hs_rel:.2e}")
        assert same, f"M={Mrows}: stream tuple output differs from the plain path"
    # elementwise sites: a single-partition consumer (out_proj-like) -- reuse partition 0 of this
    # layer as the consumer with P=1 by building a P=1 layer from the first module
    l1 = nn.Module()
    l1.weight = nn.Parameter(mods[0]["weight"].clone(), requires_grad=False)
    l1.weight_scale = nn.Parameter(mods[0]["weight_scale"].clone(), requires_grad=False)
    l1.theta = nn.Parameter(mods[0]["theta"].unsqueeze(0).clone(), requires_grad=False)
    l1.pairs = nn.Parameter(mods[0]["pairs"].unsqueeze(0).clone(), requires_grad=False)
    l1.channel_scales = nn.Parameter(mods[0]["channel_scales"].reshape(1, -1).clone(), requires_grad=False)
    l1.pq_output_partition_sizes = [sizes[0]]
    l1.to(dev)
    method.process_weights_after_loading(l1)
    w128 = (torch.randn(128, device=dev) * 0.5).to(torch.bfloat16)
    for mode, label in ((0, "silu-mul"), (1, "attn-gate"), (2, "gdn-norm")):
        for Mrows in [1, 8, 64, 600]:
            if mode == 0:
                x = (torch.randn(Mrows, 2 * K, device=dev) * 0.8).to(torch.bfloat16)
                yy = x
                g, u = x[:, :K].float(), x[:, K:].float()
                hs_ref = ((g / (1 + torch.exp(-g))).to(torch.bfloat16).float() * u).to(torch.bfloat16)
            else:
                x = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
                yy = (torch.randn(Mrows, K, device=dev) * 0.8).to(torch.bfloat16)
                if mode == 1:
                    hs_ref = (x.float() * (1 / (1 + torch.exp(-yy.float()))).to(torch.bfloat16).float()).to(torch.bfloat16)
                else:
                    xv = x.float().view(Mrows, -1, 128)
                    n = xv * torch.rsqrt(xv.pow(2).mean(-1, keepdim=True) + 1e-6) * w128.float()
                    z = yy.float().view(Mrows, -1, 128)
                    hs_ref = (n * (z / (1 + torch.exp(-z)))).view(Mrows, K).to(torch.bfloat16)
            hs, a, as_tok = torch.ops.radiance.pqm_ew_rot(mode, x, yy, w128 if mode == 2 else l1.cs, 1e-6, l1.rec, l1.cs)
            hs_rel = ((hs.float() - hs_ref.float()).norm() / hs_ref.float().norm()).item()
            y_pre = method.apply(l1, (hs, a, as_tok))
            y_plain = method.apply(l1, hs)
            same = torch.equal(y_pre, y_plain)
            print(f"  {label:9s} M={Mrows:4d}: pre == plain {same}  hs rel {hs_rel:.2e}")
            assert same, f"{label} M={Mrows}: stream tuple output differs from the plain path"
            assert hs_rel < 5e-3, f"{label} M={Mrows}: hs rel {hs_rel} vs torch reference"
    print("PASS: per-token rotation-stream producers are output-identical to the plain path")


if __name__ == "__main__":
    main()
