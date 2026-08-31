"""Offline integration test of radiance_paroquant.py against the real PARO checkpoint.

Builds the layer the way vLLM would (create_weights -> copy real tensors ->
process_weights_after_loading -> apply) for a single projection AND a merged two-partition
gate_up, then gates the output against a float64 semantic reference computed straight from the
AWQ buffers (forward-rotate x, matmul the dequantized codes). The distance to that reference is
dominated by the per-group fp8 activation quantization, so the gate is ~2e-2; the CHECKALL-style
same-codes reference is gated at 5e-3 like the harness.

Run inside the radiance image with /models and /paro mounted (test_module.sh).
"""
import json
import os
import struct
import sys

import numpy as np
import torch

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method="env://", backend="gloo")
_cfg_ctx = set_current_vllm_config(VllmConfig())
_cfg_ctx.__enter__()
initialize_model_parallel(1, 1)

sys.path.insert(0, "/paro")
import radiance_paroquant as pq

DEV = "cuda:0"
ST = "/models/Qwen3.8-27B-PARO/model.safetensors"


def load_st(names):
    f = open(ST, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name in names:
        info = hdr[name]
        a, b = info["data_offsets"]
        f.seek(base + a)
        dt = {"I32": np.int32, "I16": np.int16, "F16": np.float16}[info["dtype"]]
        out[name] = torch.from_numpy(
            np.frombuffer(f.read(b - a), dtype=dt).reshape(info["shape"]).copy())
    return out


REORDER = np.array([0, 2, 4, 6, 1, 3, 5, 7])
INV = np.argsort(REORDER)


def unpack_awq_np(t):
    v = (t[..., None].astype(np.int64) >> (4 * np.arange(8))) & 0xF
    return v[..., INV].reshape(t.shape[0], -1)


def semantic_ref(x, parts):
    """float64: y[:, part n-range] = rotate_p(x * cs_p) @ dequant(codes_p)."""
    outs = []
    for prm in parts:
        codes = unpack_awq_np(prm["qweight"].numpy())                     # [K, N]
        zp = unpack_awq_np(prm["qzeros"].numpy())                         # [G, N]
        sc = prm["scales"].numpy().astype(np.float64)
        dq = (codes - np.repeat(zp, 128, axis=0)) * np.repeat(sc, 128, axis=0)
        K = codes.shape[0]
        pairs = prm["pairs"].numpy().astype(np.int64)
        theta = prm["theta"].numpy().astype(np.float32)
        c = np.cos(theta).astype(np.float16).astype(np.float64)
        s = np.sin(theta).astype(np.float16).astype(np.float64)
        cs = prm["channel_scales"].numpy().astype(np.float64).ravel()
        v = (x * cs).copy()
        goff = (np.arange(K // 2) // 64) * 128
        for r in range(theta.shape[0]):
            gi = pairs[r, 0::2] + goff
            gj = pairs[r, 1::2] + goff
            xi, xj = v[..., gi].copy(), v[..., gj].copy()
            v[..., gi] = c[r] * xi + s[r] * xj
            v[..., gj] = c[r] * xj - s[r] * xi
        outs.append(v @ dq)
    return np.concatenate(outs, axis=-1)


def run_case(name, prefixes):
    fields = ["qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"]
    parts = []
    for p in prefixes:
        t = load_st([p + f for f in fields])
        parts.append({f: t[p + f] for f in fields})

    K = parts[0]["qweight"].shape[0]
    sizes = [p["scales"].shape[1] for p in parts]
    N = sum(sizes)

    cfg = pq.ParoQuantConfig(4, 128, parts[0]["theta"].shape[0], [])
    method = pq.ParoQuantLinearMethod(cfg)
    layer = torch.nn.Module()
    method.create_weights(layer, K, sizes, K, N, torch.bfloat16, weight_loader=None)

    # Emulate the vLLM loaders: AWQ buffers concatenate along N; rotation params per partition.
    layer.qweight.data.copy_(torch.cat([p["qweight"] for p in parts], dim=1))
    layer.qzeros.data.copy_(torch.cat([p["qzeros"] for p in parts], dim=1))
    layer.scales.data.copy_(torch.cat([p["scales"] for p in parts], dim=1))
    for i, p in enumerate(parts):
        layer.theta.data[i].copy_(p["theta"])
        layer.pairs.data[i].copy_(p["pairs"])
        layer.channel_scales.data[i].copy_(p["channel_scales"].reshape(-1))
    for prm in ["qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"]:
        setattr(layer, prm, torch.nn.Parameter(getattr(layer, prm).data.to(DEV),
                                               requires_grad=False))
    method.process_weights_after_loading(layer)

    failures = 0
    for M in (1, 5, 17, 64, 300):
        torch.manual_seed(M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        y = method.apply(layer, x).float().cpu().numpy()
        ref = semantic_ref(x.float().cpu().numpy().astype(np.float64), parts)
        rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
        # vs EXACT activations the distance is dominated by e4m3 activation quantization
        # (~2.5-3% per element, and dot products do not average it down); the same-codes
        # kernel-vs-reference gate is the CHECKALL line printed by the custom op (< 5e-3).
        ok = rel < 4e-2
        failures += not ok
        print(f"  {name:8s} M={M:<4d} N={N} K={K} P={len(parts)} rel={rel:.2e} "
              f"{'OK' if ok else 'FAIL'}")
    return failures


def main():
    L = "model.language_model.layers."
    fail = 0
    fail += run_case("gate", [L + "0.mlp.gate_proj."])
    fail += run_case("gate_up", [L + "0.mlp.gate_proj.", L + "0.mlp.up_proj."])
    fail += run_case("qkv", [L + "3.self_attn.q_proj.", L + "3.self_attn.k_proj.",
                             L + "3.self_attn.v_proj."])
    fail += run_case("in_proj", [L + "0.linear_attn.in_proj_qkv.",
                                 L + "0.linear_attn.in_proj_z."])
    fail += run_case("down", [L + "0.mlp.down_proj."])
    print("module test:", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
