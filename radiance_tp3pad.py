#!/usr/bin/env python3
"""TP=3 for Qwen3.8-27B via zero-weight dummy heads (RADIANCE_TP_PAD=3).

The checkpoint's head counts (24 q / 4 kv / 16 GDN key / 48 GDN value heads, 17408 MLP width,
248320 vocab) do not divide by 3, so vLLM refuses --tensor-parallel-size 3 in three separate
places (Qwen3NextAttention asserts, QwenGatedDeltaNetAttention divides, VocabParallelEmbedding
divides). Upstream rejected padding (vllm-project/vllm#11797). This module pads at RUNTIME,
Megatron-style: the config is widened to the next multiple of 3 and every affected checkpoint
tensor is padded with dummy heads / channels whose weights are exactly zero, BEFORE vLLM's own
sharding weight loaders see it. Contiguous sharding then puts every dummy on the last rank;
ranks 0 and 1 are all-real. Nothing is rewritten on disk.

Geometry (TP3_PADDING_PLAN.md):

  target (text_config)      stock -> padded   per rank    why
    num_attention_heads      24  -> 36        12          keeps GQA 6 (the R4D attention kernels
    num_key_value_heads       4  ->  6         2          are compiled for head 256 / GQA 6)
    linear_num_key_heads     16  -> 18         6          16 does not divide by 3
    linear_num_value_heads   48  -> 54        18          V/K ratio 3 is baked into the GDN layout
    intermediate_size     17408  -> 17472    5824         multiple of 192 -> per-rank down_proj K % 64 == 0
    vocab (runtime pad)   248320 -> 248448   82816        pad multiple 64 -> 192 (VocabParallelEmbedding)

  DFlash2 drafter (32 q / 8 kv / head 128, fp8 128x128 block scales)
    num_attention_heads      32  -> 48        16          keeps GQA 4: q head h reads kv head h // 4,
                                                          so the ratio is part of the trained weights
                                                          (36 / 12 = GQA 3 remapped every real q head
                                                          and the drafter accepted nothing, 2026-09-07)
    num_key_value_heads       8  -> 12         4          NOT 9: the drafter's KV bytes per token must
                                                          equal the target's (12 x 128 = 6 x 256, as
                                                          stock 8 x 128 = 4 x 256) or the two cache
                                                          groups get different block sizes and the
                                                          sliding-window prefix-cache lookup asserts
                                                          ("does not support fine-grained (partial)
                                                          cache hits", measured 2026-09-06)
    intermediate_size     17408  -> 17664    5888         multiple of 384: whole 128-blocks per rank

Fills: MXFP4 packed weight (u8) 0x00 = two e2m1 +0; MXFP4 e8m0 scale 0x7F = 2^0 on dummy ROWS but
0x00 = 2^-127 on padded K COLUMNS of real rows (the kernel folds a row against its max exponent; see
fill_value) and never 0xFF, that is NaN; fp8 weights 0x00; fp32 / bf16 scales 1.0; bf16 values
(conv1d, A_log, dt_bias) 0.
A zero q / k head gives RMSNorm(0) = 0 -> RoPE(0) = 0 -> uniform softmax over V = 0 -> 0, and its
attention gate is sigmoid(0) * 0 = 0; a zero GDN head keeps a zero state from a zero init
(A_log = 0 is a finite decay). Every K-pad lands at the global end of the row and every e8m0
group boundary is 32-aligned, so no scale group straddles real and dummy columns.

Three hooks, all installed by patch_tp3_pad.py and ALL inert unless RADIANCE_TP_PAD is set
(TP 1/2 serves stay byte-identical; with the env off none of the functions below does any work):
  maybe_pad_config(hf_config, hf_text_config)   ModelConfig.__post_init__, after hf_text_config
  pad_weights(weights, model_config)            DefaultModelLoader.load_weights, around the iterator
  vocab_pad_multiple(padding_size)              VocabParallelEmbedding.__init__

Knobs:
  RADIANCE_TP_PAD=3               enable (the only supported value; the launcher sets it at TP=3)
  RADIANCE_TP_PAD_INTERMEDIATE    padded MLP width (default 17472; 17408 = leave it stock, for the
                                  TP=2 heads-only gate)
  RADIANCE_TP_PAD_DRAFTER=0       leave the DFlash2 drafter unpadded (its padded 48/12 geometry
                                  shards at TP 1, 2, 3, 4 and 6; the knob is an A/B lever)
  RADIANCE_TP_PAD_STRICT=0        demote a coverage mismatch in the weight tally to a warning

The shape planning (plan_shape) is torch-free so tp3pad_selftest.py can check every tensor of
both checkpoints against their safetensors headers on a host without torch or a GPU.
"""
import os
import re
import sys
from collections import OrderedDict
from fractions import Fraction
from math import lcm

TAG = "[radiance.tp3pad]"
PAD = 3                      # the tensor-parallel size the dummies make divisible
VOCAB_PAD_BASE = 64          # vLLM's DEFAULT_VOCAB_PADDING_SIZE


def _log(msg):
    sys.stderr.write(f"{TAG} {msg}\n")
    sys.stderr.flush()


def enabled() -> int:
    """0 when off; PAD when RADIANCE_TP_PAD=3. Any other value is a configuration error."""
    v = os.environ.get("RADIANCE_TP_PAD", "").strip()
    if v in ("", "0"):
        return 0
    if v != str(PAD):
        raise RuntimeError(f"{TAG} RADIANCE_TP_PAD={v!r}: only {PAD} is supported")
    return PAD


def padded_intermediate() -> int:
    v = int(os.environ.get("RADIANCE_TP_PAD_INTERMEDIATE", "") or 17472)
    if v < 17408 or v % 64:
        raise RuntimeError(f"{TAG} RADIANCE_TP_PAD_INTERMEDIATE={v}: must be >= 17408 and a multiple of 64")
    return v


def pad_drafter() -> bool:
    return os.environ.get("RADIANCE_TP_PAD_DRAFTER", "1") == "1"


# ------------------------------------------------------------------ geometry
# Exact stock values are a precondition: this module knows ONE model. A different checkpoint with
# RADIANCE_TP_PAD set is an error, not something to guess at.
TARGET_STOCK = OrderedDict(
    hidden_size=5120, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
    linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
    linear_value_head_dim=128, intermediate_size=17408, vocab_size=248320,
)
DRAFTER_STOCK = OrderedDict(
    hidden_size=5120, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    intermediate_size=17408, vocab_size=248320,
)


def target_padded(intermediate=None):
    return OrderedDict(num_attention_heads=36, num_key_value_heads=6, linear_num_key_heads=18,
                       linear_num_value_heads=54,
                       intermediate_size=padded_intermediate() if intermediate is None else intermediate)


def drafter_padded():
    return OrderedDict(num_attention_heads=48, num_key_value_heads=12, intermediate_size=17664)


# ------------------------------------------------------------------ tensor rules
# (label, name regex, kind, old, new). Sizes are LOGICAL (bf16 element counts along the padded
# axis); the per-tensor size is scaled by the tensor's own ratio to the logical size, which is
# how one rule serves an MXFP4 packed weight (K/2 columns), its e8m0 scale (K/32), an fp8 weight
# (K) and a 128x128 block scale (K/128) alike.
#   rows      dim 0 (output channels): old -> new
#   kcols     dim 1 (input channels) of a 2-D tensor: old -> new; 1-D tensors are left alone
#   sections  dim 0 is a concatenation of old sections, each padded to its new size in place
_T_ATTN = r"\.self_attn\.{p}\.(weight|weight_scale)$"
_T_GDN = r"\.linear_attn\.{p}\.(weight|weight_scale)$"
_T_MLP = r"\.mlp\.{p}\.(weight|weight_scale)$"


def target_rules(intermediate=None):
    inter = padded_intermediate() if intermediate is None else intermediate
    qkv_old, qkv_new = [2048, 2048, 6144], [2304, 2304, 6912]
    return [
        ("q_proj", _T_ATTN.format(p="q_proj"), "rows", 12288, 18432),
        ("k_proj", _T_ATTN.format(p="k_proj"), "rows", 1024, 1536),
        ("v_proj", _T_ATTN.format(p="v_proj"), "rows", 1024, 1536),
        ("o_proj", _T_ATTN.format(p="o_proj"), "kcols", 6144, 9216),
        ("gate_proj", _T_MLP.format(p="gate_proj"), "rows", 17408, inter),
        ("up_proj", _T_MLP.format(p="up_proj"), "rows", 17408, inter),
        ("down_proj", _T_MLP.format(p="down_proj"), "kcols", 17408, inter),
        ("in_proj_qkv", _T_GDN.format(p="in_proj_qkv"), "sections", qkv_old, qkv_new),
        ("in_proj_z", _T_GDN.format(p="in_proj_z"), "rows", 6144, 6912),
        ("in_proj_b", _T_GDN.format(p="in_proj_b"), "rows", 48, 54),
        ("in_proj_a", _T_GDN.format(p="in_proj_a"), "rows", 48, 54),
        ("out_proj", _T_GDN.format(p="out_proj"), "kcols", 6144, 6912),
        ("conv1d", r"\.linear_attn\.conv1d\.weight$", "sections", qkv_old, qkv_new),
        ("A_log", r"\.linear_attn\.A_log$", "rows", 48, 54),
        ("dt_bias", r"\.linear_attn\.dt_bias$", "rows", 48, 54),
    ]


_D_ATTN = r"\.self_attn\.{p}\.(weight|weight_scale_inv)$"
_D_MLP = r"\.mlp\.{p}\.(weight|weight_scale_inv)$"


def drafter_rules():
    return [
        ("q_proj", _D_ATTN.format(p="q_proj"), "rows", 4096, 6144),
        ("k_proj", _D_ATTN.format(p="k_proj"), "rows", 1024, 1536),
        ("v_proj", _D_ATTN.format(p="v_proj"), "rows", 1024, 1536),
        ("o_proj", _D_ATTN.format(p="o_proj"), "kcols", 4096, 6144),
        ("gate_proj", _D_MLP.format(p="gate_proj"), "rows", 17408, 17664),
        ("up_proj", _D_MLP.format(p="up_proj"), "rows", 17408, 17664),
        ("down_proj", _D_MLP.format(p="down_proj"), "kcols", 17408, 17664),
    ]


def rules_for(spec, intermediate=None):
    if spec == "target":
        return target_rules(intermediate)
    if spec == "drafter":
        return drafter_rules()
    raise ValueError(spec)


_COMPILED = {}


def _compiled(spec, intermediate):
    key = (spec, intermediate)
    if key not in _COMPILED:
        _COMPILED[key] = [(lab, re.compile(rx), kind, old, new)
                          for lab, rx, kind, old, new in rules_for(spec, intermediate)]
    return _COMPILED[key]


def _scaled(size, old, new, what):
    """size is `old` measured in this tensor's own units; return the same units of `new`."""
    r = Fraction(size, old)
    out = r * new
    if out.denominator != 1:
        raise RuntimeError(f"{TAG} {what}: {size} is {r} x {old}, which does not scale to {new}")
    return int(out)


def plan_shape(name, shape, spec, intermediate=None):
    """(label, kind, new_shape, sections) for a tensor this spec pads, else None. Torch-free.

    sections is (old_sizes, new_sizes) in the tensor's own dim-0 units for kind == sections."""
    shape = list(shape)
    for lab, rx, kind, old, new in _compiled(spec, intermediate):
        if not rx.search(name):
            continue
        if old == new:
            return None                           # RADIANCE_TP_PAD_INTERMEDIATE=17408: MLP left stock
        if kind == "rows":
            if shape[0] * new % old:
                raise RuntimeError(f"{TAG} {name}: dim0 {shape[0]} is not {old} in any unit")
            return lab, kind, [_scaled(shape[0], old, new, name)] + shape[1:], None
        if kind == "kcols":
            if len(shape) < 2:
                return None                       # a per-channel scale has no K axis
            return lab, kind, [shape[0], _scaled(shape[1], old, new, name)] + shape[2:], None
        if kind == "sections":
            tot_old = sum(old)
            r = Fraction(shape[0], tot_old)
            olds = [r * o for o in old]
            news = [r * n for n in new]
            if any(x.denominator != 1 for x in olds + news):
                raise RuntimeError(f"{TAG} {name}: dim0 {shape[0]} is not a whole multiple of {old}")
            olds = [int(x) for x in olds]
            news = [int(x) for x in news]
            return lab, kind, [sum(news)] + shape[1:], (olds, news)
        raise ValueError(kind)
    return None


def fill_value(name, dtype_name, kind="rows"):
    """The pad value for a tensor, by role. dtype_name is torch's str (e.g. 'torch.uint8')."""
    leaf = name.rsplit(".", 1)[-1]
    if dtype_name.endswith("uint8"):
        # MXFP4 checkpoint: 'weight' is packed e2m1 pairs, 'weight_scale' is e8m0 exponents.
        # A padded ROW is a whole dummy output channel: its weights are zero, so its exponent is
        # free and 0x7F (2^0) keeps it an ordinary-looking row. A padded COLUMN group sits on a
        # REAL row, and the radiance kernel folds that row's groups against the row's MAXIMUM
        # exponent (radiance_mxfp4.make_row_ref): 2^0 there outranks every real group (typically
        # 2^-4 .. 2^-12) and re-expresses the real weights as e4m3 relative to 2^0, flushing them to
        # subnormal or zero -- measured 2026-09-06 as a serve that emitted nothing but token 0.
        # 0x00 = 2^-127 never wins the max, and 0 x 2^-127 is still 0. (0xFF is NaN; never that.)
        if leaf == "weight_scale":
            return 0x00 if kind == "kcols" else 0x7F
        return 0
    if "float8" in dtype_name:
        return 0.0
    if "scale" in leaf:
        return 1.0
    return 0.0


# ------------------------------------------------------------------ config hook
def _identify(hf_config, hf_text_config):
    """'target' | 'drafter' | None for a config this module knows how to pad."""
    arch = list(getattr(hf_config, "architectures", None) or [])
    if any("DFlash2" in a for a in arch) or getattr(hf_config, "dflash_config", None) is not None:
        return "drafter"
    mt = getattr(hf_text_config, "model_type", "") or getattr(hf_config, "model_type", "")
    if mt in ("qwen3_5_text", "qwen3_5"):
        return "target"
    return None


def maybe_pad_config(hf_config, hf_text_config):
    """Widen the head counts / MLP width on hf_text_config in place. Returns the spec applied
    ('target' / 'drafter') or None. No-op unless RADIANCE_TP_PAD is set; idempotent (a marker
    attribute survives to_dict()/deepcopy, so a wrapped or copied config is not padded twice)."""
    if not enabled():
        return None
    for cfg in (hf_text_config, hf_config):
        marker = getattr(cfg, "_radiance_tp_pad", None)
        if marker is not None:
            return marker if marker in ("target", "drafter") else None
    spec = _identify(hf_config, hf_text_config)
    if spec is None:
        _log(f"config {getattr(hf_config, 'model_type', '?')} {getattr(hf_config, 'architectures', '?')} "
             f"is not a model this pads; left unchanged")
        setattr(hf_text_config, "_radiance_tp_pad", "none")
        return None
    if spec == "drafter" and not pad_drafter():
        _log("RADIANCE_TP_PAD_DRAFTER=0: drafter config left unpadded (serves at TP 1/2 only)")
        setattr(hf_text_config, "_radiance_tp_pad", "none")
        return None
    stock = TARGET_STOCK if spec == "target" else DRAFTER_STOCK
    padded = target_padded() if spec == "target" else drafter_padded()
    for k, v in stock.items():
        got = getattr(hf_text_config, k, None)
        if got != v:
            raise RuntimeError(f"{TAG} {spec} config {k}={got}, expected the stock {v}: this is not the "
                               f"checkpoint the padding was derived for (unset RADIANCE_TP_PAD)")
    for k, v in padded.items():
        setattr(hf_text_config, k, v)
    setattr(hf_text_config, "_radiance_tp_pad", spec)
    _log(f"{spec} config padded for TP={PAD}: " +
         ", ".join(f"{k} {stock[k]}->{v}" for k, v in padded.items()))
    return spec


def vocab_pad_multiple(padding_size):
    """VocabParallelEmbedding pad-to multiple: unchanged when off, lcm(padding_size, 64*3) when on
    (248320 -> 248448, 82816 per rank). Stays a multiple of the caller's own size (LoRA's 256)."""
    if not enabled():
        return padding_size
    return lcm(int(padding_size), VOCAB_PAD_BASE * PAD)


# ------------------------------------------------------------------ weight hook
def _spec_of(model_config):
    for attr in ("hf_text_config", "hf_config"):
        cfg = getattr(model_config, attr, None)
        m = getattr(cfg, "_radiance_tp_pad", None) if cfg is not None else None
        if m in ("target", "drafter"):
            return m, cfg
    return None, None


def _expected_counts(spec, cfg, mtp_layers):
    """label -> expected padded-tensor count, from the config's layer table."""
    if spec == "drafter":
        n = int(getattr(cfg, "num_hidden_layers", 0))
        return {"q_proj": 2 * n, "k_proj": 2 * n, "v_proj": 2 * n, "o_proj": 2 * n,
                "gate_proj": 2 * n, "up_proj": 2 * n, "down_proj": 2 * n}
    lt = list(getattr(cfg, "layer_types", []) or [])
    n_full = lt.count("full_attention") + mtp_layers
    n_lin = lt.count("linear_attention")
    n_all = len(lt) + mtp_layers
    # MXFP4 layers carry (weight, weight_scale) and both are padded. The fp8 MTP layer carries the
    # same two names, but its weight_scale is per output channel (1-D), which a K-pad leaves alone:
    # o_proj / down_proj pad one tensor per MTP layer, the rest two.
    n_full_mx, n_all_mx = lt.count("full_attention"), len(lt)
    exp = {"q_proj": 2 * n_full, "k_proj": 2 * n_full, "v_proj": 2 * n_full,
           "o_proj": 2 * n_full_mx + mtp_layers,
           "gate_proj": 2 * n_all, "up_proj": 2 * n_all, "down_proj": 2 * n_all_mx + mtp_layers,
           "in_proj_qkv": 2 * n_lin, "in_proj_z": 2 * n_lin, "in_proj_b": 2 * n_lin,
           "in_proj_a": 2 * n_lin, "out_proj": 2 * n_lin,
           "conv1d": n_lin, "A_log": n_lin, "dt_bias": n_lin}
    if int(getattr(cfg, "intermediate_size", 0)) == TARGET_STOCK["intermediate_size"]:
        for k in ("gate_proj", "up_proj", "down_proj"):      # MLP left stock: nothing to pad
            del exp[k]
    return exp


def pad_tensor(name, t, plan):
    """Return the padded copy of t per plan_shape's plan."""
    import torch
    lab, kind, new_shape, sections = plan
    fill = fill_value(name, str(t.dtype), kind)
    # zeros() is a memset and works for every dtype including float8; full() is only asked for
    # the two non-zero fills (e8m0 0x7F on u8, 1.0 on fp32 / bf16 scales).
    if fill == 0:
        out = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
    else:
        out = torch.full(new_shape, fill, dtype=t.dtype, device=t.device)
    if kind == "sections":
        olds, news = sections
        src = dst = 0
        for o, n in zip(olds, news):
            out.narrow(0, dst, o).copy_(t.narrow(0, src, o))
            src += o
            dst += n
    elif kind == "rows":
        out.narrow(0, 0, t.shape[0]).copy_(t)
    else:  # kcols
        out.narrow(1, 0, t.shape[1]).copy_(t)
    return out


def pad_weights(weights, model_config):
    """Wrap a (name, tensor) iterable so every tensor the spec names comes out padded. Returns the
    iterable untouched (not even wrapped) when this model is not being padded."""
    spec, cfg = _spec_of(model_config)
    if spec is None:
        return weights
    inter = getattr(cfg, "intermediate_size", None) if spec == "target" else None

    def gen():
        tally = OrderedDict()
        seen = 0
        mtp = set()
        for name, t in weights:
            seen += 1
            m = re.match(r"^mtp\.layers\.(\d+)\.", name)
            if m:
                mtp.add(m.group(1))
            plan = plan_shape(name, t.shape, spec, inter)
            if plan is None:
                yield name, t
                continue
            old_shape = tuple(t.shape)
            t = pad_tensor(name, t, plan)
            key = (plan[0], old_shape, tuple(t.shape))
            tally[key] = tally.get(key, 0) + 1
            yield name, t
        counts = {}
        for (lab, old, new), n in tally.items():
            _log(f"padded {lab} {n}x {list(old)}->{list(new)}")
            counts[lab] = counts.get(lab, 0) + n
        exp = _expected_counts(spec, cfg, len(mtp))
        bad = {k: (counts.get(k, 0), v) for k, v in exp.items() if counts.get(k, 0) != v}
        extra = sorted(set(counts) - set(exp))
        total = sum(counts.values())
        if bad or extra:
            msg = (f"{spec} coverage MISMATCH: {total} tensors padded of {seen}; "
                   f"got/expected {bad}" + (f"; unexpected labels {extra}" if extra else ""))
            if os.environ.get("RADIANCE_TP_PAD_STRICT", "1") == "1":
                raise RuntimeError(f"{TAG} {msg} (RADIANCE_TP_PAD_STRICT=0 to demote)")
            _log("WARNING " + msg)
        else:
            _log(f"{spec} coverage OK: {total} tensors padded of {seen} "
                 f"({len(exp)} patterns{', MTP layers ' + str(sorted(mtp)) if mtp else ''})")

    return gen()
