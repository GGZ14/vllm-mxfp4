#!/usr/bin/env python3
"""MoE lane: fast prefill attention on gfx1201, behind RADIANCE_MOE_PREFILL_ATTN (default off).

On the R9700, vLLM's Triton unified attention runs prefill at 2.7 TFLOP/s for Qwen3.5-35B-A3B
(head 256, 16 q / 2 kv heads, fp8 KV): 63-87% of MoE-lane prefill time. Its gfx1201 launch config is
decode-shaped (BLOCK_M 16 = 2 query tokens per program, num_stages 2 -> 182 VGPR spills), and Triton
3.6 lowers the fp8 -> f32 KV dequant to ~15 software VALU ops per element. See
radiance-moe-attn/README.md for the measurements.

RADIANCE_MOE_PREFILL_ATTN (read at run time):
  off     stock behaviour (the patched code paths are inert).
  triton  2D (prefill-shaped) Triton launches with max_seqlen_q >= RADIANCE_MOE_PREFILL_MIN_Q
          (default 17, just above the 3D split-KV cap of 16) and head 256 use BLOCK_M 128 /
          TILE 32 / 8 warps / 1 stage: ~10.8x on the attention kernel at 16k. No layout change;
          works with --attention-backend TRITON_ATTN.
  r4d     additionally, with --attention-backend R4D (main model AND MTP drafter), request runs of
          >= RADIANCE_MOE_PREFILL_MIN_Q query tokens go to libr4d's paged prefill kernel
          instantiated at GQA 8 (module r4d_moe, built from patch/r4d_moe/): ~40x at 16k. Every other
          run -- decode, MTP verify (1+8 tokens), drafter catch-up, short chunks -- goes to the stock
          Triton path on a sub-batch, including patch_attn_3d_multiq.py's 3D split-KV kernel. The R4D
          decode kernel is never used here: it holds 64 rows = 8 query tokens at GQA 8, less than a
          9-token MTP-8 verify.

Parts:
  1. triton_unified_attention.py: the prefill launch config (modes triton and r4d).
  2. radiance_r4d_attn.py: bind r4d_moe, accept GQA 8, route runs (mode r4d only).
  3. copy r4d_moe*.so next to radiance_r4d_attn.py (mode r4d only; built by moe-r4d/build.sh
     against this image; RADIANCE_R4D_MOE_SO names the built file, default r4d_moe<EXT> next to
     this patch).
Must run after patch_attn_3d_multiq.py (independent anchors, but that is the tested order).
"""
import os
import shutil
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
MODE = os.environ.get("RADIANCE_MOE_PREFILL_ATTN", "off")
if MODE not in ("off", "triton", "r4d"):
    raise SystemExit(f"  FAIL  RADIANCE_MOE_PREFILL_ATTN={MODE!r}: expected off, triton or r4d")
if MODE == "off":
    print("  SKIP  moe prefill attention: RADIANCE_MOE_PREFILL_ATTN=off")
    raise SystemExit(0)

# ---- 1. Triton prefill launch config ---------------------------------------------------------
apply(
    SP / "vllm/v1/attention/ops/triton_unified_attention.py",
    "    if tuned_large_head:\n"
    "        BLOCK_M = 32\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 2\n",
    "    if tuned_large_head:\n"
    "        BLOCK_M = 32\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 2\n"
    "    # radiance: gfx1201 prefill tiles (patch_moe_prefill_attn.py). The default BLOCK_M 16 fetches\n"
    "    # and dequantises every KV tile per 2 query tokens (GQA 8) and spills at num_stages 2.\n"
    "    _rad_env = __import__('os').environ\n"
    "    if (\n"
    "        not tuned_large_head\n"
    "        and _rad_env.get('RADIANCE_MOE_PREFILL_ATTN', 'off') in ('triton', 'r4d')\n"
    "        and head_size == 256\n"
    "        and num_queries_per_kv <= 16\n"
    "        and 128 % num_queries_per_kv == 0\n"
    "        and not use_td\n"
    "        and max_seqlen_q >= int(_rad_env.get('RADIANCE_MOE_PREFILL_MIN_Q', '17'))\n"
    "    ):\n"
    "        BLOCK_M = 128\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 1\n",
    "radiance: gfx1201 prefill tiles (patch_moe_prefill_attn.py)",
    "unified_attention: gfx1201 prefill launch config",
)

if MODE != "r4d":
    raise SystemExit(0)

# ---- 2. R4D backend: GQA-8 prefill kernel + run routing ---------------------------------------
R4DF = SP / "radiance_r4d_attn.py"

apply(
    R4DF,
    '_DECODE = tuple(_bind("attn_decode_paged", kv_dtype=d, q_len=1, **_PAGED) for d in _KV_DTYPES)\n',
    '_DECODE = tuple(_bind("attn_decode_paged", kv_dtype=d, q_len=1, **_PAGED) for d in _KV_DTYPES)\n'
    "\n"
    "# --- radiance MoE prefill attention (patch_moe_prefill_attn.py) ---\n"
    "# RADIANCE_MOE_PREFILL_ATTN=r4d: libr4d's prefill kernel instantiated at GQA 8 (module r4d_moe)\n"
    "# serves request runs of >= RADIANCE_MOE_PREFILL_MIN_Q query tokens; every other run goes to the\n"
    "# stock Triton path. Asked for explicitly, so a missing module is fatal, never a silent fallback.\n"
    '_MOE_MODE = os.environ.get("RADIANCE_MOE_PREFILL_ATTN", "off")\n'
    '_MOE_MIN_Q = int(os.environ.get("RADIANCE_MOE_PREFILL_MIN_Q", "17"))\n'
    "_MOE_PREFILL = None\n"
    "_MOE_GQA = 0\n"
    'if _MOE_MODE == "r4d" and USE_R4D:\n'
    "    _dlf = sys.getdlopenflags()\n"
    "    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)\n"
    "    try:\n"
    "        import r4d_moe as _r4d_moe\n"
    "    finally:\n"
    "        sys.setdlopenflags(_dlf)\n"
    "    _MOE_PREFILL = (\n"
    "        _r4d_moe.attn_prefill_h256_gqa8_fp8kv,\n"
    "        _r4d_moe.attn_prefill_h256_gqa8_bf16kv,\n"
    "    )\n"
    "    _MOE_GQA = int(_r4d_moe.ATTN_GQA)\n"
    "    assert int(_r4d_moe.ATTN_BLOCK_SIZE) == BLOCK_SIZE, (_r4d_moe.ATTN_BLOCK_SIZE, BLOCK_SIZE)\n",
    "radiance MoE prefill attention (patch_moe_prefill_attn.py)",
    "radiance_r4d_attn: bind r4d_moe GQA-8 prefill",
)

apply(
    R4DF,
    '        if r4d.select("attn_prefill_paged", **geometry) is None:\n',
    "        # radiance MoE: GQA 8 is served by r4d_moe's prefill + the Triton path (see _rad_moe_forward)\n"
    "        self._rad_moe = (\n"
    "            _MOE_PREFILL is not None\n"
    "            and self.num_queries_per_kv == _MOE_GQA\n"
    "            and self.head_size == HEAD_DIM\n"
    "        )\n"
    "        if self._rad_moe:\n"
    '            _say(f"MoE prefill attention: r4d_moe GQA {_MOE_GQA} prefill for runs >= {_MOE_MIN_Q} "\n'
    '                 "query tokens, Triton for the rest")\n'
    '        if not self._rad_moe and r4d.select("attn_prefill_paged", **geometry) is None:\n',
    "radiance MoE: GQA 8 is served by r4d_moe",
    "radiance_r4d_attn: accept GQA 8 in MoE mode",
)

apply(
    R4DF,
    "    def forward(\n"
    "        self,\n"
    "        layer: torch.nn.Module,\n",
    "    def _rad_moe_forward(self, layer, query, key, value, kv_cache, m, output, plan):\n"
    "        # radiance MoE prefill attention (patch_moe_prefill_attn.py). Long runs -> r4d_moe GQA-8\n"
    "        # prefill; the rest -> the stock Triton forward on sub-batches of consecutive requests.\n"
    "        # A batch with no long run (every decode / verify step, so every graph capture) takes the\n"
    "        # unmodified Triton call with the unmodified metadata.\n"
    "        if not any(g[2] >= _MOE_MIN_Q for g in plan):\n"
    "            return TritonAttentionImpl.forward(\n"
    "                self, layer, query, key, value, kv_cache, m, output\n"
    "            )\n"
    "        if (\n"
    '            getattr(m, "mm_prefix_range_tensor", None) is not None\n'
    '            or getattr(m, "rswa_prefix_lens", None) is not None\n'
    "        ):\n"
    "            return TritonAttentionImpl.forward(\n"
    "                self, layer, query, key, value, kv_cache, m, output\n"
    "            )\n"
    "        variant, block_stride, head_stride = self._geometry(kv_cache, query, output)\n"
    "        k_descale, v_descale = self._descale_ptrs(layer, query.device)\n"
    "        block_table = m.block_table\n"
    "        max_blocks = block_table.shape[1]\n"
    "        q_row = self.num_heads * self.head_size * query.element_size()\n"
    "        o_row = self.num_heads * self.head_size * output.element_size()\n"
    "        q_base, o_base = query.data_ptr(), output.data_ptr()\n"
    "        bt_base, sl_base = block_table.data_ptr(), m.seq_lens.data_ptr()\n"
    "        kv_ptr = kv_cache.data_ptr()\n"
    "        max_ctx = m.r4d_max_ctx\n"
    "        stream = torch.cuda.current_stream().cuda_stream\n"
    "        launch = _MOE_PREFILL[variant]\n"
    "        seg = None  # [first_req, end_req, first_tok, end_tok, max_q_len] of the pending Triton run\n"
    "\n"
    "        def flush(seg):\n"
    "            r0, r1, t0, t1, mq = seg\n"
    "            sub = copy.copy(m)\n"
    "            sub.num_actual_tokens = t1 - t0\n"
    "            sub.query_start_loc = m.query_start_loc[r0 : r1 + 1] - t0\n"
    "            sub.seq_lens = m.seq_lens[r0:r1]\n"
    "            sub.block_table = m.block_table[r0:r1]\n"
    "            sub.max_query_len = mq\n"
    "            TritonAttentionImpl.forward(\n"
    "                self, layer, query[t0:t1], key[t0:t1], value[t0:t1], kv_cache, sub,\n"
    "                output[t0:t1],\n"
    "            )\n"
    "\n"
    "        for first_req, num_seqs, q_len, first_tok in plan:\n"
    "            if q_len < _MOE_MIN_Q:\n"
    "                end_req, end_tok = first_req + num_seqs, first_tok + num_seqs * q_len\n"
    "                if seg is not None and seg[1] == first_req and seg[3] == first_tok:\n"
    "                    seg = [seg[0], end_req, seg[2], end_tok, max(seg[4], q_len)]\n"
    "                else:\n"
    "                    if seg is not None:\n"
    "                        flush(seg)\n"
    "                    seg = [first_req, end_req, first_tok, end_tok, q_len]\n"
    "                continue\n"
    "            launch(\n"
    "                q_base + first_tok * q_row,\n"
    "                kv_ptr,\n"
    "                bt_base + first_req * max_blocks * 4,\n"
    "                sl_base + first_req * 4,\n"
    "                o_base + first_tok * o_row,\n"
    "                k_descale,\n"
    "                v_descale,\n"
    "                0,\n"
    "                num_seqs,\n"
    "                q_len,\n"
    "                self.num_heads,\n"
    "                self.num_kv_heads,\n"
    "                self.head_size,\n"
    "                BLOCK_SIZE,\n"
    "                max_blocks,\n"
    "                block_stride,\n"
    "                head_stride,\n"
    "                self.scale,\n"
    "                0,\n"
    "                max_ctx,\n"
    "                stream,\n"
    "            )\n"
    "        if seg is not None:\n"
    "            flush(seg)\n"
    "        return output\n"
    "\n"
    "    def forward(\n"
    "        self,\n"
    "        layer: torch.nn.Module,\n",
    "def _rad_moe_forward(self, layer, query, key, value, kv_cache, m, output, plan):",
    "radiance_r4d_attn: MoE run routing",
)

apply(
    R4DF,
    "        variant, block_stride, head_stride = self._geometry(kv_cache, query, output)\n"
    "        k_descale, v_descale = self._descale_ptrs(layer, query.device)\n"
    "        block_table = attn_metadata.block_table\n",
    '        if getattr(self, "_rad_moe", False):\n'
    "            return self._rad_moe_forward(\n"
    "                layer, query, key, value, kv_cache, attn_metadata, output, plan\n"
    "            )\n"
    "        variant, block_stride, head_stride = self._geometry(kv_cache, query, output)\n"
    "        k_descale, v_descale = self._descale_ptrs(layer, query.device)\n"
    "        block_table = attn_metadata.block_table\n",
    'if getattr(self, "_rad_moe", False):',
    "radiance_r4d_attn: MoE dispatch",
)

apply(
    R4DF,
    "import os\nimport sys\n\nimport torch\n",
    "import copy\nimport os\nimport sys\n\nimport torch\n",
    "import copy\nimport os\nimport sys\n",
    "radiance_r4d_attn: import copy",
)

# ---- 3. the r4d_moe extension ----------------------------------------------------------------
ext = sysconfig.get_config_var("EXT_SUFFIX")
src = Path(os.environ.get("RADIANCE_R4D_MOE_SO") or Path(__file__).resolve().parent / f"r4d_moe{ext}")
dst = SP / f"r4d_moe{ext}"
if not src.exists():
    raise SystemExit(f"  FAIL  r4d_moe: {src} missing (build it with moe-r4d/build.sh in this image)")
if not dst.exists() or dst.read_bytes() != src.read_bytes():
    shutil.copyfile(src, dst)
    print(f"  OK    r4d_moe: {dst}")
else:
    print("  NOOP  r4d_moe already installed")
