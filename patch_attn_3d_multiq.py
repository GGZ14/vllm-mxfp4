#!/usr/bin/env python3
"""TRITON_ATTN: use the split-KV (3D) unified-attention kernel for short multi-token queries.

`unified_attention` picks its 3D kernel (grid num_q_blocks x kv_heads x 16 KV segments, then a
reduce) only when `max_seqlen_q == 1`. Speculative-decode verify steps (1+k query tokens per
sequence) and drafter catch-up passes therefore run the 2D kernel, whose grid is just
num_q_blocks x kv_heads: at batch 1 on a GQA-8 model that is ~10 workgroups walking the whole KV.
Measured on one R9700, Qwen3.5-35B-A3B MXFP4 + MTP-8: decode 72.5 / 42.2 / 21.9 tok/s at
830 / 6.1k / 18.3k context, against 79.8 / 74.5 / 64.8 with speculation off.

The kernel already supports several query tokens per sequence in 3D mode: q blocks resolve their
sequence from cu_seqlens_q, every row is causally masked per token, fully masked rows are guarded
in softmax_step (m_j -> 0, L -> 0), and reduce_segments maps each token to its sequence. What
does not fit is the scratch: softmax_segm_* are indexed by query token but sized to
seq_threshold_3D rows (sequences). So:

1. triton_attn.py sizes the scratch to max(seq_threshold_3D, RADIANCE_ATTN_3D_TOKENS) rows
   (default 128; ~32 MB at 16 q heads x 16 segments x 256 dims fp32).
2. unified_attention allows 3D while max_seqlen_q <= RADIANCE_ATTN_3D_MAX_Q (default 16) and the
   batch's query tokens fit those rows; anything larger keeps the 2D kernel.
   RADIANCE_ATTN_3D_MAX_Q=1 restores stock behavior (read per call, so it A/Bs in-process).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

apply(
    SP / "vllm/v1/attention/backends/triton_attn.py",
    "        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n"
    "        headdim_padded = next_power_of_2(self.headdim)\n"
    "        self.softmax_segm_output = torch.empty(\n"
    "            (\n"
    "                self.seq_threshold_3D,\n"
    "                self.num_heads_q,\n"
    "                self.num_par_softmax_segments,\n"
    "                headdim_padded,\n"
    "            ),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.softmax_segm_max = torch.empty(\n"
    "            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.softmax_segm_expsum = torch.empty(\n"
    "            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n",
    "        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n"
    "        headdim_padded = next_power_of_2(self.headdim)\n"
    "        # radiance: 3D scratch rows are per query token (multi-token verify uses 3D too)\n"
    "        import os as _os\n"
    "        _segm_rows = max(\n"
    "            self.seq_threshold_3D,\n"
    "            int(_os.environ.get('RADIANCE_ATTN_3D_TOKENS', '128')),\n"
    "        )\n"
    "        self.softmax_segm_output = torch.empty(\n"
    "            (\n"
    "                _segm_rows,\n"
    "                self.num_heads_q,\n"
    "                self.num_par_softmax_segments,\n"
    "                headdim_padded,\n"
    "            ),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.softmax_segm_max = torch.empty(\n"
    "            (_segm_rows, self.num_heads_q, self.num_par_softmax_segments),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.softmax_segm_expsum = torch.empty(\n"
    "            (_segm_rows, self.num_heads_q, self.num_par_softmax_segments),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n",
    "radiance: 3D scratch rows are per query token",
    "triton_attn: 3D scratch sized per query token",
)

apply(
    SP / "vllm/v1/attention/ops/triton_unified_attention.py",
    "    use_3d = not (\n"
    "        seq_threshold_3D is None\n"
    "        or num_par_softmax_segments is None\n"
    "        or softmax_segm_output is None\n"
    "        or softmax_segm_max is None\n"
    "        or softmax_segm_expsum is None\n"
    "        or max_seqlen_q > 1\n"
    "        or num_seqs > seq_threshold_3D\n"
    "        or is_batch_invariant\n"
    "    )\n",
    "    # radiance: split-KV for short multi-token queries too (spec-decode verify, drafter\n"
    "    # catch-up). Scratch rows are per query token, so the batch's tokens must fit them.\n"
    "    _rad_max_q = int(__import__('os').environ.get('RADIANCE_ATTN_3D_MAX_Q', '16'))\n"
    "    use_3d = not (\n"
    "        seq_threshold_3D is None\n"
    "        or num_par_softmax_segments is None\n"
    "        or softmax_segm_output is None\n"
    "        or softmax_segm_max is None\n"
    "        or softmax_segm_expsum is None\n"
    "        or max_seqlen_q > _rad_max_q\n"
    "        or (max_seqlen_q > 1 and q.shape[0] > softmax_segm_output.shape[0])\n"
    "        or num_seqs > seq_threshold_3D\n"
    "        or is_batch_invariant\n"
    "    )\n",
    "radiance: split-KV for short multi-token queries",
    "unified_attention: 3D kernel for short multi-token queries",
)
