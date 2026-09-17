#!/usr/bin/env python3
"""Lazy GDN state snapshots under speculative decode (RADIANCE_GDN_LAZY=1; radiance_gdn_lazy.py).

Three source patches, every one of them inert unless the env knob is set at runtime:
  1. mamba/abstract.py: the GDN MambaSpec asks for ONE speculative block (the candidate stash)
     instead of num_speculative_tokens -- this is where the per-request page count falls 9 -> 3.
  2. gdn_attn.py: the persistent spec-state-index buffer is 2 columns wide (running, stash), and the
     metadata carries the stash column for every row so a prefill can invalidate its stash.
  3. v1/worker/mamba_utils.py: the fused align copies skip TEMPORAL states (LAZY_TEMPORAL) and the
     lazy materialize kernel provides them (pre-forward migration after the Triton launch, post-step
     checkpoint BEFORE it, because the Triton kernel resets num_accepted in place).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"]) / "vllm"
SENT = "radiance lazy gdn"

# ---- 1. spec blocks -------------------------------------------------------------------------
apply(SP / "model_executor/layers/mamba/abstract.py",
'''            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
''',
'''            num_speculative_blocks=_radiance_lazy_spec_blocks(self, vllm_config),  # radiance lazy gdn
''', SENT, "abstract.get_kv_cache_spec: one stash block under RADIANCE_GDN_LAZY")
apply(SP / "model_executor/layers/mamba/abstract.py",
'''class MambaBase(AttentionLayerBase):''',
'''def _radiance_lazy_spec_blocks(layer, vllm_config):
    """radiance lazy gdn: a lazy cache keeps one stash block per request instead of one block per
    draft token (radiance_gdn_lazy.py). GDN layers only; everything else keeps the stock count."""
    import os as _os
    n = (vllm_config.speculative_config.num_speculative_tokens
         if vllm_config.speculative_config else 0)
    if n > 0 and _os.environ.get("RADIANCE_GDN_LAZY", "0") == "1" \\
            and "GDN" in str(getattr(layer.mamba_type, "name", layer.mamba_type)).upper():
        return 1
    return n


class MambaBase(AttentionLayerBase):''', SENT + " helper", "abstract: lazy spec-block helper")

# ---- 2. builder ------------------------------------------------------------------------------
G = SP / "v1/attention/backends/gdn_attn.py"
apply(G,
'''    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
''',
'''    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    # radiance lazy gdn: the stash block of every batch row (window column 1), for prefill
    # invalidation. None unless RADIANCE_GDN_LAZY=1.
    radiance_stash_indices: torch.Tensor | None = None
''', SENT + " field", "gdn_attn: stash-index metadata field")
apply(G,
'''        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
''',
'''        import os as _rl_os  # radiance lazy gdn: 2 state columns (running, stash)
        self._rad_lazy = _rl_os.environ.get("RADIANCE_GDN_LAZY", "0") == "1"
        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, 2 if self._rad_lazy else self.num_spec + 1),
''', SENT + " width", "gdn_attn: 2-wide spec-state buffer under lazy")
# shared-build fast path: the per-group block table is `bt`; attach its column 1
apply(G,
'''            self.num_accepted_tokens[:bs].copy_(sh["acc_src"], non_blocking=True)
            return _dc.replace(
                sh["md"],
''',
'''            self.num_accepted_tokens[:bs].copy_(sh["acc_src"], non_blocking=True)
            return _dc.replace(
                sh["md"],
                radiance_stash_indices=(bt[:bs, 1] if self._rad_lazy else None),  # radiance lazy gdn
''', SENT + " fast", "gdn_attn: stash indices on the shared-build path")
apply(G,
'''        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
''',
'''        attn_metadata = GDNAttentionMetadata(
            radiance_stash_indices=(block_table_tensor[:, 1] if self._rad_lazy else None),  # radiance lazy gdn
            num_prefills=num_prefills,
''', SENT + " full", "gdn_attn: stash indices on the full build path")

# ---- 3. copies -------------------------------------------------------------------------------
M = SP / "v1/worker/mamba_utils.py"
apply(M,
'''    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
):
''',
'''    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
    LAZY_TEMPORAL: tl.constexpr = False,  # radiance lazy gdn: temporal states are materialised elsewhere
):
''', SENT + " post-sig", "mamba_utils: LAZY_TEMPORAL on the postprocess kernel")
apply(M,
'''    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
''', SENT + " post-skip", "mamba_utils: postprocess skips temporal states under lazy")
apply(M,
'''    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = True,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
'''    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = True,
    LAZY_TEMPORAL: tl.constexpr = False,  # radiance lazy gdn
):
    """Pre-copy mamba "align" state across block boundaries.
''', SENT + " pre-sig", "mamba_utils: LAZY_TEMPORAL on the precopy kernel")
apply(M,
'''    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
''', SENT + " pre-skip", "mamba_utils: precopy skips temporal states under lazy")
# remember what the tables need
apply(M,
'''        if self.is_initialized:
            return

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
''',
'''        if self.is_initialized:
            return
        # radiance lazy gdn: the materialize tables are built from these on first use
        self._radiance_kv_cfg = kv_cache_config
        self._radiance_fwd_ctx = forward_context
        self._radiance_copy_funcs = mamba_state_copy_funcs

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
''', SENT + " tables", "mamba_utils: keep the forward context for the lazy tables")
# V1 postprocess: not supported under lazy
apply(M,
'''        if num_reqs == 0 or not self.is_initialized:
            return

        # Initialize output to current values (unchanged unless src==dst)
''',
'''        if num_reqs == 0 or not self.is_initialized:
            return
        if _radiance_lazy():  # radiance lazy gdn
            raise RuntimeError("RADIANCE_GDN_LAZY needs the V2 model runner (align postprocess)")

        # Initialize output to current values (unchanged unless src==dst)
''', SENT + " v1", "mamba_utils: V1 postprocess refuses lazy")
# precopy: Triton (conv only) then materialize
apply(M,
'''            idx_mapping,
            num_reqs,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=idx_mapping is not None,
        )
''',
'''            idx_mapping,
            num_reqs,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=idx_mapping is not None,
            LAZY_TEMPORAL=_radiance_lazy(),
        )
        if _radiance_lazy():  # radiance lazy gdn: temporal migration = base + replay
            import radiance_gdn_lazy
            radiance_gdn_lazy.materialize(self, 0, num_reqs, state_idx_gpu, src_col_gpu,
                                          token_bias_gpu, idx_mapping)
''', SENT + " precopy", "mamba_utils: lazy materialize after the precopy")
apply(M,
'''        if num_reqs == 0 or not self.is_initialized:
            return
        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            state_idx_gpu,
            None,  # num_scheduled: unused under PRECOMPUTED_NEW_COMPUTED
''',
'''        if num_reqs == 0 or not self.is_initialized:
            return
        if _radiance_lazy():  # radiance lazy gdn: checkpoint = base + replay, BEFORE the reset
            import radiance_gdn_lazy
            radiance_gdn_lazy.materialize(self, 1, num_reqs, num_accepted_tokens_gpu,
                                          state_idx_gpu, new_num_computed_tokens_gpu, idx_mapping)
        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            state_idx_gpu,
            None,  # num_scheduled: unused under PRECOMPUTED_NEW_COMPUTED
''', SENT + " postalign", "mamba_utils: lazy materialize before the align postprocess")
apply(M,
'''            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
        )
''',
'''            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
            LAZY_TEMPORAL=_radiance_lazy(),
        )
''', SENT + " postalign-flag", "mamba_utils: LAZY_TEMPORAL on the align postprocess launch")
apply(M,
'''def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:''',
'''def _radiance_lazy() -> bool:
    """radiance lazy gdn: RADIANCE_GDN_LAZY=1 (read once)."""
    import os as _os
    v = getattr(_radiance_lazy, "_v", None)
    if v is None:
        v = _radiance_lazy._v = _os.environ.get("RADIANCE_GDN_LAZY", "0") == "1"
    return v


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:''', SENT + " flag", "mamba_utils: lazy flag helper")
print("patch_gdn_lazy: done")
