#!/usr/bin/env python3
"""DFlash2 speculative decoding, backported to vLLM 0.27.1.

DFlash2 is a block-diffusion drafter: one backbone pass proposes a whole block of positions at
once, keeps the target head's top-K candidates at every position, and a selector walks one path
through them. vLLM gained it in vllm-project/vllm#52816, which merged ten days after 0.27.1 was
tagged, so a 0.27.1 tree has DFlash1 and nothing that reads a `DFlash2DraftModel` checkpoint. This
carries that PR back.

Two pieces are new source rather than edits, and are installed from the sibling dflash2/ directory:

    qwen3_dflash2.py   the grouped dynamic depthwise convolution and the candidate selector
    speculator.py      the path-walk kernel and the proposal, as vllm/v1/worker/gpu/spec_decode/
                       dflash2/speculator.py

The seven edits below are the seams those two need. In upstream order:

  1. CAUSALITY. A DFlash2 config states `is_causal` at the top level; 0.27.1 only reads
     `dflash_config.causal` and otherwise infers causality from the layer type. Every layer in this
     checkpoint is `sliding_attention`, which that inference calls causal -- the opposite of what a
     block-diffusion drafter wants, and a silently worse draft rather than an error.
  2. SUBCLASS SEAMS. `decoder_layer_cls` and `model_cls` on the DFlash1 model and its wrapper, so
     DFlash2 substitutes its own layer and model instead of copying 700 lines to change two names.
  3. THE REGISTRY. `DFlash2DraftModel` -> the new module.
  4. VOCAB-PARALLEL TOP-K. `LogitsProcessor.get_top_k_tokens` widens the existing single-token
     `get_top_tokens` reduction to K, so the selector's candidates cost O(batch * 2K * tp_size) of
     communication instead of an all-gather of the whole vocabulary (248320 columns here, per
     position, per step). FlashInfer's radix kernel is CUDA-only, so on ROCm this is torch.topk.
  5. GUMBEL. `gumbel_noised_argmax`, the noised-argmax body as a callable the selector's walk can
     use, so a draft and its verification draw the same noise. Upstream also rewrites
     `gumbel_block_argmax` to call it; that half is NOT applied, because 0.27.1's copy of that
     function differs from the one the PR was written against and deduplicating it here would be a
     rewrite of a hot sampling kernel for tidiness. The two copies agree.
  6. SPECULATOR SELECTION. The `dflash` arm of `init_speculator` splits on the checkpoint's
     architecture. Upstream also forces the V2 model runner for a DFlash2 draft (below), and both
     halves are needed: reaching the V1 proposer with a DFlash2 checkpoint drafts as DFlash1
     without raising.
  7. DRAFT LOGIT CACHE. `draft_logits_spec` lets a speculator say how its cached proposal
     distribution is shaped. DFlash2 writes only K columns per row, so it needs fp32 filled with
     -inf rather than a zeroed buffer -- an unwritten column at 0.0 is a uniform-weight candidate
     the rejection sampler would treat as real. The base returns 0.27.1's existing float32/0.0, so
     every other speculator keeps the buffer it has today. Note the PR's version of this returns
     `model_config.head_dtype`, which 0.27.1's DraftModelSpeculator does not use here.

The V2 model runner is where the DFlash2 speculator lives, and Qwen3.8 is not one of the
architectures that defaults to it, so the config edit that forces V2 for a DFlash2 draft is load
bearing. That also means a DFlash2 serve does not run the V1 draft path, and the RADIANCE dynamic
drafting, n-gram and 2-bit draft head hooks -- all of which attach to the V1 runner's MTP proposer
-- are inert under it. They are left installed and simply never fire.

Every edit is idempotent and anchors on source that would have to change for the replacement to be
wrong, so drift shows up as a failed build rather than as a silent divergence."""
import shutil
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
CFG = SP / "vllm/config/vllm.py"
LP = SP / "vllm/model_executor/layers/logits_processor.py"
DF1 = SP / "vllm/model_executor/models/qwen3_dflash.py"
REG = SP / "vllm/model_executor/models/registry.py"
GUM = SP / "vllm/v1/worker/gpu/sample/gumbel.py"
SEL = SP / "vllm/v1/worker/gpu/spec_decode/__init__.py"
SPEC = SP / "vllm/v1/worker/gpu/spec_decode/speculator.py"

# --- 1. causality: honour a top-level is_causal -----------------------------------------------
DF1_CAUSAL_OLD = (
    '    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""\n'
    "    override = (getattr(config, \"dflash_config\", None) or {}).get(\"causal\")\n"
    "    if override is not None:\n"
    "        return override\n"
)
DF1_CAUSAL_NEW = (
    '    """Resolve explicit causality before falling back to legacy layer defaults."""\n'
    "    is_causal = getattr(config, \"is_causal\", None)\n"
    "    if is_causal is not None:\n"
    "        return bool(is_causal)\n"
    "    override = (getattr(config, \"dflash_config\", None) or {}).get(\"causal\")\n"
    "    if override is not None:\n"
    "        return bool(override)\n"
)

# --- 2. subclass seams on the DFlash1 model and its wrapper -----------------------------------
DF1_LAYERCLS_OLD = (
    "@support_torch_compile\n"
    "class DFlashQwen3Model(nn.Module):\n"
    "    hf_to_vllm_mapper = WeightsMapper(\n"
)
DF1_LAYERCLS_NEW = (
    "@support_torch_compile\n"
    "class DFlashQwen3Model(nn.Module):\n"
    "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
    "\n"
    "    hf_to_vllm_mapper = WeightsMapper(\n"
)
DF1_LAYERUSE_OLD = (
    "        self.layers = nn.ModuleList(\n"
    "            [\n"
    "                DFlashQwen3DecoderLayer(\n"
)
DF1_LAYERUSE_NEW = (
    "        self.layers = nn.ModuleList(\n"
    "            [\n"
    "                self.decoder_layer_cls(\n"
)
DF1_MODELCLS_OLD = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
)
DF1_MODELCLS_NEW = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    model_cls = DFlashQwen3Model\n"
    "\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
)
DF1_MODELUSE_OLD = (
    "        self.model = DFlashQwen3Model(\n"
    "            vllm_config=vllm_config,\n"
)
DF1_MODELUSE_NEW = (
    "        self.model = self.model_cls(\n"
    "            vllm_config=vllm_config,\n"
)

# --- 3. the registry --------------------------------------------------------------------------
REG_OLD = '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
REG_NEW = (
    '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
    '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n'
)

# --- 4. vocab-parallel top-K ------------------------------------------------------------------
LP_IMPORT_OLD = (
    '"""A layer that compute logits from hidden_stats."""\n'
    "\n"
    "import torch\n"
)
LP_IMPORT_NEW = (
    '"""A layer that compute logits from hidden_stats."""\n'
    "\n"
    "from collections.abc import Callable\n"
    "from functools import cache\n"
    "\n"
    "import torch\n"
)
LP_TOPK_OLD = (
    "from vllm.platforms import current_platform\n"
)
LP_TOPK_NEW = (
    "from vllm.platforms import current_platform\n"
    "from vllm.utils.flashinfer import has_flashinfer\n"
    "\n"
    "\n"
    "@cache\n"
    "def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:\n"
    '    """FlashInfer\'s radix top-k, or None for torch.topk."""\n'
    "    if not current_platform.is_cuda():\n"
    "        return None\n"
    "    if not has_flashinfer():\n"
    "        return None\n"
    "    from flashinfer import top_k\n"
    "\n"
    "    return top_k\n"
    "\n"
    "\n"
    "def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    impl = _flashinfer_topk()\n"
    "    if impl is None or not scores.is_cuda:\n"
    "        return torch.topk(scores, k, dim=-1)\n"
    "    return impl(scores, k, sorted=True, deterministic=True)\n"
)
LP_METHOD_OLD = (
    "    def extra_repr(self) -> str:\n"
    '        s = f"vocab_size={self.vocab_size}"\n'
)
LP_METHOD_NEW = (
    "    def get_top_k_tokens(\n"
    "        self,\n"
    "        lm_head: VocabParallelEmbedding,\n"
    "        hidden_states: torch.Tensor,\n"
    "        k: int,\n"
    "        embedding_bias: torch.Tensor | None = None,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
    '        """Vocab-parallel top-k without all-gathering full logits.\n'
    "\n"
    "        The `get_top_tokens` reduction widened from one token to k, returning\n"
    "        the values as well as the global ids. Communication is\n"
    "        O(batch * 2k * tp_size) rather than O(batch * vocab_size).\n"
    "\n"
    "        Scale and soft cap are applied to the k selected values rather than\n"
    "        the whole vocabulary; both are monotonic, so the selection is the same\n"
    "        and only k entries are touched.\n"
    '        """\n'
    "        if self.scale <= 0.0 and self.scale != 1.0:\n"
    "            raise ValueError(\n"
    '                "The local top-k reduction optimization is not supported for "\n'
    '                "non-positive logit scaling factors."\n'
    "            )\n"
    "\n"
    "        logits = self._apply_head(lm_head, hidden_states, embedding_bias)\n"
    "\n"
    "        # Mask out padding entries beyond org_vocab_size on this shard.\n"
    "        num_pad = lm_head.shard_indices.num_org_vocab_padding\n"
    "        if num_pad > 0:\n"
    "            logits[..., -num_pad:] = -float(\"inf\")\n"
    "\n"
    "        values, ids = _topk(logits, k)\n"
    "        # Convert shard-local indices to global vocab indices.\n"
    "        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index\n"
    "\n"
    "        if lm_head.tp_size > 1:\n"
    "            values = tensor_model_parallel_all_gather(values, dim=-1)\n"
    "            ids = tensor_model_parallel_all_gather(ids, dim=-1)\n"
    "            values, selected = _topk(values, k)\n"
    "            ids = ids.gather(-1, selected)\n"
    "\n"
    "        values = values.float()\n"
    "        if self.scale != 1.0:\n"
    "            values = values * self.scale\n"
    "        if self.soft_cap is not None:\n"
    "            values = torch.tanh(values / self.soft_cap) * self.soft_cap\n"
    "        return ids, values\n"
    "\n"
    "    def extra_repr(self) -> str:\n"
    '        s = f"vocab_size={self.vocab_size}"\n'
)

# --- 5. the noised argmax as a callable -------------------------------------------------------
GUM_OLD = (
    "@triton.jit\n"
    "def gumbel_block_argmax(\n"
)
GUM_NEW = (
    "@triton.jit\n"
    "def gumbel_noised_argmax(\n"
    "    logits,\n"
    "    keys,\n"
    "    mask,\n"
    "    seed,\n"
    "    pos,\n"
    "    temp,\n"
    "    USE_FP64: tl.constexpr,\n"
    "    APPLY_TEMPERATURE: tl.constexpr = True,\n"
    "):\n"
    '    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.\n'
    "\n"
    "    `keys` indexes the noise, so the same token draws the same noise wherever it\n"
    "    appears; `pos` and `seed` place the draw in the request's stream, which is\n"
    "    what lets a draft and its verification agree.\n"
    '    """\n'
    "    if temp != 0.0 and APPLY_TEMPERATURE:\n"
    "        # Match the behavior of _temperature_kernel: if that kernel uses\n"
    "        # tl.div_rn, this must too.\n"
    "        logits = logits / temp\n"
    "\n"
    "    if USE_FP64:\n"
    "        logits = logits.to(tl.float64)\n"
    "    if temp != 0.0:\n"
    "        gumbel_seed = tl.randint(seed, pos)\n"
    "        if USE_FP64:\n"
    "            u = tl_rand64(gumbel_seed, keys, includes_zero=False)\n"
    "            gumbel_noise = -tl.log(-tl.log(u))\n"
    "        else:\n"
    "            u = tl_rand32(gumbel_seed, keys, includes_zero=False)\n"
    "            # log1p keeps the winning tail at u -> 0, where fp32 resolves it.\n"
    "            gumbel_noise = -tl.log(-tldevice.log1p(-u))\n"
    "        logits = tl.where(mask, logits + gumbel_noise, float(\"-inf\"))\n"
    "\n"
    "    return tl.max(logits, axis=0, return_indices=True)\n"
    "\n"
    "\n"
    "@triton.jit\n"
    "def gumbel_block_argmax(\n"
)

# --- 6. speculator selection and the V2 model runner ------------------------------------------
SEL_OLD = (
    '    if speculative_config.method == "dflash":\n'
    "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n"
)
SEL_NEW = (
    '    if speculative_config.method == "dflash":\n'
    '        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:\n'
    "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
    "                DFlash2Speculator,\n"
    "            )\n"
    "\n"
    "            return DFlash2Speculator(vllm_config, device)\n"
    "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n"
)
CFG_BRANCH_OLD = (
    "        if self._dflash_needs_multi_kv_group():\n"
    "            return True\n"
)
CFG_BRANCH_NEW = (
    "        if self._dflash_needs_multi_kv_group():\n"
    "            return True\n"
    "\n"
    "        # The DFlash2 candidate selector exists only in the V2 speculator. On V1\n"
    "        # the same checkpoint drafts through DFlashProposer, which never calls\n"
    "        # it, so the draft degrades to DFlash1 silently. Force V2 as for dspark.\n"
    "        if self._is_dflash2_draft():\n"
    "            return True\n"
)
CFG_HELPER_OLD = (
    "    def _dflash_needs_multi_kv_group(self) -> bool:\n"
)
CFG_HELPER_NEW = (
    "    def _is_dflash2_draft(self) -> bool:\n"
    '        """Whether the DFlash draft is a DFlash2 one, by the architecture the\n'
    '        speculator selects on (v1/worker/gpu/spec_decode/__init__.py)."""\n'
    "        spec = self.speculative_config\n"
    '        if spec is None or spec.method != "dflash":\n'
    "            return False\n"
    '        draft_config = getattr(spec, "draft_model_config", None)\n'
    "        if draft_config is None:\n"
    "            return False\n"
    '        return "DFlash2DraftModel" in (draft_config.architectures or [])\n'
    "\n"
    "    def _dflash_needs_multi_kv_group(self) -> bool:\n"
)

# --- 7. the cached proposal distribution ------------------------------------------------------
SPEC_BUF_OLD = (
    "            self.draft_logits = torch.zeros(\n"
    "                self.max_num_reqs,\n"
    "                self.num_speculative_steps,\n"
    "                self.vocab_size,\n"
    "                dtype=torch.float32,\n"
    "                device=device,\n"
    "            )\n"
)
SPEC_BUF_NEW = (
    "            dtype, fill = self.draft_logits_spec(vllm_config)\n"
    "            self.draft_logits = torch.full(\n"
    "                (\n"
    "                    self.max_num_reqs,\n"
    "                    self.num_speculative_steps,\n"
    "                    self.vocab_size,\n"
    "                ),\n"
    "                fill,\n"
    "                dtype=dtype,\n"
    "                device=device,\n"
    "            )\n"
)
SPEC_HOOK_OLD = (
    "    def _validate_local_argmax_reduction(self) -> None:\n"
)
SPEC_HOOK_NEW = (
    "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n"
    '        """Dtype and fill for the cached proposal distribution.\n'
    "\n"
    "        Speculators that write only a subset of columns each step override this.\n"
    '        """\n'
    "        return torch.float32, 0.0\n"
    "\n"
    "    def _validate_local_argmax_reduction(self) -> None:\n"
)


def install_modules():
    """Drop in the two upstream modules that are new files rather than edits."""
    here = Path(__file__).resolve().parent / "dflash2"
    targets = [
        (here / "qwen3_dflash2.py", SP / "vllm/model_executor/models/qwen3_dflash2.py"),
        (here / "speculator.py", SP / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"),
    ]
    pkg = SP / "vllm/v1/worker/gpu/spec_decode/dflash2"
    pkg.mkdir(exist_ok=True)
    init = pkg / "__init__.py"
    if not init.exists():
        init.write_text(
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        )
    for src, dst in targets:
        if not src.exists():
            raise SystemExit(f"  FAIL  DFlash2 module: {src} missing")
        shutil.copyfile(src, dst)
        print(f"  OK    installed {dst.relative_to(SP)}")


def main():
    apply(DF1, DF1_CAUSAL_OLD, DF1_CAUSAL_NEW, "is_causal = getattr(config",
          "DFlash draft honours a top-level is_causal")
    apply(DF1, DF1_LAYERCLS_OLD, DF1_LAYERCLS_NEW, "decoder_layer_cls = DFlashQwen3DecoderLayer",
          "DFlash model decoder-layer seam")
    apply(DF1, DF1_LAYERUSE_OLD, DF1_LAYERUSE_NEW, "self.decoder_layer_cls(",
          "DFlash model builds layers through the seam")
    apply(DF1, DF1_MODELCLS_OLD, DF1_MODELCLS_NEW, "model_cls = DFlashQwen3Model",
          "DFlash wrapper model seam")
    apply(DF1, DF1_MODELUSE_OLD, DF1_MODELUSE_NEW, "self.model = self.model_cls(",
          "DFlash wrapper builds the model through the seam")
    apply(REG, REG_OLD, REG_NEW, "DFlash2DraftModel", "DFlash2 architecture -> qwen3_dflash2")
    apply(LP, LP_IMPORT_OLD, LP_IMPORT_NEW, "from functools import cache",
          "logits processor top-k imports")
    apply(LP, LP_TOPK_OLD, LP_TOPK_NEW, "def _flashinfer_topk(", "vocab-parallel top-k helper")
    apply(LP, LP_METHOD_OLD, LP_METHOD_NEW, "def get_top_k_tokens(",
          "LogitsProcessor.get_top_k_tokens")
    apply(GUM, GUM_OLD, GUM_NEW, "def gumbel_noised_argmax(", "gumbel noised argmax as a callable")
    apply(SEL, SEL_OLD, SEL_NEW, "DFlash2Speculator", "select the DFlash2 speculator")
    apply(CFG, CFG_BRANCH_OLD, CFG_BRANCH_NEW, "if self._is_dflash2_draft():",
          "a DFlash2 draft forces the V2 model runner")
    apply(CFG, CFG_HELPER_OLD, CFG_HELPER_NEW, "def _is_dflash2_draft(",
          "DFlash2 draft detection")
    apply(SPEC, SPEC_BUF_OLD, SPEC_BUF_NEW, "self.draft_logits_spec(vllm_config)",
          "cached proposal distribution is speculator-shaped")
    apply(SPEC, SPEC_HOOK_OLD, SPEC_HOOK_NEW, "def draft_logits_spec(",
          "draft_logits_spec default")
    install_modules()


if __name__ == "__main__":
    main()
