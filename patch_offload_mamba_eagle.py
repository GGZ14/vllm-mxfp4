#!/usr/bin/env python3
"""KV offloading on a hybrid model under speculative decoding: keep target groups loadable.

vLLM's OffloadingConnector scheduler excludes the "volatile" trailing chunk of EAGLE/MTP draft
attention groups from store and load. When speculative decoding is on but no KV cache group is
tagged `is_eagle_group` (the DFlash2 backport tags none), it falls back to marking EVERY group:

    if use_eagle and not eagle_groups:
        eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

A Mamba/GDN group holds one state chunk (its sliding window is 1 chunk), so dropping its trailing
chunk leaves nothing to restore. Every prefix hit on a hybrid model needs that state, so the tier
stores gigabytes and never loads: observed 2026-09-25 on one R9700, Qwen3.8-27B MXFP4 + DFlash2,
vLLM 0.27.1: GPU_to_CPU 12.95 GB, CPU_to_GPU 0 B, external_prefix_cache_hits 0 over 20 lookups.

Only draft-model KV is volatile. Target attention KV and GDN state at block boundaries are what
the GPU prefix cache already reuses under speculation. So the fallback marks only groups whose
layers sit past the target's num_hidden_layers (vLLM numbers draft layers after the target's);
if none are found it marks every non-Mamba group (conservative). Groups are logged either way.
Same fallback line is on upstream main as of 2026-09-25.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
path = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

apply(
    path,
    "        if use_eagle and not eagle_groups:\n"
    "            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))\n",
    "        if use_eagle and not eagle_groups:\n"
    "            # radiance: only draft-model KV is volatile. Marking Mamba/GDN groups drops\n"
    "            # their only chunk and makes the tier write-only on hybrid models.\n"
    "            import re as _re\n"
    "            _groups = kv_cache_config.kv_cache_groups\n"
    "            _n_target = getattr(\n"
    "                vllm_config.model_config.hf_text_config, 'num_hidden_layers', None)\n"
    "            def _layer_idx(name):\n"
    "                m = _re.search(r'layers\\.(\\d+)\\.', name)\n"
    "                return int(m.group(1)) if m else -1\n"
    "            _attn = [i for i, g in enumerate(_groups)\n"
    "                     if not isinstance(g.kv_cache_spec, MambaSpec)]\n"
    "            _draft = [i for i in _attn if _n_target is not None and any(\n"
    "                _layer_idx(n) >= _n_target for n in _groups[i].layer_names)]\n"
    "            eagle_groups = set(_draft or _attn)\n"
    "            for _i, _g in enumerate(_groups):\n"
    "                logger.info('radiance offload group %d %s %s', _i,\n"
    "                            type(_g.kv_cache_spec).__name__, _g.layer_names[:2])\n",
    "radiance: only draft-model KV is volatile",
    "offload: mark only draft groups volatile",
)
