#!/usr/bin/env python3
"""Install the three radiance_tp3pad hooks (TP=3 via zero-weight dummy heads).

vLLM 0.27.1 cannot serve Qwen3.8-27B at --tensor-parallel-size 3: the attention, GDN and vocab
layers all divide head counts / vocab by the TP size and assert. radiance_tp3pad.py widens the
config and pads the checkpoint tensors at load; this patch only wires it in, at the three places
where a padded value has to exist before vLLM's own code reads it:

  vllm/config/model.py                  ModelConfig.__post_init__, right after hf_text_config is
                                        resolved. get_hf_text_config returns the NESTED config
                                        object, so mutating it there is seen by every later reader
                                        (target, MTP, and the DFlash2 drafter's own ModelConfig).
  model_loader/default_loader.py        DefaultModelLoader.load_weights: the weights iterator is
                                        wrapped BEFORE model.load_weights, i.e. before the sharding
                                        weight_loaders slice it per rank.
  layers/vocab_parallel_embedding.py    VocabParallelEmbedding.__init__: pad_vocab_size's multiple
                                        (64) becomes 192 so per-rank vocab divides by 3. The tail is
                                        zero-filled by the stock weight_loader and masked by the stock
                                        LogitsProcessor (org_vocab_size), so nothing else changes.

Every hook returns immediately unless RADIANCE_TP_PAD is set, so a TP 1/2 serve is byte-identical
with or without this patch. If radiance_tp3pad.py is missing from site-packages the hooks stay
silent too -- unless RADIANCE_TP_PAD is set, in which case the ImportError is fatal on purpose.

Idempotent, rerun-safe, fatal-loud on a moved anchor (a vLLM upgrade must not silently drop
padding while the launcher still asks for TP=3)."""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
MODEL_CFG = SP / "vllm/config/model.py"
LOADER = SP / "vllm/model_executor/model_loader/default_loader.py"
VOCAB = SP / "vllm/model_executor/layers/vocab_parallel_embedding.py"

CFG_ANCHOR = (
    "        self.hf_config = hf_config\n"
    "        if dict_overrides:\n"
    "            self._apply_dict_overrides(hf_config, dict_overrides)\n"
    "        self.hf_text_config = get_hf_text_config(self.hf_config)\n"
)
CFG_NEW = CFG_ANCHOR + (
    "        # --- radiance (patch_tp3_pad.py): TP=3 via dummy-head padding, RADIANCE_TP_PAD ---\n"
    "        # hf_text_config is the nested config object itself, so widening it here is seen by\n"
    "        # every later reader. Inert unless RADIANCE_TP_PAD is set.\n"
    "        try:\n"
    "            import radiance_tp3pad as _radiance_tp3pad\n"
    "        except ImportError:\n"
    "            import os as _os\n"
    "            if _os.environ.get(\"RADIANCE_TP_PAD\", \"\").strip() not in (\"\", \"0\"):\n"
    "                raise\n"
    "        else:\n"
    "            _radiance_tp3pad.maybe_pad_config(self.hf_config, self.hf_text_config)\n"
)

LOADER_ANCHOR = (
    "        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))\n"
)
LOADER_NEW = (
    "        # --- radiance (patch_tp3_pad.py): pad checkpoint tensors with dummy heads before the\n"
    "        # sharding weight loaders see them. Returns the iterator untouched unless this model's\n"
    "        # config was padded (RADIANCE_TP_PAD).\n"
    "        _weights = self.get_all_weights(model_config, model)\n"
    "        try:\n"
    "            import radiance_tp3pad as _radiance_tp3pad\n"
    "        except ImportError:\n"
    "            if os.environ.get(\"RADIANCE_TP_PAD\", \"\").strip() not in (\"\", \"0\"):\n"
    "                raise\n"
    "        else:\n"
    "            _weights = _radiance_tp3pad.pad_weights(_weights, model_config)\n"
    "        loaded_weights = model.load_weights(_weights)\n"
)

VOCAB_ANCHOR = (
    "        self.padding_size = padding_size\n"
    "        self.org_vocab_size = org_num_embeddings or num_embeddings\n"
)
VOCAB_NEW = (
    "        # --- radiance (patch_tp3_pad.py): pad the vocab to a multiple TP=3 divides (64 -> 192\n"
    "        # when RADIANCE_TP_PAD is set; unchanged otherwise). The padded tail is zero-filled by\n"
    "        # weight_loader below and masked by LogitsProcessor via org_vocab_size, as before.\n"
    "        try:\n"
    "            import radiance_tp3pad as _radiance_tp3pad\n"
    "        except ImportError:\n"
    "            import os as _os\n"
    "            if _os.environ.get(\"RADIANCE_TP_PAD\", \"\").strip() not in (\"\", \"0\"):\n"
    "                raise\n"
    "        else:\n"
    "            padding_size = _radiance_tp3pad.vocab_pad_multiple(padding_size)\n"
    "        self.padding_size = padding_size\n"
    "        self.org_vocab_size = org_num_embeddings or num_embeddings\n"
)


def main():
    apply(MODEL_CFG, CFG_ANCHOR, CFG_NEW, "patch_tp3_pad.py): TP=3 via dummy-head padding",
          "tp3pad: ModelConfig hook (maybe_pad_config)")
    # `os` is imported at the top of default_loader.py.
    apply(LOADER, LOADER_ANCHOR, LOADER_NEW, "patch_tp3_pad.py): pad checkpoint tensors",
          "tp3pad: DefaultModelLoader hook (pad_weights)")
    apply(VOCAB, VOCAB_ANCHOR, VOCAB_NEW, "patch_tp3_pad.py): pad the vocab",
          "tp3pad: VocabParallelEmbedding hook (vocab_pad_multiple)")


if __name__ == "__main__":
    main()
