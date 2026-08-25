#!/usr/bin/env python3
"""Collect per-input-channel activation statistics for the DFlash2 drafter's linears.

AWQ needs to know which input channels carry large activations, and for a drafter that is not
recoverable offline: its inputs are the TARGET model's hidden states at layers [5,19,33,47,61]
(for `fc`) and its own internal activations (everything else). So the statistics come from a live
serve, via a forward pre-hook on each linear we intend to scale.

RUN IT EAGER. The drafter's whole point is replaying one captured CUDA graph, and a Python
pre-hook does not run on graph replay -- it fires during warmup and then silently never again,
which yields statistics from a handful of profile steps and looks like success. Pass
EXTRA="--enforce-eager".

Inert unless RADIANCE_DFLASH_CALIB names an output path. Statistics are a running per-channel
absmax plus a sum of |x| and a token count, so the consumer can pick either; both are accumulated
in fp32 on the GPU and moved to CPU when dumped. The dump fires on a token count as well as at
exit, because `podman stop` can SIGKILL the worker before atexit runs and losing a calibration
pass to that is silent.

The method and its call site are inserted at ONE anchor on purpose. Splitting them across two
anchors put the method on DFlashQwen3ForCausalLM (which owns the file's unique `load_weights`)
while the call ran in DFlashQwen3Model.__init__, and the model died at load with
`'DFlash2Qwen3Model' object has no attribute '_radiance_install_calib_hooks'`.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
DF = SP / "vllm" / "model_executor" / "models" / "qwen3_dflash.py"

ANCHOR = """        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
"""

NEW = '''        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self._radiance_install_calib_hooks()

    def _radiance_install_calib_hooks(self):
        """Per-input-channel activation accumulators on the linears AWQ will scale."""
        import atexit, os, sys
        import torch as _t
        path = os.environ.get("RADIANCE_DFLASH_CALIB")
        if not path:
            return
        stats = {}
        limit = int(os.environ.get("RADIANCE_DFLASH_CALIB_TOKENS", "200000"))
        done = []
        nhook = []

        def _tag():
            # Both TP ranks must be kept, and separately: qkv_proj / gate_up_proj are
            # column-parallel so their INPUT is unsharded and either rank has the whole story,
            # but o_proj / down_proj are row-parallel and each rank only ever sees half of their
            # input channels. The consumer concatenates those two halves.
            try:
                from vllm.distributed import get_tensor_model_parallel_rank as _r
                return "tp%d" % _r()
            except Exception:
                return "pid%d" % os.getpid()

        def dump():
            if not stats:
                sys.stderr.write("[radiance.calib] NO STATS -- hooks never fired; was the "
                                 "drafter replaying a CUDA graph? Use --enforce-eager.\\n")
                return
            out = {k: {"absmax": v[0].cpu(), "absmean": (v[1] / max(v[2], 1)).cpu(),
                       "tokens": v[2], "in_features": int(v[0].numel())}
                   for k, v in stats.items()}
            p = f"{path}.{_tag()}.pt"
            _t.save(out, p)
            sys.stderr.write(f"[radiance.calib] wrote {len(out)} tensors to {p} "
                             f"({next(iter(stats.values()))[2]} tokens)\\n")

        def make_hook(name):
            def hook(_mod, args):
                x = args[0]
                if not _t.is_tensor(x):
                    return
                xf = x.detach().reshape(-1, x.shape[-1]).float()
                a = xf.abs()
                s = stats.get(name)
                if s is None:
                    stats[name] = s = [a.amax(0), a.sum(0), 0]
                else:
                    _t.maximum(s[0], a.amax(0), out=s[0])
                    s[1] += a.sum(0)
                s[2] += int(xf.shape[0])
                # Wait until EVERY hooked linear has enough rows. Triggering on a single layer's
                # count dumps during the profile run -- one 8192-token forward through `fc` alone
                # crossed a 2000-row threshold and wrote a file containing exactly one tensor.
                if (not done and len(stats) >= nhook[0]
                        and min(v[2] for v in stats.values()) >= limit):
                    done.append(1)
                    dump()
            return hook

        # vLLM fuses q/k/v into qkv_proj and gate/up into gate_up_proj. That is also the right
        # granularity for AWQ: fused layers share one input, so they must share one input-channel
        # scale. Matching the checkpoint's separate q_proj/k_proj/v_proj names here would hook
        # nothing -- the first attempt caught 11 of the 21 linears for exactly that reason.
        want = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "fc")
        n = 0
        for name, mod in self.named_modules():
            if name.rsplit(".", 1)[-1] in want and hasattr(mod, "weight"):
                mod.register_forward_pre_hook(make_hook(name))
                n += 1
        nhook.append(n)
        sys.stderr.write(f"[radiance.calib] hooks on {n} drafter linears -> "
                         f"{path}.{_tag()}.pt (dump when all reach {limit} rows)\\n")
        atexit.register(dump)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
'''


def main():
    apply(DF, ANCHOR, NEW, "_radiance_install_calib_hooks",
          "dflash: activation-statistics hooks for AWQ calibration")


if __name__ == "__main__":
    main()
