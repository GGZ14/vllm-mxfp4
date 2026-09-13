#!/usr/bin/env python3
"""Audit the radiance patch set against another vLLM version, offline.

Every `patch_*.py` resolves its target through `sysconfig.get_paths()["purelib"]`, so pointing the
whole set at a different vLLM is just a matter of running them under an interpreter whose purelib
holds that version's source. Nothing is compiled and no GPU is touched: the patches are text edits
with an `ast.parse` guard, and the Python tree inside a vLLM sdist is platform-independent -- which
is why this works with no ROCm wheel for gfx1201 in existence.

Two passes per version:
  isolated    the tree is restored from pristine before each patch, so every failure is attributed
              to its own script and a half-applied multi-hunk patch cannot poison the next one.
              This pass is the break list.
  sequential  one tree, patches in the order the Dockerfile and launchers actually run them,
              continuing past failures. Catches ordering and interaction effects.

Run the version we ship first as a control: every patch must come back OK. A FAIL against the
version production is demonstrably running means the harness is wrong, not the patch.

Usage:
    ./audit_upgrade.py 0.27.1 0.29.0      # control, then candidate
    ./audit_upgrade.py --symbols 0.29.0   # also check the monkeypatch watchlist
"""
from __future__ import annotations

import argparse
import io
import os
import json
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
NON_VLLM_TREES = ("torch", "aiter", "triton", "transformers")
CONTAINER_SP = "/opt/vllm/lib/python3.12/site-packages"

# Monkeypatch targets: no anchor audit can see these, they fail at runtime. (symbol, where we
# expect it). Sourced from the modules that rebind or read each one.
WATCHLIST = [
    ("_apply_head",                "vllm/model_executor/layers/logits_processor.py"),
    ("_greedy_sample",             "vllm/v1/spec_decode/llm_base_proposer.py"),
    ("use_local_argmax_reduction", "vllm/v1/spec_decode/llm_base_proposer.py"),
    ("_is_per_token_head_quant",   "vllm/v1/attention/backends/triton_attn.py"),
    ("_encode_layer_name",         "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
    ("GDN_AITER_TRITON_AVAILABLE", "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
    ("qwen_gdn_attention_core",    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
    ("_output_projection",         "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
    ("NO_LOGPROBS",                "vllm/v1/worker/gpu/sample/states.py"),
    ("sampling_states",            "vllm/v1/worker/gpu/sample/sampler.py"),
    ("idx_mapping_np",             "vllm/v1/worker/gpu/input_batch.py"),
    ("propose_draft_token_ids",    "vllm/v1/worker/gpu_model_runner.py"),
    ("target_input_buffers",       "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"),
    ("compile_or_warm_up_model",   "vllm/v1/worker/gpu_worker.py"),
    ("register_quantization_config", "vllm/model_executor/layers/quantization/__init__.py"),
    ("MxFp4LinearKernel",          "vllm/model_executor/kernels/linear/mxfp4/base.py"),
    ("activation_quant_key",       "vllm/model_executor/kernels/linear/mxfp4/base.py"),
    ("kMxfp4Dynamic",              "vllm/model_executor/layers/quantization/utils/quant_utils.py"),
    ("on_gfx12x",                  "vllm/platforms/rocm.py"),
    ("is_conv_state_dim_first",    "vllm/model_executor/layers/mamba/mamba_utils.py"),
    ("class GDNAttentionMetadata", "vllm/v1/attention/backends/gdn_attn.py"),
    ("register_backend",           "vllm/v1/attention/backends/registry.py"),
    ("apply_sdpa",                 "vllm/v1/attention/ops/vit_attn_wrappers.py"),
    ("AiterRMSNormDynamicQuantPattern", "vllm/compilation/passes/fusion/rocm_aiter_fusion.py"),
    ("class CudaCommunicator",     "vllm/distributed/device_communicators/cuda_communicator.py"),
    # ours, threaded in by patch_topk_composite -- expected ABSENT upstream
    ("max_top_k",                  "vllm/v1/sample/metadata.py"),
]


def patch_order(repo: Path) -> dict[str, list[str]]:
    """The three ordered lists, parsed from the files that actually invoke them."""
    tiers: dict[str, list[str]] = {}
    m = re.search(r"for p in (patch_\w+.*?); do", (repo / "Dockerfile").read_text(), re.S)
    tiers["baked"] = [t for t in m.group(1).split()
                      if t.startswith(("patch_", "install_"))] if m else []
    for lane, path in (("serve", repo / "serve-mxfp4.sh"),
                       ("paro", repo / "paroquant" / "run_paroquant.sh")):
        tiers[lane] = re.findall(r"^\s*python3 (patch_\w+)\.py", path.read_text(), re.M)
    return tiers


def fetch_vllm(version: str, work: Path) -> Path:
    """The vllm/ package tree out of the PyPI sdist (cached)."""
    tree = work / f"vllm-{version}"
    if (tree / "vllm" / "__init__.py").exists():
        return tree / "vllm"
    src = work / "src"
    src.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/vllm/json", timeout=60))
    f = next(x for x in meta["releases"][version] if x["packagetype"] == "sdist")
    tarball = src / f["filename"]
    if not tarball.exists() or tarball.stat().st_size != f["size"]:
        print(f"  fetching {f['filename']} ({f['size'] / 1e6:.0f} MB)")
        urllib.request.urlretrieve(f["url"], tarball)
    tree.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as t:
        members = []
        for mem in t.getmembers():
            parts = mem.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("vllm/"):
                mem.name = parts[1]
                members.append(mem)
        t.extractall(tree, members=members, filter="data")
    return tree / "vllm"


def base_trees(work: Path, container: str) -> Path:
    """torch/aiter/triton/transformers .py trees, taken once from a running container.

    These are not what we are bumping, but four patches target them, and the control run has to be
    able to come back all-OK.
    """
    base = work / "base_sp"
    if (base / "torch" / "__init__.py").exists():
        return base
    base.mkdir(parents=True, exist_ok=True)
    trees = " ".join(NON_VLLM_TREES)
    print(f"  taking {trees} (.py only) from container {container}")
    p = subprocess.run(
        ["podman", "exec", container, "bash", "-lc",
         f"cd {CONTAINER_SP} && find {trees} -name '*.py' -print0 | tar --null -T - -cf -"],
        capture_output=True)
    if p.returncode != 0:
        sys.exit(f"cannot read site-packages from {container}: {p.stderr.decode()[:400]}")
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
        t.extractall(base, filter="data")
    return base


def pristine_tree(version: str, work: Path, container: str) -> Path:
    """A fake site-packages: real non-vLLM trees + this version's vllm + our own modules."""
    pristine = work / f"pristine-{version}"
    if (pristine / "vllm" / "__init__.py").exists():
        return pristine
    shutil.copytree(base_trees(work, container), pristine, dirs_exist_ok=True)
    shutil.copytree(fetch_vllm(version, work), pristine / "vllm", dirs_exist_ok=True)
    for mod in list(REPO.glob("radiance_*.py")) + list((REPO / "paroquant").glob("radiance_*.py")):
        shutil.copy2(mod, pristine / mod.name)
    return pristine


def make_venv(run: Path) -> tuple[Path, Path]:
    venv = run / "venv"
    if not (venv / "bin" / "python").exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    py = venv / "bin" / "python"
    purelib = subprocess.run([str(py), "-c",
                              "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True).stdout.strip()
    return py, Path(purelib)


def classify(p: subprocess.CompletedProcess) -> tuple[str, str]:
    out = (p.stdout or "") + (p.stderr or "")
    fail = next((l.strip() for l in out.splitlines() if "FAIL" in l), "")
    if p.returncode == 0:
        if "NOOP" in out and "  OK" not in out:
            return "NOOP", "sentinel already present -- upstream carries this change"
        return "OK", ""
    if "anchor matched" in fail:
        return "ANCHOR", fail
    if "missing" in fail:
        return "MISSING", fail
    if fail:
        return "FAIL", fail
    if "Traceback" in out:
        return "ERROR", out.strip().splitlines()[-1][:160]
    return "FAIL", (out.strip().splitlines()[-1][:160] if out.strip() else f"rc={p.returncode}")


def restore(src: Path, dst: Path) -> None:
    subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=True)


def run_pass(py: Path, sp: Path, restore_from: Path | None, scripts: list[str]) -> dict:
    """Run each patch. With `restore_from`, the tree is reset before each one (isolated pass)."""
    results = {}
    for stem in scripts:
        if restore_from is not None:
            restore(restore_from, sp)
        p = subprocess.run([str(py), f"{stem}.py"], cwd=REPO, capture_output=True, text=True)
        results[stem] = classify(p)
    return results


def stage_image(py: Path, sp: Path, pristine: Path, staged: Path, baked: list[str]) -> dict:
    """Apply the baked tier once and snapshot it: this is the image the launchers patch into.

    Auditing the runtime tier against pristine upstream instead would be wrong -- several runtime
    patches anchor on text a baked patch wrote.
    """
    restore(pristine, sp)
    results = run_pass(py, sp, None, baked)
    restore(sp, staged)
    return results


def symbol_audit(vllm_tree: Path) -> list[tuple[str, str, str]]:
    rows = []
    for sym, rel in WATCHLIST:
        expected = vllm_tree.parent / rel
        if expected.exists() and sym in expected.read_text(errors="ignore"):
            rows.append((sym, "OK", rel))
            continue
        hits = [str(f.relative_to(vllm_tree.parent))
                for f in vllm_tree.rglob("*.py") if sym in f.read_text(errors="ignore")]
        if hits:
            rows.append((sym, "MOVED", ", ".join(hits[:2]) + (" ..." if len(hits) > 2 else "")))
        else:
            rows.append((sym, "GONE", f"expected in {rel}"))
    return rows


def audit(version: str, work: Path, container: str, want_symbols: bool,
          want_hunks: bool = False) -> dict:
    print(f"\n{'=' * 78}\n  vLLM {version}\n{'=' * 78}")
    pristine = pristine_tree(version, work, container)
    run = work / f"run-{version}"
    run.mkdir(parents=True, exist_ok=True)
    py, sp = make_venv(run)
    tiers = patch_order(REPO)
    ordered, seen = [], set()
    for lane in ("baked", "serve", "paro"):
        for s in tiers[lane]:
            if s not in seen:
                seen.add(s)
                ordered.append(s)

    staged = run / "staged"
    print(f"\n-- staging the {len(tiers['baked'])} baked patches (image build) --")
    seq = stage_image(py, sp, pristine, staged, tiers["baked"])

    print(f"-- isolated pass ({len(ordered)} patches, tree restored before each) --")
    iso = dict(run_pass(py, sp, pristine, tiers["baked"]))
    runtime = [s for s in ordered if s not in tiers["baked"]]
    iso.update(run_pass(py, sp, staged, runtime))

    print("-- sequential pass (launcher order, from image state, no restore) --")
    for lane in ("serve", "paro"):
        restore(staged, sp)
        for stem, res in run_pass(py, sp, None, tiers[lane]).items():
            seq.setdefault(stem, res)

    where = {s: "+".join(l for l in ("baked", "serve", "paro") if s in tiers[l]) for s in ordered}
    bad = [s for s in ordered if iso[s][0] not in ("OK", "NOOP")]
    print(f"\n  {'patch':34} {'tier':12} {'isolated':9} {'seq':9} detail")
    print(f"  {'-' * 34} {'-' * 12} {'-' * 9} {'-' * 9} {'-' * 20}")
    for s in ordered:
        st, detail = iso[s]
        sq = seq.get(s, ("-", ""))[0]
        flag = "  " if st in ("OK", "NOOP") else "! "
        print(f"{flag}{s:34} {where[s]:12} {st:9} {sq:9} {detail[:90]}")
    print(f"\n  {len(ordered) - len(bad)}/{len(ordered)} apply cleanly; {len(bad)} need work")

    if want_hunks and bad:
        hunk_map(py, sp, pristine, staged, work, tiers, bad)

    if want_symbols:
        print(f"\n-- monkeypatch watchlist (runtime failures no anchor audit can see) --")
        for sym, st, detail in symbol_audit(pristine / "vllm"):
            flag = "  " if st == "OK" else "! "
            print(f"{flag}{sym:36} {st:6} {detail[:80]}")
    return iso



SHIM = '''import ast, sys
from pathlib import Path

def apply(path, anchor, new, sentinel, label):
    """Tolerant stand-in for _patchlib.apply: applies what applies, reports what does not,
    and never aborts -- so one pass yields every hunk's verdict instead of just the first miss."""
    if not path.exists():
        print(f"HUNK\\tMISSING\\t{label}", file=sys.stderr); return
    s = path.read_text()
    if sentinel in s:
        print(f"HUNK\\tNOOP\\t{label}", file=sys.stderr); return
    n = s.count(anchor)
    if n == 1:
        s = s.replace(anchor, new, 1)
        ast.parse(s)
        path.write_text(s)
        print(f"HUNK\\tOK\\t{label}", file=sys.stderr)
    else:
        print(f"HUNK\\tMISS({n}x)\\t{label}", file=sys.stderr)
'''


def hunk_map(py: Path, sp: Path, pristine: Path, staged: Path, work: Path,
             tiers: dict, only: list[str]) -> None:
    """Per-hunk verdicts. The anchor audit stops each patch at its first miss; this shims
    _patchlib so every hunk reports, which is what actually sizes the port."""
    shim_dir = work / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "_patchlib.py").write_text(SHIM)
    env = dict(os.environ, PYTHONPATH=str(shim_dir))
    print("\n-- per-hunk map (tolerant _patchlib; every hunk reports) --")
    for lane, base in (("baked", pristine), ("serve", staged), ("paro", staged)):
        if lane != "baked":
            restore(staged, sp)
        else:
            restore(pristine, sp)
        for stem in tiers[lane]:
            # `python patch_x.py` would put the repo (and the real _patchlib) at sys.path[0];
            # run_path with an explicit insert puts the shim first instead.
            p = subprocess.run(
                [str(py), "-c",
                 f"import sys, runpy; sys.path.insert(0, {str(shim_dir)!r}); "
                 f"runpy.run_path({stem + '.py'!r}, run_name='__main__')"],
                cwd=REPO, capture_output=True, text=True, env=env)
            hunks = [l.split("\t") for l in (p.stderr or "").splitlines() if l.startswith("HUNK")]
            if stem not in only or not hunks:
                continue
            bad = sum(1 for h in hunks if h[1].startswith("MISS") or h[1] == "MISSING")
            print(f"\n  {stem}  [{lane}]  {len(hunks)} hunks, {bad} broken")
            for h in hunks:
                flag = "  " if h[1] in ("OK", "NOOP") else "! "
                print(f"    {flag}{h[1]:10} {h[2][:88]}")
        if lane == "baked":
            restore(sp, staged)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("versions", nargs="+", help="vLLM versions; run the shipped one first as control")
    ap.add_argument("--work", default="/tmp/radiance-audit", help="scratch root")
    ap.add_argument("--from-container", default="vllmint5",
                    help="running container to take the non-vLLM .py trees from")
    ap.add_argument("--symbols", action="store_true", help="also run the monkeypatch watchlist")
    ap.add_argument("--hunks", action="store_true", help="per-hunk map for the broken patches")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    runs = {v: audit(v, work, a.from_container, a.symbols, a.hunks) for v in a.versions}

    if len(a.versions) > 1:
        control, *rest = a.versions
        for cand in rest:
            broke = [s for s in runs[cand]
                     if runs[control][s][0] in ("OK", "NOOP")
                     and runs[cand][s][0] not in ("OK", "NOOP")]
            adopted = [s for s in runs[cand] if runs[cand][s][0] == "NOOP"]
            print(f"\n{'=' * 78}\n  {control} -> {cand}: {len(broke)} regressions\n{'=' * 78}")
            for s in broke:
                print(f"  ! {s:34} {runs[cand][s][1][:110]}")
            if adopted:
                print("  upstream now carries (patch is deletable):")
                for s in adopted:
                    print(f"    - {s}")
            if not broke:
                print("  none -- the patch set applies unchanged")


if __name__ == "__main__":
    main()
