#!/usr/bin/env python3
"""Shared helper for the radiance source patches.

Every `patch_*.py` applies an idempotent one-shot string replacement to an installed vLLM /
transformers source file; they all used a byte-identical `apply()`. It lives here once instead
of being copy-pasted into each patch. Each patch runs as `python patch_X.py` from the directory
that holds this file (the build's `/opt/patches`, or the repo root when run standalone), so
`from _patchlib import apply` resolves as a sibling import.
"""
import ast


def apply(path, anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new` in `path`. Skips if
    `sentinel` is already present; a missing file or non-unique anchor is fatal."""
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({path})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    path.write_text(s)
    print(f"  OK    {label}")


def apply_any(path, variants, sentinel, label):
    """`apply()` over several (anchor, new) shapes: the first that matches uniquely wins.

    The launchers patch the *shipped* image at container start while the Dockerfile builds against
    whatever vLLM is pinned, so one repo drives two versions at once. A patch re-anchored for a
    new vLLM therefore has to keep the old shape working, or the next production restart dies on
    an image that was fine a minute ago. Fatal only when no shape matches -- the counts are
    reported so a genuine drift is distinguishable from a version we simply do not carry.
    """
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    counts = []
    for i, (anchor, new) in enumerate(variants):
        n = s.count(anchor)
        counts.append(n)
        if n == 1:
            s = s.replace(anchor, new, 1)
            ast.parse(s)
            path.write_text(s)
            print(f"  OK    {label}" + (f" (shape {i + 1})" if i else ""))
            return
    raise SystemExit(f"  FAIL  {label}: no shape matched, counts {counts} ({path})")
