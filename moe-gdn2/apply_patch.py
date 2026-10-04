#!/usr/bin/env python3
"""Apply a one-file unified diff in place: exact context, no fuzz, no offset search.

moe-gdn2/build.sh uses it because the vllm-radiance image ships neither `patch` nor `git`. It only has to
handle radiance_gdn2_vs_v050.patch, which is generated against one known file (build.sh checks that file's
md5 before and the result's md5 after), so any mismatch is fatal rather than guessed around.

    apply_patch.py TARGET PATCH
"""
import re
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def die(msg):
    raise SystemExit(f"  FAIL  apply_patch: {msg}")


def main(target, patch):
    src = Path(target).read_text().splitlines(keepends=True)
    lines = Path(patch).read_text().splitlines(keepends=True)
    out, pos, i, hunks = [], 0, 0, 0
    while i < len(lines) and not HUNK.match(lines[i]):
        i += 1  # skip the ---/+++ header
    while i < len(lines):
        m = HUNK.match(lines[i])
        if not m:
            die(f"expected a hunk header at patch line {i + 1}: {lines[i][:60]!r}")
        old_start, old_n = int(m.group(1)), int(m.group(2) or 1)
        start = old_start if old_n == 0 else old_start - 1
        if start < pos:
            die(f"hunk at old line {old_start} overlaps the previous one")
        out.extend(src[pos:start])
        pos, i, hunks = start, i + 1, hunks + 1
        seen_old = 0
        while i < len(lines) and not HUNK.match(lines[i]):
            tag, body = lines[i][:1], lines[i][1:]
            if tag in (" ", "-"):
                if pos >= len(src) or src[pos] != body:
                    die(f"hunk at old line {old_start}: context mismatch at source line {pos + 1}")
                if tag == " ":
                    out.append(src[pos])
                pos, seen_old = pos + 1, seen_old + 1
            elif tag == "+":
                out.append(body)
            elif tag == "\\":
                die("'\\ No newline at end of file' markers are not supported")
            else:
                die(f"unexpected patch line {i + 1}: {lines[i][:60]!r}")
            i += 1
        if seen_old != old_n:
            die(f"hunk at old line {old_start}: covers {seen_old} old lines, header says {old_n}")
    if not hunks:
        die("no hunks in the patch")
    out.extend(src[pos:])
    Path(target).write_text("".join(out))
    print(f"  OK    apply_patch: {hunks} hunks applied to {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
