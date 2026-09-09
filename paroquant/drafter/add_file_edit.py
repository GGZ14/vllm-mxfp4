"""Append file-edit prompts (old file + change request) to a prompts file. Tries nuprl/EditPackFT,
then commitpackft's raw jsonl files, then falls back to synthesizing edit tasks over CodeAlpaca."""
import json, random, sys
from datasets import load_dataset
out, want = sys.argv[1], int(sys.argv[2]); rng = random.Random(5); rows = []; MAXC = 7000
def mk(old, msg, lang, fname):
    return (f"Here is the current content of `{fname}`:\n\n```{lang}\n{old}\n```\n\nMake this change: {msg}\n\n"
            "Return the complete updated file in a single code block, followed by a one-line note on what changed.")
try:
    ds = load_dataset("nuprl/EditPackFT", split="train", streaming=True)
    for ex in ds:
        old, msg = ex.get("old_contents") or "", ex.get("message") or ex.get("subject") or ""
        lang = (ex.get("lang") or "python").lower(); fn = ex.get("old_file") or ex.get("new_file") or "file"
        if 400 <= len(old) <= MAXC - 400 and 10 <= len(msg) <= 400:
            rows.append({"source": f"file_edit_{lang[:10]}", "prompt": mk(old, msg, lang, fn)})
        if len(rows) >= want: break
except Exception as e:
    print("EditPackFT failed:", repr(e)[:160])
if len(rows) < want:
    try:
        for lang in ("python", "javascript", "go", "java"):
            ds = load_dataset("json", data_files=f"hf://datasets/bigcode/commitpackft/data/{lang}/data.jsonl", split="train", streaming=True)
            for ex in ds:
                old, msg = ex.get("old_contents") or "", ex.get("message") or ""
                if 400 <= len(old) <= MAXC - 400 and 10 <= len(msg) <= 400:
                    rows.append({"source": f"file_edit_{lang}", "prompt": mk(old, msg, lang, ex.get("old_file") or "file")})
                if len(rows) >= want: break
            if len(rows) >= want: break
    except Exception as e:
        print("commitpackft raw failed:", repr(e)[:160])
if len(rows) < want:
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    asks = ["add input validation and clear error messages", "add type hints and docstrings", "refactor into smaller functions without changing behavior",
            "add unit tests at the bottom using the standard testing library", "handle the empty-input edge case and add logging", "convert it to an idiomatic, more efficient implementation"]
    idx = list(range(len(ds))); rng.shuffle(idx)
    for i in idx:
        ex = ds[i]; code = ex.get("output") or ""
        if 300 <= len(code) <= MAXC - 400 and "def " in code or "function" in code:
            rows.append({"source": "file_edit_synth", "prompt": mk(code, rng.choice(asks), "python", "main.py")})
        if len(rows) >= want: break
start = 20000
with open(out, "a") as f:
    for i, r in enumerate(rows): r["id"] = start + i; f.write(json.dumps(r) + "\n")
from collections import Counter; print("file_edit added:", len(rows), dict(Counter(r["source"] for r in rows)))
