"""Prompt mix for drafter self-distillation: chat (ultrachat_200k), code (CodeAlpaca-20k), math
(GSM8K train). Responses are generated later by OUR target through the served API, so only the
prompts come from datasets. Writes prompts.jsonl: {id, source, prompt}."""
import json, random, sys
from datasets import load_dataset
out = sys.argv[1]; n_chat, n_code, n_math = 1200, 700, 500
rng = random.Random(7); rows = []
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
for ex in ds:
    m = ex["messages"][0]
    if m["role"] == "user" and 30 <= len(m["content"]) <= 2000:
        rows.append({"source": "ultrachat", "prompt": m["content"]})
    if len(rows) >= n_chat: break
ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
idx = list(range(len(ds))); rng.shuffle(idx); k = 0
for i in idx:
    ex = ds[i]; p = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
    if 20 <= len(p) <= 2000:
        rows.append({"source": "codealpaca", "prompt": p}); k += 1
    if k >= n_code: break
ds = load_dataset("openai/gsm8k", "main", split="train")
idx = list(range(len(ds))); rng.shuffle(idx)
for i in idx[:n_math]:
    rows.append({"source": "gsm8k", "prompt": ds[i]["question"]})
rng.shuffle(rows)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        r["id"] = i; f.write(json.dumps(r) + "\n")
print("prompts:", len(rows), {s: sum(r["source"] == s for r in rows) for s in ("ultrachat", "codealpaca", "gsm8k")})
