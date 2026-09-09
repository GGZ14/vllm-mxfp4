"""Round-2 prompt mix: the categories round 1 lost (summarization, file edit, reasoning, json,
prose) plus fresh chat/code/math. Every source is public/ungated; each is best-effort so a missing
dataset shrinks the mix instead of failing. Prompts are capped by characters (~1.8k tokens)."""
import json, random, sys
from datasets import load_dataset
out = sys.argv[1]; rng = random.Random(11); rows = []
MAXC = 7000

def add(source, prompt, n_cap, counter):
    if counter[0] >= n_cap or not prompt or len(prompt) < 20: return False
    rows.append({"source": source, "prompt": prompt[:MAXC]}); counter[0] += 1; return True

def stream(path, split, **kw):
    try: return load_dataset(path, split=split, streaming=True, **kw)
    except Exception as e: print(f"skip {path}: {e!r}"[:200]); return []

# --- summarization: long news + government reports + arxiv ---
c = [0]
for ex in stream("abisee/cnn_dailymail", "test", name="3.0.0"):
    a = ex["article"]
    if 2500 <= len(a) <= MAXC - 200: add("summ_cnn", f"Summarize the following article in 3-5 sentences.\n\n{a}", 280, c)
    if c[0] >= 280: break
c = [0]
for ex in stream("ccdv/govreport-summarization", "test"):
    a = ex["report"]
    if len(a) >= 4000: add("summ_govreport", f"Write an executive summary (one paragraph) of this report, then list its five key findings as bullet points.\n\n{a[:MAXC-300]}", 200, c)
    if c[0] >= 200: break
c = [0]
for ex in stream("ccdv/arxiv-summarization", "test"):
    a = ex["article"]
    if len(a) >= 4000: add("summ_arxiv", f"Read this paper excerpt and produce an abstract of at most 150 words, then explain the main contribution to a non-expert in two sentences.\n\n{a[:MAXC-300]}", 150, c)
    if c[0] >= 150: break
# --- file edit: real commits (old file + instruction) ---
c = [0]
for ex in stream("bigcode/commitpackft", "train", data_dir="python"):
    old, msg = ex.get("old_contents", ""), ex.get("message", "")
    if 400 <= len(old) <= MAXC - 400 and msg:
        add("file_edit", f"Here is the current content of `{ex.get('old_file','file.py')}`:\n\n```python\n{old}\n```\n\nMake this change: {msg}\n\nReturn the complete updated file in a single code block, followed by a one-line note on what changed.", 350, c)
    if c[0] >= 350: break
c = [0]
for ex in stream("bigcode/commitpackft", "train", data_dir="javascript"):
    old, msg = ex.get("old_contents", ""), ex.get("message", "")
    if 400 <= len(old) <= MAXC - 400 and msg:
        add("file_edit_js", f"Current `{ex.get('old_file','index.js')}`:\n\n```javascript\n{old}\n```\n\nTask: {msg}\n\nReply with the full modified file in one code block and a short explanation.", 150, c)
    if c[0] >= 150: break
# --- reasoning: hard math, STEM MC, science ---
c = [0]
for ex in stream("open-r1/OpenR1-Math-220k", "train", name="default"):
    p = ex.get("problem") or ""
    if 60 <= len(p) <= 2500: add("reason_math", p, 300, c)
    if c[0] >= 300: break
c = [0]
for ex in stream("TIGER-Lab/MMLU-Pro", "test"):
    opts = ex.get("options") or []
    if ex.get("question") and opts:
        q = ex["question"] + "\n\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts)) + "\n\nThink it through, then state the answer letter."
        add("reason_mmlupro", q, 200, c)
    if c[0] >= 200: break
c = [0]
for ex in stream("allenai/ai2_arc", "test", name="ARC-Challenge"):
    ch = ex["choices"]; q = ex["question"] + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"])) + "\n\nExplain your reasoning briefly and give the answer."
    add("reason_arc", q, 100, c)
    if c[0] >= 100: break
# --- json: structured-output tasks synthesized over news / chat prompts ---
c = [0]
schemas = [
    ("Extract every person, organization and location mentioned into JSON with keys people, organizations, locations (arrays of strings). Output only JSON.", "summ_cnn"),
    ("Return a JSON object with fields: title (string), summary (string, <=40 words), topics (array of 3-6 strings), sentiment (one of positive/neutral/negative). Output only the JSON.", "summ_cnn"),
    ("Produce a JSON array of the 5 most important facts in the text; each item has fields fact (string) and confidence (number 0-1). Output only JSON.", "summ_cnn"),
]
news = [r["prompt"].split("\n\n", 1)[1] for r in rows if r["source"] == "summ_cnn"]
rng.shuffle(news)
for i, a in enumerate(news[:240]):
    ins, _ = schemas[i % len(schemas)]
    add("json_extract", f"{ins}\n\nText:\n{a[:5000]}", 240, c)
c = [0]
gen_json = ["Design a JSON schema for a {x} and give two example documents that validate against it.",
            "Return a JSON object describing a {x}: include at least 8 fields with realistic values and one nested array. Output only JSON.",
            "Write a JSON configuration for a {x} with comments explained in a separate 'notes' field. Output only JSON."]
things = ["library management system", "weather station", "recipe", "flight booking", "smart thermostat", "e-commerce order", "chess game state", "employee record", "movie review site", "bus schedule",
          "hospital appointment", "podcast episode", "git repository", "solar panel installation", "video game character", "restaurant menu", "shipping container", "IoT sensor network", "conference schedule", "bank transaction"]
for i in range(160):
    add("json_gen", gen_json[i % 3].format(x=things[i % len(things)]), 160, c)
# --- prose: creative writing ---
c = [0]
for ex in stream("euclaise/writingprompts", "train"):
    p = ex.get("prompt") or ""
    if 30 <= len(p) <= 600: add("prose", f"{p}\n\nWrite a short story (400-700 words) based on this prompt.", 300, c)
    if c[0] >= 300: break
# --- chat / code / math (fresh, different sources than round 1) ---
c = [0]
for ex in stream("HuggingFaceH4/no_robots", "train"):
    m = ex["messages"][0]
    if m["role"] == "user" and 30 <= len(m["content"]) <= 3000: add("chat_norobots", m["content"], 300, c)
    if c[0] >= 300: break
c = [0]
for ex in stream("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train"):
    p = ex.get("instruction") or ""
    if 60 <= len(p) <= 3000: add("code_selfoss", p, 350, c)
    if c[0] >= 350: break
c = [0]
for ex in stream("openai/gsm8k", "test", name="main"):
    add("math_gsm8k_test", ex["question"], 200, c)
    if c[0] >= 200: break
rng.shuffle(rows)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        r["id"] = 10000 + i; f.write(json.dumps(r) + "\n")
from collections import Counter
print("prompts:", len(rows), dict(Counter(r["source"] for r in rows)))
