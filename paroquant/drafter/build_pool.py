"""Large, diverse prompt pool for the iterative drafter loop (~25k prompts, ~35 sources). Each
round samples from it without replacement (loop.sh tracks used ids in pool_used.txt). Every source
is public; each is best-effort. Records: {id, source, prompt, messages?}."""
import json, random, sys
from datasets import load_dataset
out = sys.argv[1]; rng = random.Random(97); rows = []; MAXC = 7000
def stream(path, split, **kw):
    try: return load_dataset(path, split=split, streaming=True, **kw)
    except Exception as e: print(f"skip {path}: {e!r}"[:160]); return []
def take(src, it, n, fn):
    c = 0
    for ex in it:
        r = fn(ex)
        if r:
            if isinstance(r, str): r = {"prompt": r}
            r["source"] = src; rows.append(r); c += 1
        if c >= n: break
    print(src, c, flush=True)
# chat
take("ultrachat", stream("HuggingFaceH4/ultrachat_200k", "train_sft"), 4000, lambda ex: ex["messages"][0]["content"] if ex["messages"][0]["role"] == "user" and 30 <= len(ex["messages"][0]["content"]) <= 3000 else None)
def multiturn(ex):
    m = ex["messages"]
    if len(m) >= 3 and m[0]["role"] == "user" and m[2]["role"] == "user" and sum(len(x["content"]) for x in m[:3]) <= 6000:
        return {"prompt": m[2]["content"], "messages": [{"role": x["role"], "content": x["content"]} for x in m[:3]]}
take("chat_multiturn", stream("HuggingFaceH4/ultrachat_200k", "train_gen"), 1500, multiturn)
take("no_robots", stream("HuggingFaceH4/no_robots", "train"), 2000, lambda ex: ex["messages"][0]["content"] if ex["messages"][0]["role"] == "user" and 30 <= len(ex["messages"][0]["content"]) <= 4000 else None)
take("lmsys_like_oasst", stream("OpenAssistant/oasst2", "train"), 1500, lambda ex: ex["text"] if ex.get("role") == "prompter" and ex.get("lang") == "en" and 30 <= len(ex["text"]) <= 3000 else None)
# code
take("magicoder", stream("ise-uiuc/Magicoder-OSS-Instruct-75K", "train"), 4000, lambda ex: {"prompt": ex["problem"], "lang": ex.get("lang")} if 80 <= len(ex.get("problem") or "") <= 4000 else None)
take("self_oss", stream("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train"), 1500, lambda ex: ex["instruction"] if 60 <= len(ex.get("instruction") or "") <= 3000 else None)
take("codealpaca", stream("sahil2801/CodeAlpaca-20k", "train"), 1000, lambda ex: (ex["instruction"] + ("\n\n" + ex["input"] if ex.get("input") else "")) if 20 <= len(ex["instruction"]) <= 2000 else None)
take("code_feedback", stream("m-a-p/CodeFeedback-Filtered-Instruction", "train"), 2000, lambda ex: ex["query"] if 60 <= len(ex.get("query") or "") <= 4000 else None)
def editpack(ex):
    old, msg = ex.get("old_contents") or "", ex.get("message") or ex.get("subject") or ""
    lang = (ex.get("lang") or "python").lower(); fn = ex.get("old_file") or "file"
    if 400 <= len(old) <= MAXC - 400 and 10 <= len(msg) <= 400:
        return f"Here is the current content of `{fn}`:\n\n```{lang}\n{old}\n```\n\nMake this change: {msg}\n\nReturn the complete updated file in a single code block, followed by a one-line note on what changed."
take("file_edit", stream("nuprl/EditPackFT", "train"), 1500, editpack)
# synthetic polyglot (languages the instruct sets lack)
TASKS = ["Write a {L} program that reads a CSV file path from the command line, groups rows by column {n}, and prints per-group counts and the average of column {m}. Include error handling.",
 "Implement an LRU cache in {L} with get/put in O(1), capacity {cap}, and unit tests.", "In {L}, write a small HTTP server exposing GET /health and POST /{res} (JSON body, in-memory store) with proper status codes and validation.",
 "Write a {L} function that parses ISO-8601 timestamps with offsets, converts them to UTC, and returns them sorted; add tests for {n} edge cases.", "Implement a concurrent worker pool in {L} with at most {cap} workers and graceful shutdown on interrupt.",
 "Write a {L} CLI that walks a directory tree, computes SHA-256 of every file larger than {n} KB, and reports duplicates.", "Implement a token-bucket rate limiter in {L} ({cap} req/s, burst {m}) for middleware, with tests.",
 "Write {L} code to merge {n} sorted streams lazily and explain the complexity.", "Here is a {L} function with a bug: it should return the {n} most frequent words in a text but sometimes returns fewer. Write the corrected version, an explanation, and a test that would have caught it.",
 "Refactor a {L} module that mixes database access, business rules and HTTP handling for {res} into three layers; show the code and interfaces.", "Write a {L} binary search tree with insert, delete, in-order iteration and a balance check; include tests.",
 "Implement retry with exponential backoff and jitter in {L} for an unreliable {res} API client; make it cancellable.", "Write a {L} program that tails a log file, parses 'ts level=... msg=...' lines and prints a per-minute error count.",
 "Implement a small plugin system in {L} where plugins for {res} processing are discovered at startup and run in order.", "Write {L} code that connects to PostgreSQL, runs a parameterized query over {res}, streams results in batches of {cap}, and handles reconnects.",
 "Implement Dijkstra's shortest path in {L} over a weighted graph read from a file; print the path between two given nodes.", "Write a {L} library function that validates and normalizes email addresses and phone numbers for {n} countries, with table-driven tests.",
 "Implement a pub/sub event bus in {L} with typed topics, backpressure and at-least-once delivery.", "Write a {L} program computing a moving average and standard deviation over a stream with window {cap}, in constant memory.",
 "Implement in {L} a state machine for a {res} order with states created, paid, shipped, delivered, cancelled; illegal transitions must error.", "Write a {L} JSON parser for a JSON subset without external libraries, with tests.",
 "Implement a thread-safe in-memory key-value store in {L} with TTL expiry and a snapshot-to-disk command.", "Write a {L} script that reads {n} URLs from stdin, fetches them concurrently with a {m}-second timeout, and prints status and latency per URL.",
 "Explain how error handling idioms differ between {L} and Python, then write a {L} example wrapping and inspecting errors from a {res} service call.", "Implement a trie in {L} with insert, lookup, prefix search returning up to {cap} results, and deletion.",
 "Write a {L} program that parses command-line flags for a {res} tool (verbose, config path, dry-run) and prints usage on error.", "Write unit tests in {L} for a {res} parser module: valid input, malformed input, and {n} boundary cases.",
 "Implement a producer-consumer pipeline in {L} with three stages and bounded buffers of size {cap}.", "Write a {L} function to deep-merge two configuration maps with arrays replaced rather than concatenated; include tests.",
 "Write a {L} program that implements a tiny expression evaluator (+ - * / parentheses, variables) with a REPL and tests."]
SQL = ["Write PostgreSQL SQL to find the top {n} {res} by revenue per month for the last year, handling months with no sales; note index choices.", "Given users(id, signup_date) and orders(id, user_id, amount, created_at): write SQL for {n}-day retention cohorts by signup month.",
 "Write a SQL migration adding soft-delete to a {res} table and rewrite the reporting queries; include indexes.", "Write a recursive CTE listing a {res} hierarchy with depth and path, then the {n} largest subtrees.",
 "Write SQL to deduplicate {res} rows keeping the most recent per natural key, safely in production.", "Write a window-function query computing a {cap}-row moving average of daily {res} counts and flag days more than 2 standard deviations off.",
 "Design 3NF tables for a {res} system, write DDL with constraints, and {n} dashboard queries.", "Write SQL that pivots monthly {res} totals into columns for the last {n} months without PIVOT."]
RES = ["orders", "invoices", "sensors", "tickets", "users", "shipments", "products", "sessions", "events", "payments", "documents", "jobs", "devices", "accounts"]
for L, key, n in (("Go", "go", 500), ("Rust", "rust", 250), ("TypeScript", "typescript", 300), ("Kotlin", "kotlin", 200), ("Ruby", "ruby", 200), ("SQL", "sql", 250), ("JavaScript", "javascript", 250), ("Bash", "shell", 200), ("C", "c", 200), ("Scala", "scala", 100), ("Elixir", "elixir", 100), ("Zig", "zig", 100), ("Dart", "dart", 100), ("Lua", "lua", 100)):
    pool = SQL if key == "sql" else TASKS
    for i in range(n):
        t = pool[i % len(pool)]; rows.append({"source": f"code_{key}", "prompt": t.format(L=L, n=rng.choice([3, 5, 7, 10, 12, 20]), m=rng.choice([2, 4, 5, 8, 15]), cap=rng.choice([8, 16, 32, 64, 100, 256]), res=rng.choice(RES))})
    print(f"code_{key}", n)
# summarization / long documents
take("summ_cnn", stream("abisee/cnn_dailymail", "train", name="3.0.0"), 1200, lambda ex: f"Summarize the following article in 3-5 sentences.\n\n{ex['article']}" if 2500 <= len(ex["article"]) <= MAXC - 200 else None)
take("summ_govreport", stream("ccdv/govreport-summarization", "train"), 600, lambda ex: f"Write an executive summary of this report, then list its five key findings.\n\n{ex['report'][:MAXC-300]}" if len(ex["report"]) >= 4000 else None)
take("summ_arxiv", stream("ccdv/arxiv-summarization", "train"), 500, lambda ex: f"Produce an abstract of at most 150 words for this paper excerpt, then explain the main contribution to a non-expert.\n\n{ex['article'][:MAXC-300]}" if len(ex["article"]) >= 4000 else None)
take("summ_xsum", stream("EdinburghNLP/xsum", "train"), 500, lambda ex: f"Summarize this article in one sentence, then in one paragraph.\n\n{ex['document']}" if 1500 <= len(ex["document"]) <= MAXC else None)
# reasoning / math / science
take("reason_openr1", stream("open-r1/OpenR1-Math-220k", "train", name="default"), 1500, lambda ex: ex["problem"] if 60 <= len(ex.get("problem") or "") <= 2500 else None)
take("reason_mmlupro", stream("TIGER-Lab/MMLU-Pro", "test"), 1000, lambda ex: (ex["question"] + "\n\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(ex["options"])) + "\n\nThink it through, then state the answer letter.") if ex.get("options") else None)
take("reason_arc", stream("allenai/ai2_arc", "train", name="ARC-Challenge"), 500, lambda ex: ex["question"] + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"])) + "\n\nExplain briefly and give the answer.")
take("gsm8k", stream("openai/gsm8k", "train", name="main"), 1500, lambda ex: ex["question"])
take("math_hendrycks", stream("EleutherAI/hendrycks_math", "train", name="algebra"), 400, lambda ex: ex["problem"] if 40 <= len(ex.get("problem") or "") <= 2000 else None)
take("science_qa", stream("allenai/sciq", "train"), 400, lambda ex: ex["question"] + " Explain the reasoning.")
# structured output
news = [r["prompt"].split("\n\n", 1)[1] for r in rows if r["source"] == "summ_cnn"][:900]
schemas = ["Extract every person, organization and location mentioned into JSON with keys people, organizations, locations (arrays of strings). Output only JSON.",
           "Return a JSON object with fields title, summary (<=40 words), topics (3-6 strings), sentiment (positive/neutral/negative). Output only JSON.",
           "Produce a JSON array of the 5 most important facts in the text; each item has fact (string) and confidence (0-1). Output only JSON.",
           "Convert the article into a JSON timeline: an array of {date, event} objects in chronological order. Output only JSON."]
for i, a in enumerate(news): rows.append({"source": "json_extract", "prompt": f"{schemas[i % len(schemas)]}\n\nText:\n{a[:5000]}"})
print("json_extract", len(news))
things = ["library management system", "weather station", "recipe", "flight booking", "smart thermostat", "e-commerce order", "chess game state", "employee record", "movie review site", "bus schedule", "hospital appointment", "podcast episode", "git repository", "solar installation", "game character", "restaurant menu", "shipping container", "IoT sensor network", "conference schedule", "bank transaction", "CI pipeline", "kubernetes deployment", "invoice", "GraphQL API", "music playlist"]
gen = ["Design a JSON schema for a {x} and give two example documents that validate against it.", "Return a JSON object describing a {x} with at least 8 fields, realistic values and one nested array. Output only JSON.",
       "Write a YAML configuration for a {x} with comments, then the equivalent JSON.", "Produce a JSON API response for listing {x}s with pagination metadata and 3 example items. Output only JSON."]
for i in range(500): rows.append({"source": "json_gen", "prompt": gen[i % 4].format(x=things[i % len(things)])})
print("json_gen", 500)
# prose
take("prose_wp", stream("euclaise/writingprompts", "train"), 1500, lambda ex: f"{ex['prompt']}\n\nWrite a short story (400-700 words) based on this prompt." if 30 <= len(ex.get("prompt") or "") <= 600 else None)
# tools
def tool(ex):
    sysm, chat = ex.get("system") or "", ex.get("chat") or ""
    if "USER:" in chat and len(sysm) < 4000:
        user = chat.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        if 10 <= len(user) <= 2000: return {"prompt": user, "messages": [{"role": "system", "content": sysm.replace("SYSTEM: ", "", 1)}, {"role": "user", "content": user}]}
take("tool_calling", stream("glaiveai/glaive-function-calling-v2", "train"), 1500, tool)
rng.shuffle(rows)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        r["id"] = 100000 + i; r["prompt"] = r["prompt"][:MAXC]; f.write(json.dumps(r) + "\n")
from collections import Counter; print("POOL", len(rows), dict(Counter(r["source"] for r in rows)))
