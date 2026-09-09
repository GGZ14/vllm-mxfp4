"""Round-2 extension: polyglot code (Go, Rust, TypeScript, C++, Java, C#, Shell, SQL, Kotlin, Swift,
PHP, Ruby), multi-turn chat (a prior exchange in context), and tool-calling turns. Public datasets,
best-effort per source. Writes {id, source, prompt, messages?} -- generate.py sends `messages`
when present so multi-turn and tool prompts keep their structure."""
import json, random, sys
from datasets import load_dataset
out = sys.argv[1]; rng = random.Random(23); rows = []
def stream(path, split, **kw):
    try: return load_dataset(path, split=split, streaming=True, **kw)
    except Exception as e: print(f"skip {path}: {e!r}"[:160]); return []
# --- polyglot code: Magicoder OSS-Instruct (has a language field) ---
want = {"go": 110, "rust": 110, "typescript": 100, "cpp": 90, "java": 90, "csharp": 70, "shell": 70, "sql": 60, "kotlin": 50, "swift": 50, "php": 50, "ruby": 50, "javascript": 60}
got = {k: 0 for k in want}
for ex in stream("ise-uiuc/Magicoder-OSS-Instruct-75K", "train"):
    lang = (ex.get("lang") or "").lower(); p = ex.get("problem") or ""
    key = {"c++": "cpp", "c#": "csharp", "typescript": "typescript", "js": "javascript"}.get(lang, lang)
    if key in want and got[key] < want[key] and 80 <= len(p) <= 4000:
        rows.append({"source": f"code_{key}", "prompt": p}); got[key] += 1
    if all(got[k] >= want[k] for k in want): break
print("magicoder:", got)
# top up languages Magicoder lacks with explicit tasks
import itertools
TASKS = [
 "Write a {L} program that reads a CSV file path from the command line, groups rows by the value in column {n}, and prints per-group counts and the average of column {m}. Include error handling.",
 "Implement an LRU cache in {L} with get/put in O(1), capacity {cap}, and unit tests.",
 "In {L}, write a small HTTP server exposing GET /health and POST /{res} (JSON body, in-memory store) with proper status codes and request validation.",
 "Write a {L} function that parses ISO-8601 timestamps with offsets, converts them to UTC, and returns them sorted; add tests for {n} edge cases.",
 "Implement a concurrent worker pool in {L} that processes jobs from a queue with at most {cap} workers and graceful shutdown on interrupt.",
 "Write a {L} CLI that walks a directory tree, computes SHA-256 of every file larger than {n} KB, and reports duplicates.",
 "Implement a token-bucket rate limiter in {L} ({cap} requests per second, burst {m}) suitable for middleware, with tests.",
 "Write {L} code to merge {n} sorted streams lazily (iterator style) and explain the complexity.",
 "Here is a {L} function with a bug: it is supposed to return the {n} most frequent words in a text but sometimes returns fewer. Write the corrected version with a short explanation and a test that would have caught the bug.",
 "Refactor a {L} module that mixes database access, business rules and HTTP handling for {res} into three layers; show the resulting code and interfaces.",
 "Write a {L} implementation of a binary search tree with insert, delete, in-order iteration and a balance check; include tests.",
 "Implement retry with exponential backoff and jitter in {L} for an unreliable {res} API client; make it cancellable.",
 "Write a {L} program that tails a log file, parses lines like '2026-01-01T00:00:00Z level=error msg=...' and prints a per-minute error count.",
 "Design and implement a small plugin system in {L} where plugins for {res} processing are discovered at startup and run in order.",
 "Write {L} code that connects to PostgreSQL, runs a parameterized query over {res}, streams results in batches of {cap}, and handles reconnects.",
 "Implement Dijkstra's shortest path in {L} over a weighted graph read from a file; print the path between two given nodes.",
 "Write a {L} library function that validates and normalizes email addresses and phone numbers for {n} countries, with table-driven tests.",
 "Implement a simple pub/sub event bus in {L} with typed topics, backpressure and at-least-once delivery semantics.",
 "Write a {L} program that computes a moving average and standard deviation over a stream of numbers with window {cap}, in constant memory.",
 "Convert this description into {L}: a state machine for a {res} order with states created, paid, shipped, delivered, cancelled; illegal transitions must error.",
 "Write a {L} JSON parser for a subset of JSON (objects, arrays, strings, numbers, booleans, null) without external libraries, with tests.",
 "Implement a thread-safe in-memory key-value store in {L} with TTL expiry and a snapshot-to-disk command.",
 "Write a {L} script that reads {n} URLs from stdin, fetches them concurrently with a timeout of {m} seconds, and prints status code and latency per URL.",
 "Explain how error handling idioms differ between {L} and Python, then write a {L} example that wraps and inspects errors from a {res} service call.",
 "Write a {L} implementation of matrix multiplication for {n}x{n} matrices, then optimize it (blocking, or SIMD/vectorization where the language allows) and benchmark both.",
 "Implement a trie in {L} with insert, exact lookup, prefix search returning up to {cap} results, and deletion.",
 "Write a {L} program that parses command-line flags for a {res} tool (verbose, config path, dry-run) and prints a usage message on error.",
 "Given a {L} codebase for {res}, write the unit tests for its parser module: cover valid input, malformed input, and {n} boundary cases.",
 "Implement a producer-consumer pipeline in {L} with three stages (read, transform, write) and bounded buffers of size {cap}.",
 "Write a {L} function to deep-merge two configuration objects/maps with arrays replaced rather than concatenated; include tests.",
]
SQL_TASKS = [
 "Write PostgreSQL SQL to find the top {n} {res} by revenue per month for the last year, handling months with no sales; note index choices.",
 "Given users(id, signup_date) and orders(id, user_id, amount, created_at): write SQL for {n}-day retention cohorts by signup month.",
 "Write a SQL migration adding soft-delete to a {res} table and rewrite the reporting queries to respect it; include indexes.",
 "Write a recursive CTE in SQL to list a {res} hierarchy with depth and path, then a query for the {n} largest subtrees.",
 "Write SQL to deduplicate {res} rows keeping the most recent per natural key, and explain how to do it safely in production.",
 "Write a window-function query computing a {cap}-row moving average of daily {res} counts and flag days more than 2 standard deviations off.",
 "Design tables for a {res} system (3NF), write the DDL with constraints, and {n} example queries an admin dashboard would need.",
 "Write SQL that pivots monthly {res} totals into columns for the last {n} months without a PIVOT extension.",
]
RES = ["orders", "invoices", "sensors", "tickets", "users", "shipments", "products", "sessions", "events", "payments", "documents", "jobs"]
def synth(L, key, n_need, seed):
    r = random.Random(seed); pool = SQL_TASKS if key == "sql" else TASKS; out = []
    for i in range(n_need):
        t = pool[i % len(pool)]
        out.append({"source": f"code_{key}", "prompt": t.format(L=L, n=r.choice([3, 5, 7, 10, 12, 20]), m=r.choice([2, 4, 5, 8, 15]), cap=r.choice([8, 16, 32, 64, 100, 256]), res=r.choice(RES))})
    return out
for L, key in (("Go", "go"), ("Rust", "rust"), ("TypeScript", "typescript"), ("C++", "cpp"), ("Java", "java"), ("C#", "csharp"), ("Bash", "shell"), ("Kotlin", "kotlin"), ("Swift", "swift"), ("PHP", "php"), ("Ruby", "ruby"), ("SQL", "sql"), ("JavaScript", "javascript")):
    n_need = max(0, want.get(key, 60) - got.get(key, 0)) + (30 if key in ("go", "rust", "typescript") else 10)
    rows += synth(L, key, n_need, hash(key) & 0xffff)
# --- multi-turn chat: a real first exchange as context, user follows up ---
c = 0
for ex in stream("HuggingFaceH4/ultrachat_200k", "train_sft"):
    m = ex["messages"]
    if len(m) >= 3 and m[0]["role"] == "user" and m[1]["role"] == "assistant" and m[2]["role"] == "user" and sum(len(x["content"]) for x in m[:3]) <= 6000:
        rows.append({"source": "chat_multiturn", "prompt": m[2]["content"], "messages": [{"role": x["role"], "content": x["content"]} for x in m[:3]]}); c += 1
    if c >= 250: break
# --- tool calling: system prompt with tool definitions, user request ---
c = 0
for ex in stream("glaiveai/glaive-function-calling-v2", "train"):
    sysm, chat = ex.get("system") or "", ex.get("chat") or ""
    if "USER:" in chat and len(sysm) < 4000:
        user = chat.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        if 10 <= len(user) <= 2000:
            rows.append({"source": "tool_calling", "prompt": user, "messages": [{"role": "system", "content": sysm.replace("SYSTEM: ", "", 1)}, {"role": "user", "content": user}]}); c += 1
    if c >= 250: break
rng.shuffle(rows)
with open(out, "w") as f:
    for i, r in enumerate(rows): r["id"] = 30000 + i; f.write(json.dumps(r) + "\n")
from collections import Counter; print("prompts:", len(rows), dict(Counter(r["source"] for r in rows)))
