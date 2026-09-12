"""Decode throughput vs CONTEXT LENGTH, on non-repetitive output.

Why this exists: bench_real_decode.py uses ~40-token prompts, so both of the decode numbers this
build publishes are near-zero-context. But decode cost grows with context -- the full-attention
layers re-read the whole KV cache on every forward, and with MTP that is paid again on every
drafter pass. On prod FP8 decode falls 217 -> 57.8 tok/s from 6.6k to 97k. A 857k-token KV cache
is this build's selling point, so that curve is the number that matters.

Filler is REAL PROSE (wikitext-2 test), not random words: the dup-8gram guard only means something
if the model is given something coherent to continue from. A unique tag is prepended so the block
hashes differ and --enable-prefix-caching cannot serve the prefill from cache.

Acceptance is read per arm as a delta of the cumulative vllm:spec_decode_* counters, because
acceptance itself varies with context and a single whole-run figure would hide that.
"""
import json, os, sys, time, urllib.request, random, string

URL = "http://localhost:8080/v1/completions"
METRICS = "http://localhost:8080/metrics"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-MXFP4")
CORPUS = os.environ.get("BENCH_CORPUS", "/home/brian/pibench-local/corpus/wikitext2_test.txt")
GEN = int(os.environ.get("BENCH_GEN", "400"))
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
SEED = os.environ.get("BENCH_SEED")   # 0 = greedy: identical text across drafters, so acc/draft is comparable
# 0 reproduces bench_real_decode.py's operating point, so the two harnesses tie together.
TARGETS = [int(x) for x in os.environ.get("BENCH_CTX", "0,8000,32000,100000,200000").split(",")]

# The zero-context arm gets its own wording: reusing the long-context instruction verbatim
# left a dangling "Ignoring everything above" with nothing above it, and the model stopped
# after 41 tokens -- too few to time.
ZERO_INSTR = ("Write a detailed technical explanation of how a GPU memory controller schedules "
              "requests across channels and banks. Be specific and avoid repetition.\n\n")
PREAMBLE = "Reference material (ignore its style; it is unrelated background):\n\n"
INSTR = ("\n\n=== end of reference material ===\n\n"
         "Ignoring everything above, write a detailed technical explanation of how a GPU memory "
         "controller schedules requests across channels and banks. Be specific, write in your "
         "own words, and do not repeat yourself.\n\n")

_text = open(CORPUS, encoding="utf-8").read()


SETTLE = float(os.environ.get("BENCH_SETTLE", "2.0"))


def counters():
    """(drafts, draft_tokens, accepted) -- cumulative, so only deltas are meaningful.

    Settles first: the engine core pushes these asynchronously, so without a pause the delta
    straddles the neighbouring request. The self-check in run() is what proves it worked --
    drafts x (acc/draft + 1) must equal the tokens actually generated.
    """
    time.sleep(SETTLE)
    try:
        raw = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return None
    out = {}
    for line in raw.splitlines():
        for k, m in (("n", "vllm:spec_decode_num_drafts_total"),
                     ("d", "vllm:spec_decode_num_draft_tokens_total"),
                     ("a", "vllm:spec_decode_num_accepted_tokens_total")):
            if line.startswith(m):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out if len(out) == 3 else None


def filler(target_tokens):
    if target_tokens <= 0:
        return ""
    # ~4.5 chars/token on this corpus; the exact count is read back from the response, not assumed.
    need = int(target_tokens * 4.5)
    if need > len(_text):
        sys.exit(f"corpus has {len(_text)} chars, need ~{need} for {target_tokens} tokens")
    # BENCH_SEED fixes the prefix (same text for every server under A/B); unset = fresh slice per run
    rng = random.Random(int(SEED) * 1000003 + target_tokens) if SEED else random.Random(time.time_ns())
    off = rng.randrange(0, len(_text) - need + 1)
    return _text[off:off + need]


def run(target):
    tag = "".join(random.Random(time.time_ns()).choices(string.ascii_lowercase, k=16))
    prompt = ((f"[session {tag}]\n" + PREAMBLE + filler(target) + INSTR) if target > 0
              else ZERO_INSTR)
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": GEN,
                       "temperature": TEMP, "seed": 1, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    before = counters()
    t0 = time.time(); tf = None; last = t0; usage = None; text = []
    for line in urllib.request.urlopen(req, timeout=1800):
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        try:
            d = json.loads(p)
        except Exception:
            continue
        if d.get("choices") and d["choices"][0].get("text"):
            now = time.time()
            if tf is None:
                tf = now
            last = now
            text.append(d["choices"][0]["text"])
        if d.get("usage"):
            usage = d["usage"]
    after = counters()

    ct = usage["completion_tokens"]; pt = usage["prompt_tokens"]
    dec = max(last - tf, 1e-9)
    toks = "".join(text).split()
    grams = [" ".join(toks[i:i + 8]) for i in range(max(0, len(toks) - 7))]
    dup = 1 - (len(set(grams)) / max(1, len(grams)))

    acc = ""
    if before and after:
        dn, dd, da = (after[k] - before[k] for k in ("n", "d", "a"))
        if dn > 0:
            implied = dn * (da / dn + 1)
            ok = "" if abs(implied - ct) <= 0.08 * ct else f" [!counters off by {implied/ct:.2f}x]"
            # ms/step is the metric that reflects the DECODE PATH; tok/s does not. Tokens emitted
            # per engine step is acc/draft + 1, and acceptance swings run-to-run with content --
            # at 32k the same config measured 68.4 and 78.1 tok/s purely on acceptance (1.53 vs
            # 1.85), while step time held at 37.0 vs 36.5 ms. Compare step time between configs.
            acc = (f" | acc/draft {da/dn:5.3f} | {1000*(da/dn + 1)/(ct/dec):6.2f} ms/step"
                   f" | rate {100*da/max(dd,1):5.2f}%{ok}")
    print(f"ctx {pt:>7} tok | gen {ct:4d} | TTFT {(tf-t0)*1000:8.0f} ms "
          f"| decode {ct/dec:7.1f} tok/s | dup-8gram {dup*100:4.1f}%{acc}", flush=True)


print(f"=== decode vs context ({MODEL}, gen {GEN}, filler {os.path.basename(CORPUS)}) ===",
      flush=True)
REPS = int(os.environ.get("BENCH_REPS", "1"))
for t in TARGETS:
    for r in range(REPS):
        run(t)
print("DONE", flush=True)
