"""Held-out drafter measurement on the served target: N unused pool prompts, single stream, prod sampling
(temperature 0.7, seeded per prompt), spec-decode counters read from /metrics before and after.
Reports acc/draft, tokens per update and completion tok/s over the whole set -- tens of thousands of
tokens across every pool source, instead of the 3-5 prompts of bench_decode_ctx.
Usage: eval_drafter.py <heldout.jsonl> <out.json> [max_tokens]"""
import json, sys, time, urllib.request
prompts, out = sys.argv[1], sys.argv[2]; max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
URL = "http://localhost:8080/v1/chat/completions"; MODEL = "Qwen3.8-PARO-MXFP4"; METRICS = "http://localhost:8080/metrics"
def counters():
    time.sleep(2.0); raw = urllib.request.urlopen(METRICS, timeout=10).read().decode(); o = {}
    for line in raw.splitlines():
        for k, m in (("n", "vllm:spec_decode_num_drafts_total"), ("d", "vllm:spec_decode_num_draft_tokens_total"), ("a", "vllm:spec_decode_num_accepted_tokens_total")):
            if line.startswith(m): o[k] = float(line.rsplit(" ", 1)[1])
    return o
rows = [json.loads(l) for l in open(prompts)]
c0 = counters(); t0 = time.time(); toks = 0; secs = 0.0; per = []
for i, r in enumerate(rows):
    body = json.dumps({"model": MODEL, "messages": r.get("messages") or [{"role": "user", "content": r["prompt"]}],
                       "max_tokens": max_tokens, "temperature": 0.7, "seed": int(r["id"])}).encode()
    a = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
        ct = d["usage"]["completion_tokens"]
    except Exception as e:  # noqa
        print("error", r["id"], repr(e), flush=True); continue
    dt = time.time() - a; toks += ct; secs += dt; per.append({"id": r["id"], "source": r["source"], "tokens": ct, "secs": round(dt, 2)})
    if (i + 1) % 20 == 0: print(f"{i+1}/{len(rows)} {toks} tok {toks/secs:.1f} tok/s", flush=True)
c1 = counters(); n = c1["n"] - c0["n"]; dr = c1["d"] - c0["d"]; ac = c1["a"] - c0["a"]
res = {"prompts": len(per), "completion_tokens": toks, "tok_s": round(toks / secs, 2), "drafts": n, "acc_per_draft": round(ac / n, 4),
       "tok_per_update": round((ac + n) / n, 4), "wall_s": round(time.time() - t0, 1), "per": per}
json.dump(res, open(out, "w")); print("RESULT", json.dumps({k: v for k, v in res.items() if k != "per"}), flush=True)
