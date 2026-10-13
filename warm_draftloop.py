#!/usr/bin/env python3
"""Pre-capture the MoE drafter-loop graphs (RADIANCE_MOE_DRAFT_GRAPH) for batch sizes 1..N right after startup
(radiance-glue/ Phase 3, RADIANCE_MOE_DRAFT_WARM=N).

radiance_moe_draftloop captures one graph per (batch, padded batch) on that key's third use, so the first burst at
each new concurrency pays the capture inside a user request (1-2 s TTFT at 12 streams, radiance-gaps). This runs in
the background inside the container: it waits until the server answers /v1/models, then for k = 1..N sends k
concurrent greedy requests (short prompt, 24 forced tokens) so every k reaches at least three drafter loop passes.
Logs one line per round and a summary to stderr ("[radiance.warm] ..."). Never fails the serve.
    warm_draftloop.py PORT N MODEL
"""
import json
import sys
import threading
import time
import urllib.request

port, n, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
base = f"http://127.0.0.1:{port}/v1"


def log(msg):
    sys.stderr.write(f"[radiance.warm] {msg}\n")
    sys.stderr.flush()


def one(i, k, out):
    body = {"model": model, "temperature": 0, "max_tokens": 24, "min_tokens": 24, "ignore_eos": True,
            "messages": [{"role": "user", "content": f"Warm-up {k}.{i}: count from 1 to 40, comma separated."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        out[i] = True
    except Exception as e:  # noqa: BLE001
        out[i] = repr(e)


def main():
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(base + "/models", timeout=5).read()
            break
        except Exception:  # noqa: BLE001
            if time.time() - t0 > 3600:
                log("server never came up; giving up")
                return
            time.sleep(2)
    tr = time.time()
    for k in range(1, n + 1):
        out = [None] * k
        ts = time.time()
        th = [threading.Thread(target=one, args=(i, k, out)) for i in range(k)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        bad = [o for o in out if o is not True]
        log(f"round B={k}: {time.time() - ts:.2f} s" + (f", {len(bad)} failed: {bad[0]}" if bad else ""))
    log(f"done: {n} rounds in {time.time() - tr:.1f} s after the server answered")


main()
