"""Deploy smoke checks for the MoE lane under async scheduling (not a benchmark): tool calls (1 + 8 concurrent),
guided JSON (json_schema), and temperature-0.7 sampling. Prints PASS/FAIL per check and exits non-zero on any FAIL."""
import json
import sys
import threading
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.5-35B-A3B-MXFP4"
fails = []


def chat(body):
    body = {"model": MODEL, "chat_template_kwargs": {"enable_thinking": False}, **body}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}]


def tool_call(i, out):
    try:
        city = ["Paris", "Tokyo", "Lima", "Oslo", "Cairo", "Perth", "Quito", "Seoul"][i % 8]
        r = chat({"messages": [{"role": "user", "content": f"What is the weather in {city} right now? Use the tool."}],
                  "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 200})
        tc = r["choices"][0]["message"].get("tool_calls") or []
        args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
        ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and city.lower() in str(args.get("city", "")).lower()
        out[i] = (ok, tc[0]["function"]["arguments"] if tc else r["choices"][0]["message"].get("content", "")[:120])
    except Exception as e:  # noqa: BLE001
        out[i] = (False, f"error: {e}")


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


out = {}
tool_call(0, out)
check("tool call (single)", out[0][0], out[0][1])
out = {}
th = [threading.Thread(target=tool_call, args=(i, out)) for i in range(8)]
[t.start() for t in th]
[t.join() for t in th]
n_ok = sum(1 for v in out.values() if v[0])
check("tool call (8 concurrent)", n_ok == 8, f"{n_ok}/8 valid; e.g. {out[3][1]}")

schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "hobbies": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "age", "hobbies"], "additionalProperties": False}
try:
    r = chat({"messages": [{"role": "user", "content": "Invent a fictional person."}], "temperature": 0, "max_tokens": 200,
              "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}}})
    txt = r["choices"][0]["message"]["content"]
    obj = json.loads(txt)
    ok = isinstance(obj.get("name"), str) and isinstance(obj.get("age"), int) and isinstance(obj.get("hobbies"), list)
    check("guided JSON (json_schema)", ok, txt[:160])
except Exception as e:  # noqa: BLE001
    check("guided JSON (json_schema)", False, f"error: {e}")

for k in range(3):
    try:
        r = chat({"messages": [{"role": "user", "content": "Write two sentences about the ocean at night."}],
                  "temperature": 0.7, "top_p": 0.95, "max_tokens": 90})
        txt = r["choices"][0]["message"]["content"].strip()
        ok = len(txt.split()) >= 12 and r["choices"][0]["finish_reason"] in ("stop", "length")
        check(f"temperature 0.7 sample {k + 1}", ok, txt.replace("\n", " ")[:160])
    except Exception as e:  # noqa: BLE001
        check(f"temperature 0.7 sample {k + 1}", False, f"error: {e}")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
