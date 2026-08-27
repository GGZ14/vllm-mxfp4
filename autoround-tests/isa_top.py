"""Top instructions inside one kernel, to find what the remaining budget is spent on."""
import re, sys
lines = open(sys.argv[1]).read().split("\n")
want = sys.argv[2]
cur, per = None, {}
for l in lines:
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(;.*)?$", l)
    if m and "gemm" in m.group(1):
        cur = m.group(1); per[cur] = []
        continue
    if cur:
        mi = re.match(r"^\s+([a-z][a-z0-9_]+)", l)
        if mi: per[cur].append(mi.group(1))
for n, ops in per.items():
    if want not in n: continue
    from collections import Counter
    c = Counter(ops); tot = len(ops)
    print(f"{n[:60]}  total={tot}")
    for op, k in c.most_common(14):
        print(f"   {op:<28} {k:>5}  {100.0*k/tot:>5.1f}%")
    break
