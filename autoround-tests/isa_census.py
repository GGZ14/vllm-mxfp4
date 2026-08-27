"""Per-kernel ISA census. The interesting quantity is not the WMMA count -- that is fixed by the
tile -- but the address-arithmetic and predication tail around it, which is where a memory-bound
decode kernel and a 55%-of-peak prefill kernel actually spend their instruction slots."""
import re
import sys

lines = open(sys.argv[1]).read().split("\n")
cur, per = None, {}
for l in lines:
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(;.*)?$", l)
    if m and "gemm" in m.group(1):
        cur = m.group(1)
        per[cur] = {}
        continue
    if cur:
        mi = re.match(r"^\s+([a-z][a-z0-9_]+)", l)
        if mi:
            per[cur][mi.group(1)] = per[cur].get(mi.group(1), 0) + 1


def short(n):
    """Name kernels by family and by their leading template args, since several instantiations
    of each appear in one module and mislabelling them makes the table meaningless."""
    fam = ("mx-folded" if "folded" in n else
           "mx-decode" if "mxfp4" in n else
           "ar-decode" if "decode" in n else "ar-prefill")
    args = re.findall(r"ILi(\d+)", n) + re.findall(r"ILb(\d+)", n)
    return f"{fam}<{','.join(args[:3])}>"


keys = ["v_wmma_f32_16x16x16_fp8_fp8", "v_add_co_u32", "s_and_saveexec_b32",
        "v_lshlrev_b64_e32", "v_mad_co_u64_u32", "s_wait_loadcnt", "v_dual_mov_b32",
        "v_perm_b32"]
hdr = ["wmma", "addr64", "saveexec", "shl64", "madu64", "waitload", "dualmov", "perm", "TOTAL"]
print(f"{'kernel':<15}" + "".join(f"{h:>10}" for h in hdr))
for n, c in sorted(per.items(), key=lambda kv: short(kv[0])):
    tot = sum(c.values())
    print(f"{short(n):<15}" + "".join(f"{c.get(k, 0):>10}" for k in keys) + f"{tot:>10}")
    print(f"{'':<15}" + "".join(f"{100.0*c.get(k,0)/tot:>9.1f}%" for k in keys) + f"{'':>10}")
