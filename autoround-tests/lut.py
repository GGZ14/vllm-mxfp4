"""Build and verify the int4-sym -> e4m3 lookup used by the AutoRound W4A8 kernel.

The whole kernel rests on one fact: AutoRound sym int4 stores code c in 0..15 meaning value
(c - 8), so the dequantized weight is (c-8)*s. Every integer in [-8, 7] is EXACTLY representable
in e4m3 (3 mantissa bits cover 1.000..1.111 at any exponent), so the weight leg is lossless and
the zero point never has to enter the matmul as a correction term -- it is folded into the table.
"""
import numpy as np


def e4m3_byte(v):
    """Exact e4m3 encoding of a small integer; raises if v is not representable."""
    if v == 0:
        return 0x00
    s = 0x80 if v < 0 else 0x00
    a = abs(v)
    e = int(np.floor(np.log2(a)))
    m = a / (2.0 ** e) - 1.0          # in [0,1)
    mant = m * 8.0
    assert abs(mant - round(mant)) < 1e-12, f"{v} needs >3 mantissa bits"
    mant = int(round(mant))
    E = e + 7
    assert 1 <= E <= 14, f"{v} exponent {E} out of e4m3 normal range"
    return s | (E << 3) | mant


def e4m3_decode(b):
    s = -1.0 if (b >> 7) else 1.0
    E = (b >> 3) & 0xF
    m = b & 7
    return s * (m * 2.0 ** -9 if E == 0 else (1 + m / 8.0) * 2.0 ** (E - 7))


LUT = [e4m3_byte(c - 8) for c in range(16)]

print("code -> value -> e4m3 byte -> decoded (must match value exactly)")
ok = True
for c in range(16):
    d = e4m3_decode(LUT[c])
    good = d == float(c - 8)
    ok &= good
    print(f"  c={c:>2}  v={c-8:>3}  0x{LUT[c]:02X}  decoded={d:>5}  {'OK' if good else 'MISMATCH'}")
print(f"\nall 16 codes exact in e4m3: {ok}")
assert ok

# Two 8-entry tables, addressed by the same 3-bit selector, one zeroed by the selector trick.
# Path A (c < 8)  -> negative half, entries LUT[0..7]
# Path B (c >= 8) -> positive half, entries LUT[8..15]
A = LUT[0:8]
B = LUT[8:16]


def pack(t4):
    return (t4[0] | (t4[1] << 8) | (t4[2] << 16) | (t4[3] << 24)) & 0xFFFFFFFF


print()
print("// negative half (codes 0..7)")
print(f"  A_lo = 0x{pack(A[0:4]):08X}u   A_hi = 0x{pack(A[4:8]):08X}u")
print("// positive half (codes 8..15)")
print(f"  B_lo = 0x{pack(B[0:4]):08X}u   B_hi = 0x{pack(B[4:8]):08X}u")


def vperm(hi, lo, sel):
    """Model v_perm_b32(hi, lo, sel): the 8-byte pool is {hi, lo}, index 0..3 = lo, 4..7 = hi."""
    pool = [(lo >> (8 * i)) & 0xFF for i in range(4)] + [(hi >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        s = (sel >> (8 * i)) & 0xFF
        if s < 8:
            b = pool[s]
        elif s < 12:
            b = 0x00
        else:
            b = 0xFF if (pool[s & 3] >> 7) else 0x00
        out |= b << (8 * i)
    return out


# The selector trick: force the unwanted path's selector into 8..11, where v_perm emits 0x00,
# so the two halves can simply be OR-ed. Clearing bit2 when bit3 is set maps 8..15 -> 8..11 and
# leaves 0..7 untouched.
def selectors(c4):
    selA = c4 & ~((c4 & 0x08080808) >> 1) & 0xFFFFFFFF
    d = c4 ^ 0x08080808
    selB = d & ~((d & 0x08080808) >> 1) & 0xFFFFFFFF
    return selA, selB


print("\nverifying the two-vperm selector trick over all 16 codes in all 4 byte lanes")
bad = 0
for c in range(16):
    c4 = c * 0x01010101
    selA, selB = selectors(c4)
    got = vperm(pack(A[4:8]), pack(A[0:4]), selA) | vperm(pack(B[4:8]), pack(B[0:4]), selB)
    want = LUT[c] * 0x01010101
    if got != want:
        bad += 1
        print(f"  c={c:>2} got=0x{got:08X} want=0x{want:08X}  selA=0x{selA:08X} selB=0x{selB:08X}")
print(f"  mismatches: {bad}")
assert bad == 0
print("\nunpack is exact: 3 ops per selector + 2 v_perm + 1 or = 10 ops per 4 codes")
