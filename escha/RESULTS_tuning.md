# Escha W2 kernels on RDNA4 / R9700 — tuning results

Both kernels measured against the shipped MXFP4 kernels **in one binary**, arms interleaved and
repeated, weights rotated over 4 buffers so the working set clears the 64 MB Infinity Cache.
Correctness re-gated after every change: trellis decode 24/24 **bit-exact**, decode GEMM and
prefill GEMM at 1.65–1.70e-3 — the independently-computed bf16 rounding floor.

Final numbers come from `final.hip`, which builds **only the surviving arms**, five interleaved
repeats. That matters: a spilling variant elsewhere in the binary depresses clocks for everything
measured after it, and two earlier prefill passes disagreed by 5% for exactly that reason.

**MXFP4 is measured at its own best configuration too.** Its production launcher picks split-K via
`split_k_for()` and BK via `decode_bk64()`; an earlier pass pinned it to DKS=1 and made escha look
like it won the `down` shape. It does not. Best-of is compared to best-of throughout.

## Decode (µs)

| shape | M | start | tuned | gain | mxfp4 | esc/mx start → now |
|---|---|---|---|---|---|---|
| gate_up | 8  | 210.4 | 123.0 | 1.71× | 99.2  | 2.12× → 1.240× |
| gate_up | 40 | 251.3 | 150.7 | 1.67× | 114.1 | 2.25× → 1.321× |
| gate_up | 64 | 284.9 | 174.2 | 1.64× | 130.6 | 1.81× → 1.334× |
| down    | 8  | 192.8 | 60.7  | 3.18× | 45.7  | 2.35× → 1.330× |
| down    | 40 | 265.7 | 84.0  | 3.16× | 61.6  | 2.65× → 1.365× |
| down    | 64 | 295.1 | 96.7  | 3.05× | 90.6  | 2.61× → **1.067×** |

## Prefill (µs)

| shape | M | start | tuned | gain | mxfp4 | ratio | TFLOP/s |
|---|---|---|---|---|---|---|---|
| gate_up | 512  | 520.4  | 475.4  | 1.09× | 450.5  | 1.055× | 192.0 |
| gate_up | 2048 | 2128.7 | 1792.4 | 1.19× | 1707.7 | 1.050× | 203.7 |
| down    | 512  | 292.0  | 281.6  | 1.04× | 236.7  | 1.189× | 162.1 |
| down    | 2048 | 1124.0 | 937.2  | 1.20× | 893.8  | 1.048× | 194.8 |

203.7 TFLOP/s is 49% of the 412 TF/s fp8 WMMA peak; MXFP4 reaches 52% on the same shape.

## Tuned configuration

| | decode | prefill |
|---|---|---|
| block | DWN=8 (BND=128), 256 threads | EP_WN=2, TN=2 (BNF=64), 8 waves |
| k-block | KB=4 | EP_BK=64 |
| M tile | DTM = ceil(M/16) | TM=8 (512 rows); TM=4 when that starves the grid |
| split-K | **`escha_decode_split_k(nblk, ktiles)`** — see below | n/a |
| resources | 49–82 VGPR, 11.5–18.4 KB LDS, occ 14–16 | 142 VGPR (TM=4) / 230 (TM=8), occ 10 / 6 |

## The split-K rule

Largest single lever, and it cannot be read off N. `down` is N=5120, which at BND=128 is 40
workgroups on a 64-CU part — 24 CUs idle for the whole GEMM, and DKS=8 there is worth 1.8×. But an
N-keyed rule is overfit: at the *same* nblk=40, K=4352 wants DKS=4 while K=8704 wants DKS=8,
because what is being divided is k-work, not columns.

Two forces set the optimum. Each split must keep enough k-tiles to amortize its own workgroup; and
the split writes `DKS*M*N` fp32 partials and reads them back, a cost linear in DKS. Fitting both
against 18 measured `(nblk, ktiles, M)` points — spanning TP=1 and TP=2 per-GPU shapes — gives

```
ks = largest power of two <= 8 with  ktiles/ks >= 36  and  nblk*ks <= 576
```

which needs **no M term**, is exact on 14 of the 18 points and within **2.0%** on the rest. Harness
is `ksfit.hip`; the rule ships as `escha_decode_split_k()`.

| N | K | best ks (measured) | rule |
|---|---|---|---|
| 2560 | 5120 | 8, 8, 8 | 8 |
| 5120 | 4352 | 4, 4, 4 | 4 |
| 5120 | 8704 | 8, 8, 4 | 8 |
| 8704 | 4352 | 8, 4, 4 | 4 |
| 8704 | 5120 | 8, 8, 4 | 8 |
| 17408 | 5120 | 4, 4, 4 | 4 |

(three entries per cell = M of 8, 40, 64)

## What moved the numbers

Found by censusing the **loop body** of the emitted ISA — the whole-kernel census hides the loop
under prologue and epilogue and reads far too low.

1. **Split-K (the single largest win: `down` 110 → 60 µs).** Covered above.
2. **K-blocking the weight fetch.** The original loop staged one 16-wide k-tile between two
   barriers, so each wave had a single weight load in flight — MLP of 1, and 106 GB/s against a
   635 GB/s roofline. Both K=2 and K=3 need exactly two dwords per lane per tile and tiles are
   contiguous, so KB of them now issue back to back into registers.
3. **`v_perm_b32` for the half2 assembly.** The codec is a horizontal fp16 add; feeding
   `v_pk_add_f16` needs two states' halves cross-shuffled, which the compiler built from
   `v_mov_b16` / `v_and_or` / `v_lshl_or` — 49 instructions per four tiles. Two `v_perm_b32` do it.
4. **Packed LDS stores.** Trellis pairs land on adjacent rows of one column and sW is transposed to
   [n][k], so a pair is one contiguous store. Applied to both kernels.
5. **The fp8 decode kernel (+20–26% at M≥40).** f16 WMMA is 207 TF/s against fp8's 412, and f16
   fragments are twice as wide. e4m3 weights stay *more* accurate than MXFP4's own e2m1-with-
   shared-exponent, and the gate agrees — same 1.65e-3. LDS/block 19.6 → 11.5 KB, occupancy 3 → 5
   blocks/CU.
6. **TM=8 on prefill.** A workgroup decodes its whole weight slab, so each weight is decoded
   `ceil(M/BMF)` times — 8× at M=2048 with the original 256-row block.

## Measured and rejected

- **`__umul24` for the quarter-rate `v_mul_lo_u32`** — a wash, re-tested twice (once on the f16
  kernel, once on fp8 + split-K). Three full-rate ops replacing one quarter-rate op is break-even.
- **`v_cvt_pk_fp8_f16`** — does not exist on gfx1201. The f16→f32→fp8 path stays.
- **WM=8 instead of TM=8** at equal 512-row block width (142 VGPR × 16 waves vs 198 × 8) — worse
  everywhere. Registers beat waves; the loop is issue-bound, not latency-bound.
- **TN=4 on prefill** (BNF=128) — worse at TM=4 (198 VGPR, occ 7), and at TM=8 it spills **1146
  VGPRs** and runs 10× slow. A `static_assert` on `TM*TN <= 16` now refuses to build it.
- **DWN=4 and DWN=16** decode block widths, **KB=8** with split-K on, **DKS=8** on gate_up.
- **Halving the barriers** (double-buffered sA, wave-local sW ordering) — within noise. Kept: it is
  free and drops a block-wide barrier per trip.

## Why decode still trails, and why that is not a tuning failure

An `ABL` switch in the decode kernel (measurement-only; wrong results when non-zero) splits the
runtime at gate_up M=8:

| | µs |
|---|---|
| full | 148.5 |
| codec math, **no global loads** | 149.4 |
| loads + LDS + WMMA, **no codec** | 86.8 |

**No-load ≈ full: the weight traffic is entirely hidden.** Escha's 2.469 bits/weight against
MXFP4's 4.25 buys *nothing* at decode, because the trellis is computed rather than looked up —
~5.5 VALU per weight — and that ALU costs more than streaming twice the bytes off a 635 GB/s bus.
The kernel runs at 49 VGPR and **occupancy 16 waves/SIMD with no spills**, so it is not occupancy-
or latency-limited; it is codec-issue-bound, and the residual gap is the format, not the tiling.

Prefill is the opposite regime and lands within 5% of MXFP4, because there each decode amortizes
over a 512-row block and the fp8 matrix pipe is the limiter.

## Identified but not built

**Dequantize-once for prefill.** A workgroup decodes its slab per row block, so at a real 8192-token
prefill chunk each weight is decoded 16 times. A separate dequant-to-fp8 kernel writing an N×K byte
scratch, followed by a plain fp8 GEMM, would cut that to one — worth an estimated ~14% at chunk
8192, against an 89 MB scratch buffer and a second launch. Not worthwhile at M=2048 (redundancy 4,
roughly break-even), and premature while escha has no vLLM integration.
