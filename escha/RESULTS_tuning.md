# Escha W2 kernels on RDNA4 / R9700 — tuning results

Both kernels benchmarked against the shipped MXFP4 kernels **in one binary**, arms interleaved and
repeated, weights rotated over 4 buffers so the working set clears the 64 MB Infinity Cache.
Correctness re-gated after every change: trellis decode 24/24 **bit-exact**, decode GEMM and prefill
GEMM at 1.65–1.70e-3, which is the independently-computed bf16 rounding floor.

## Where it started and where it ended (µs, lower is better)

### Decode

| shape | M | before | after | gain | mxfp4 | esc/mx before → after |
|---|---|---|---|---|---|---|
| gate_up | 8  | 210.4 | 143.2 | 1.47× | 97.0  | 2.12× → 1.48× |
| gate_up | 40 | 251.3 | 181.8 | 1.38× | 113.6 | 2.25× → 1.60× |
| gate_up | 64 | 284.9 | 191.9 | 1.48× | 155.1 | 1.81× → 1.24× |
| down    | 8  | 192.8 | 109.4 | 1.76× | 81.9  | 2.35× → 1.34× |
| down    | 40 | 265.7 | 139.0 | 1.91× | 100.5 | 2.65× → 1.38× |
| down    | 64 | 295.1 | 152.7 | 1.93× | 114.8 | 2.61× → 1.33× |

### Prefill

| shape | M | before | after | gain | mxfp4 | ratio |
|---|---|---|---|---|---|---|
| gate_up | 512  | 520.4  | 477.6  | 1.09× | 442.6  | 1.079× |
| gate_up | 2048 | 2128.7 | 1778.2 | 1.20× | 1772.5 | **1.003×** |
| down    | 512  | 292.0  | 274.3  | 1.06× | 228.6  | 1.200× |
| down    | 2048 | 1124.0 | 942.9  | 1.19× | 890.6  | 1.059× |

205 TFLOP/s at gate_up M=2048, half the 412 TF/s fp8 WMMA peak, against MXFP4's ~216.

## What actually moved the numbers

Found by censusing the **loop body** of the emitted ISA, not the whole kernel — the whole-kernel
count hides the loop under prologue and epilogue and reads far too low.

1. **K-blocking the weight fetch (decode, 1.9×→1.65× of MXFP4).** The original loop staged ONE
   16-wide k-tile between two `__syncthreads()`, so each wave had exactly one weight load in
   flight — memory-level parallelism of 1, and 106 GB/s against a 635 GB/s roofline. Both K=2 and
   K=3 need exactly two dwords per lane per tile and consecutive tiles are contiguous, so KB tiles
   now issue back to back into registers.
2. **`v_perm_b32` for the half2 assembly.** The codec is a horizontal fp16 add; feeding
   `v_pk_add_f16` needs the halves of two states cross-shuffled, which the compiler built from
   `v_mov_b16` / `v_and_or` / `v_lshl_or` — 49 instructions per four tiles. Two `v_perm_b32` do it.
3. **Packed LDS stores.** Trellis pairs (j, j+1) land on *adjacent rows of one column*, and sW is
   transposed to [n][k], so each pair is one contiguous store: `ds_store_2addr_b32` at 8 per four
   tiles instead of 8 `ds_write_b16` per tile. Same fix applied to prefill (one 16-bit store).
4. **The fp8 decode kernel (+20–26% at M≥40).** The f16 pipe runs at 207 TF/s against fp8's 412,
   and f16 fragments are 16 B per lane against 8. Rounding decoded weights to e4m3 stays *more*
   accurate than MXFP4's own e2m1-with-shared-exponent weights, and the gate confirms it: same
   1.65e-3 as the f16 path. LDS/block drops 19.6 → 11.5 KB, occupancy 3 → 5 blocks/CU.
5. **TM=8 on prefill (1.20× at M=2048).** A workgroup decodes the whole K×BNF slab for its own row
   block, so every weight is decoded `ceil(M/BMF)` times — 8× at M=2048 with the original 256-row
   block. Doubling the block halves that.

## Measured and rejected

- **`__umul24` decomposition of the MCG multiply** — a wash (141.9 vs 143.1 µs). `v_mul_lo_u32` is
  quarter-rate, but three full-rate ops replacing it is break-even.
- **WM=8 instead of TM=8** (same 512-row block, 142 VGPRs over 16 waves rather than 198 over 8) —
  worse everywhere (1889.9 vs 1778.2 at gate_up 2048). Registers beat waves; the loop is
  issue-bound, not latency-bound.
- **Halving the barriers** (double-buffered sA, wave-local sW ordering) — within noise, confirming
  the same conclusion. Kept anyway: it is free and it drops a block-wide barrier per trip.
- **KB=8 everywhere** — best for K=3 and for K=2 at small M, but loses at gate_up M≥40.

## The structural finding

An `ABL` ablation switch in the decode kernel (measurement-only, wrong results when non-zero)
splits the runtime:

| | gate_up M=8 |
|---|---|
| full | 148.5 µs |
| codec math, **no global loads** | 149.4 µs |
| loads + LDS + WMMA, **no codec** | 86.8 µs |

**No-load ≈ full: the weight traffic is entirely hidden.** Escha's 2.469 bits/weight against
MXFP4's 4.25 buys *nothing* at decode, because the trellis is computed, not looked up — roughly
5.5 VALU per weight — and on this part that ALU costs more than streaming twice the bytes from a
635 GB/s bus. At M=8 the codec alone (149 µs) already exceeds MXFP4's *entire* runtime (97 µs), so
no amount of tiling flips the decode result; the remaining 1.24–1.48× is the codec and is
irreducible without changing the format. The 2.469 bits still buy what they always bought:
capacity — weights and KV headroom — not decode speed.

Prefill is the opposite regime and reaches parity, because there each decode is amortized over a
512-row block and the fp8 matrix pipe, not the codec, is the limiter.
