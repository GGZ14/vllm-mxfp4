# The escha W2 format, recovered and verified

`EschaLabs/Qwen3.8-27B-Escha-W2` is **EXL3 trellis quantization** (exllamav3, MIT, (c) Turboderp).
The escha runtime's `THIRD_PARTY_NOTICES.md` names exllamav3 as "quantization codec *format*
referenced by the escha decode kernels"; every structural detail below was then confirmed against
that source and against the model's own bytes.

There is no published spec and no PyTorch reference path (the README says the numbers come from
their CUDA-only SGLang build), so each step below is verified rather than assumed.

## Tensor mapping

| escha | EXL3 | note |
|---|---|---|
| `escha_code` I16 `[K/16, N/16, 256*bits/16]` | `trellis` | 32 words at 2 bits, 48 at 3 |
| `escha_rin` F16 `[K]` | `suh` | per-input-channel |
| `escha_rout` F16 `[N]` | `svh` | per-output-channel |
| `escha_config` I32 `[6]` | -- | `[16, bits, cb, 1, in_features, out_features]` |
| `escha_s_in`/`escha_s_out` F32 | -- | escha's own, ~1.0 +/- 0.02, NOT in EXL3 |
| `bias` F16 | -- | README: the runtime does NOT apply it |

Realized rate is **2.469 bits/weight**, not the 2.0 in `quantize_config.json`: `up_proj` and
`down_proj` are 3-bit, everything else 2-bit. `in_proj_a`, `in_proj_b` and `lm_head` are excluded;
embeddings are int8.

## Reconstruction

    W_inner[K, N] = trellis_decode(escha_code)          # in the codebook domain
    W = had_l(W_inner, 128);  W *= rin[:, None]
    W = had_r(W, 128);        W *= rout[None, :]        # W is [in, out]; nn.Linear wants the transpose

`had_l/had_r` are the normalised Sylvester Hadamard of size 128 (`H/sqrt(128)`, symmetric and
orthogonal, so it is its own inverse -- which is what makes the inverse test below possible).

## Codebook: cb=1 (MCG), verified

    v = state * 0xCBAC1FED
    v = (v & 0x8fff8fff) ^ 0x3b603b60        # lop3(a,b,c,0x6a) == (a & b) ^ c
    value = fp16_lo(v) + fp16_hi(v)

Two fp16 numbers, masked to sign+mantissa with the exponent forced, summed: QTIP's "3INST" Gaussian
codebook. `cb=0` and `cb=2` exist in EXL3 and are NOT what this checkpoint uses -- measured, they
score 0.005 and 0.002 where cb=1 scores 0.907.

## Bit layout: the part that is easy to get wrong

`pack_trellis` (pack.cu) writes MSB-first into 16-bit words, in 16 spans of 16 symbols. Reading it
back as a continuous MSB-first uint16 stream **is wrong** and fails silently -- it still yields
plausible Gaussian values with the right variance, and only the element arrangement is off.

The decoder (`dq8_aligned_2bits`, exl3_dq.cuh) reads the tile as **16 uint32** with LSB-oriented
shifts:

    i1 = t_offset >> 4;  i0 = (i1 + 15) & 15
    b  = ((u32[i0] << 32) | u32[i1]) >> (((~t_offset) & 8) << 1)
    w7 = b & 0xffff;  w6 = (b >> 2) & 0xffff;  ...  w0 = (b >> 14) & 0xffff

so symbol `t_offset + j` uses the 16-bit window at shift `14 - 2j`. Lane L covers symbols 8L..8L+7.
Consecutive states overlap by `16 - bits` bits -- that overlap is the trellis, and it is why the
codes look like maximum entropy when inspected as independent int16 (all 65536 values present,
every bit at P=0.5).

Tile elements land via the forward `tensor_core_perm` (`tile[perm] = values`), no transpose, and
`code[i, j]` maps to `W_inner[16i:16i+16, 16j:16j+16]` with the K-tile index major.

## How each claim was verified

1. **Bit layout** -- reference-free: in a valid trellis the low `16-bits` bits of state *t* equal
   the high `16-bits` bits of state *t+1*. Holds for **1275/1275** consecutive pairs at both K=2 and
   K=3, against a 0.006% chance rate. Note this test CANNOT see a constant window offset, which is
   exactly the bug it let through on the first attempt.
2. **Ground truth is valid** -- `corr(|escha_rout|, per-output-channel norm of the fp8 base)` is
   **1.0000** on every projection tested. The end-to-end fine-tune mentioned in the model card does
   not move the weights enough to invalidate weight-level comparison.
3. **Transform chain** -- pulling the fp8 reference back through the inverse gives inner-domain
   `std = 1.2318` against EXL3's `codebook_scale = 1.24371088`. That constant does not appear by
   accident.
4. **Whole decode** -- element-wise correlation against the fp8 base over a 128x128 block:

   | projection | corr | rel err |
   |---|--:|--:|
   | layer 0 mlp.gate_proj | +0.9068 | 0.446 |
   | layer 0 linear_attn.out_proj | +0.9423 | 0.341 |
   | layer 20 mlp.gate_proj | +0.9421 | 0.339 |

   The residual is the 2-bit quantisation error itself. Every competing hypothesis scored <= 0.046.

## A trap worth recording

Sorted-value correlation is **not** evidence. Two independent samples of 256 from the same Gaussian
always have matching sorted values, so an early "sorted corr = 0.995, therefore the codec is right
and only the order is wrong" reading was a false positive; forcing a one-to-one assignment dropped
agreement to 1/256, i.e. random. Only element-wise correlation over a large block discriminates.
