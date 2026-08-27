"""NumPy reference for the EXL3 trellis codec as used by the escha W2 checkpoint.

Ported from exllamav3 (MIT, (c) 2025 Turboderp): exl3_dq.cuh, codebook.cuh, quantize.py.
Verified against the fp8 base checkpoint -- see FORMAT.md for the evidence and for the two
traps that make a wrong port look right.

This file is the GATE for the HIP kernel: the kernel is checked against it, so it is written for
obviousness rather than speed.
"""
import numpy as np

MCG = np.uint32(0xCBAC1FED)
MUL1 = np.uint32(0x83DCD12D)
CODEBOOK_SCALE = 1.24371088


def tensor_core_perm():
    """perm[e] = row*16 + col: where encoded symbol e lands in the 16x16 tile."""
    p = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        rs = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = t // 4
        for j, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rs):
                p[t * 8 + j * 4 + i] = r * 16 + c
    return p


def decode_3inst(state, cb=1):
    """16-bit trellis state -> value. cb=1 (MCG) is what escha uses; see FORMAT.md.

    lop3(a, b, c, 0x6a) == (a & b) ^ c. The result is two fp16 masked to sign+mantissa with the
    exponent forced, then summed -- QTIP's 3INST Gaussian codebook.
    """
    x = state.astype(np.uint32)
    if cb == 1:
        x = (x * MCG).astype(np.uint32)
    elif cb == 0:
        x = (x * np.uint32(89226354) + np.uint32(64248484)).astype(np.uint32)
    elif cb == 2:
        x = (x * MUL1).astype(np.uint32)
        s = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
             + np.uint32(0x6400)).astype(np.uint32)
        h = (s & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        return h * np.uint16(0x1EEE).view(np.float16).astype(np.float32) \
            + np.uint16(0xC931).view(np.float16).astype(np.float32)
    else:
        raise ValueError(cb)
    x = ((x & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)).astype(np.uint32)
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
    hi = (x >> 16).astype(np.uint16).view(np.float16)
    # The add is fp16, NOT fp32. CUDA does __hadd here, so widening first would make the reference
    # disagree with every real decoder by up to half an fp16 ulp -- which the bit-exact device gate
    # duly caught (got 2.90234 vs want 2.90137). Harmless for correlation, fatal for a bit gate.
    return (lo + hi).astype(np.float32)


def tile_states(words, K):
    """One tile's packed words -> its 256 trellis states, exactly as exl3_dq.cuh reads them.

    The tile is viewed as uint32 (little-endian pairing of the stored int16), and
    `fshift(b, a, s) = ((a << 32) | b) >> s` puts the EARLIER word in the HIGH half, so stream
    position advances toward LOWER bit positions. Lane L covers symbols 8L..8L+7.

    Reading this back as a continuous MSB-first uint16 stream -- the obvious interpretation of the
    packer -- is wrong and fails SILENTLY: it still yields Gaussian values of the right variance,
    and only the element order is off. That cost a full round of debugging; see FORMAT.md.
    """
    u32 = np.ascontiguousarray(words).view(np.uint16).view(np.uint32)
    nw = u32.size                                     # 256*K/32 uint32
    st = np.zeros(256, dtype=np.uint32)
    for L in range(32):
        t = 8 * L
        if K == 2:                                    # dq8_aligned_2bits
            i1 = (t >> 4) % nw
            i0 = (i1 + nw - 1) % nw
            b = ((int(u32[i0]) << 32) | int(u32[i1])) >> (((~t) & 8) << 1)
            for j in range(8):
                st[t + j] = (b >> (14 - 2 * j)) & 0xFFFF
        else:                                         # dq8<bits, cb, align>
            b1 = (t + 257) * K
            b0 = b1 - 16
            b2 = b1 + K * 7
            i0 = b0 // 32                              # UNWRAPPED: the shift is derived from these
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2                    # so it must not use the wrapped index
            merged = (int(u32[i0 % nw]) << 32) | int(u32[i2 % nw])   # wrap only the array access
            for j in range(8):                        # w7 at s2, each earlier symbol +K
                st[t + (7 - j)] = (merged >> (s2 + K * j)) & 0xFFFF
    return st


def decode_tile(words, K, cb=1):
    """One tile's packed words -> a 16x16 float32 tile, row = K index, col = N index."""
    vals = decode_3inst(tile_states(words, K), cb)
    tile = np.zeros(256, dtype=np.float32)
    tile[tensor_core_perm()] = vals
    return tile.reshape(16, 16)


def hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def reconstruct(code, rin, rout, K, cb=1, had=128):
    """Full projection: escha tensors -> W[in, out]. nn.Linear wants the transpose."""
    kt, nt = code.shape[0], code.shape[1]
    W = np.zeros((kt * 16, nt * 16), dtype=np.float64)
    for i in range(kt):
        for j in range(nt):
            W[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = decode_tile(code[i, j], K, cb)
    H = hadamard(had) / np.sqrt(had)
    W = (H @ W.reshape(-1, had, W.shape[1])).reshape(W.shape)
    W *= np.asarray(rin, dtype=np.float64)[:, None]
    W = (W.reshape(W.shape[0], -1, had) @ H).reshape(W.shape)
    W *= np.asarray(rout, dtype=np.float64)[None, :]
    return W
