#!/usr/bin/env python3
"""Make the compressed all-reduce's wire width env-selectable (RADIANCE_AR_QBITS = 6 | 5 | 4).

The one-shot P2P all-reduce is ~80% PCIe wire on this box: each R9700's root port is Gen5 x8
(27.7 GB/s), so the 6.25-bit payload of an 80 MiB prefill message is 1.18 ms of the 1.47 ms call.
libr4d rx8 carries the same kernel at 5 and 4 bits per symbol (ar_oneshot_2rank_wht5 / wht4);
this patch lets the serve ask for one. select() is asked with bits=N only when N != 6, so an
older libr4d (no `bits` constraint on its wht6 entry) keeps answering exactly as before, and a
libr4d without the requested width logs and falls back to 6.

Idempotent; inert without the env var (default 6 = the shipped kernel)."""
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "radiance_allreduce.py"
SENTINEL = "RADIANCE_AR_QBITS"

OLD = """                qname = qext.select("allreduce", world_size=self.world_size, exact=0,
                                    dtype="bf16", numel=self._qgroup)
                if qname is None:
                    raise RuntimeError("no lossy all-reduce kernel in this build")
"""
NEW = """                # RADIANCE_AR_QBITS (patch_ar_qbits.py): symbol width on the wire. 6 is the
                # shipped kernel; 5 and 4 need a libr4d that carries them (rx8+).
                self.qbits = int(os.environ.get("RADIANCE_AR_QBITS", "6"))
                qgeom = dict(world_size=self.world_size, exact=0, dtype="bf16",
                             numel=self._qgroup)
                if self.qbits != 6:
                    qgeom["bits"] = self.qbits
                qname = qext.select("allreduce", **qgeom)
                if qname is None and self.qbits != 6:
                    _log(f"AR_QUANT: no {self.qbits}-bit all-reduce in this libr4d; using 6")
                    self.qbits = 6
                    qgeom.pop("bits")
                    qname = qext.select("allreduce", **qgeom)
                if qname is None:
                    raise RuntimeError("no lossy all-reduce kernel in this build")
                # the table answers wht6 for any width it does not carry; report what actually runs
                if qname[-1].isdigit() and int(qname[-1]) != self.qbits:
                    _log(f"AR_QUANT: {self.qbits}-bit not in this libr4d; {qname} serves")
                    self.qbits = int(qname[-1])
"""
OLD2 = """                _log(f"AR_QUANT ON (rotated {int(qext.AR_WHT6_BITS)}-bit packed payload; \""""
NEW2 = """                _log(f"AR_QUANT ON ({qname}: rotated {self.qbits}-bit packed payload; \""""

src = TARGET.read_text()
if SENTINEL in src:
    print(f"  NOOP  {TARGET.name} already applied")
    raise SystemExit(0)
for old in (OLD, OLD2):
    if src.count(old) != 1:
        print(f"  FAIL  anchor matched {src.count(old)}x, expected 1:\n{old}")
        raise SystemExit(1)
src = src.replace(OLD, NEW, 1).replace(OLD2, NEW2, 1)
TARGET.write_text(src)
print(f"  OK    {TARGET.name}")
