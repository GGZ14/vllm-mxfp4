#!/usr/bin/env python3
"""Teach radiance_allreduce.py the three-rank one-shot kernel (r4d ar_oneshot_3rank_exact).

The shipped module is written for exactly two ranks in four places: it opens ONE peer's IPC
handles (`peer = 1 - self.rank`), sizes scratch and flags for one sender, launches the 2-rank
entry point's argument list, and install_custom_ar attaches only when the TP group is 2 wide.
At TP=3 every one of those silently hands the reduction to RCCL. This patch makes the peer
list, the scratch layout and the launch follow the world size, and lets a 3-wide group attach.

Which kernel serves a group is still the library's answer (`r4d.select("allreduce",
world_size=...)`), so an r4d.so without the 3-rank unit leaves TP=3 on RCCL exactly as today,
with the same "no exact all-reduce kernel for world_size=3" line in the log. TP=2 is untouched:
the peer list has one entry, the scratch is the same 2 slots, and the launch is the same call.

Scratch at ws=3 is 2 receive regions x 2 slots x max_bytes (4 x max_bytes, vs 2 x at ws=2), so
the launcher caps RADIANCE_AR_MAX_KB at 2048 there -- decode-size messages only; prefill rides
RCCL until the third card's link has been measured (TP3_PADDING_PLAN.md, Gate M). The
compressed (wht6) path has no 3-rank kernel and disables itself through the same select().

Runs after patch_ar_maxbytes.py (independent anchors). Idempotent; fatal-loud on drift."""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
AR = SP / "radiance_allreduce.py"

ALLOC_ANCHOR = (
    "            # double-buffered scratch (2 slots) + per-block flags, both IPC-shared\n"
    "            self._scratch, sc_h, fine_used = self._alloc(2 * self.max_bytes, fine)\n"
    "            self._flags, fl_h, _ = self._alloc(maxb * 4, fine)\n"
    "            sc_handles = [None] * self.world_size\n"
    "            fl_handles = [None] * self.world_size\n"
    "            dist.all_gather_object(sc_handles, sc_h, group=group)\n"
    "            dist.all_gather_object(fl_handles, fl_h, group=group)\n"
    "            peer = 1 - self.rank\n"
    "            self._peer_scratch = ext.ar_ipc_open(sc_handles[peer])\n"
    "            self._peer_flags = ext.ar_ipc_open(fl_handles[peer])\n"
)
ALLOC_NEW = (
    "            # double-buffered scratch (2 slots) + per-block flags, both IPC-shared. One receive\n"
    "            # region per SENDER (patch_ar_3rank.py): ws=2 keeps its single region, ws=3 holds\n"
    "            # two, each 2 slots x max_bytes, and two flag rows of maxb.\n"
    "            self.n_regions = self.world_size - 1\n"
    "            self._scratch, sc_h, fine_used = self._alloc(self.n_regions * 2 * self.max_bytes, fine)\n"
    "            self._flags, fl_h, _ = self._alloc(self.n_regions * maxb * 4, fine)\n"
    "            sc_handles = [None] * self.world_size\n"
    "            fl_handles = [None] * self.world_size\n"
    "            dist.all_gather_object(sc_handles, sc_h, group=group)\n"
    "            dist.all_gather_object(fl_handles, fl_h, group=group)\n"
    "            # peers in ascending rank order: the 3-rank kernel derives each region from that\n"
    "            self.peers = [r for r in range(self.world_size) if r != self.rank]\n"
    "            self._peer_scratch_all = [ext.ar_ipc_open(sc_handles[p]) for p in self.peers]\n"
    "            self._peer_flags_all = [ext.ar_ipc_open(fl_handles[p]) for p in self.peers]\n"
    "            # the 2-rank names, used by the 2-rank launch below and by radiance_arnq\n"
    "            self._peer_scratch = self._peer_scratch_all[0]\n"
    "            self._peer_flags = self._peer_flags_all[0]\n"
)

LAUNCH_ANCHOR = (
    "        else:\n"
    "            self._ar_exact(\n"
    "                self._peer_scratch, self._scratch, self._peer_flags, self._flags,\n"
    "                self._seq.data_ptr(), self.slot16,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        return out\n"
)
LAUNCH_NEW = (
    "        elif self.world_size == 3:\n"
    "            # patch_ar_3rank.py: (my scratch, lower peer, higher peer, my flags, lower peer\n"
    "            # flags, higher peer flags, seq, slot words, rank, ...). Same block heuristic.\n"
    "            self._ar_exact(\n"
    "                self._scratch, self._peer_scratch_all[0], self._peer_scratch_all[1],\n"
    "                self._flags, self._peer_flags_all[0], self._peer_flags_all[1],\n"
    "                self._seq.data_ptr(), self.slot16, self.rank,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        else:\n"
    "            self._ar_exact(\n"
    "                self._peer_scratch, self._scratch, self._peer_flags, self._flags,\n"
    "                self._seq.data_ptr(), self.slot16,\n"
    "                inp.data_ptr(), out.data_ptr(), inp.numel(),\n"
    "                _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,\n"
    "                self.drain, self.acq,\n"
    "            )\n"
    "        return out\n"
)

QUANT_ANCHOR = (
    "        if not self.ar_quant or self._qext is None:\n"
    "            return False\n"
    "        if inp.dtype not in (torch.bfloat16, torch.float16):\n"
)
QUANT_NEW = (
    "        if not self.ar_quant or self._qext is None:\n"
    "            return False\n"
    "        if self.world_size != 2:      # patch_ar_3rank.py: no compressed 3-rank kernel\n"
    "            return False\n"
    "        if inp.dtype not in (torch.bfloat16, torch.float16):\n"
)

GATE_ANCHOR = (
    "            if \"tp\" in getattr(self, \"unique_name\", \"\") and getattr(self, \"world_size\", 1) == 2:\n"
)
GATE_NEW = (
    "            # ws 2 or 3 (patch_ar_3rank.py); which of them has a kernel is r4d.select's answer\n"
    "            if \"tp\" in getattr(self, \"unique_name\", \"\") and getattr(self, \"world_size\", 1) in (2, 3):\n"
)


def main():
    apply(AR, ALLOC_ANCHOR, ALLOC_NEW, "patch_ar_3rank.py): ws=2 keeps its single region",
          "all-reduce 3-rank: peer list / scratch regions")
    apply(AR, LAUNCH_ANCHOR, LAUNCH_NEW, "patch_ar_3rank.py: (my scratch, lower peer",
          "all-reduce 3-rank: launch arm")
    apply(AR, QUANT_ANCHOR, QUANT_NEW, "patch_ar_3rank.py: no compressed 3-rank kernel",
          "all-reduce 3-rank: quant path ws gate")
    apply(AR, GATE_ANCHOR, GATE_NEW, "ws 2 or 3 (patch_ar_3rank.py)",
          "all-reduce 3-rank: install gate")


if __name__ == "__main__":
    main()
