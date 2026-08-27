"""Derive a header of the MXFP4 kernels so cmp.hip / abl.hip can compile both families into one
binary and interleave them in a single timing loop.

Generated rather than committed: a checked-in copy would silently drift from
radiance_mxfp4_fp8.hip and the A/B would then compare against a kernel we no longer ship.
"""
import os

src = os.path.join(os.path.dirname(__file__), "..", "radiance_mxfp4_fp8.hip")
out = os.path.join(os.path.dirname(__file__), "mxfp4_kernels.h")
s = open(src).read()
s = s.replace("#include <pybind11/pybind11.h>\n", "")
s = s.replace("namespace py = pybind11;", "// (pybind alias stripped)")
s = s[: s.index("PYBIND11_MODULE")]
open(out, "w").write("#pragma once\n" + s)
print(f"wrote {out}")
