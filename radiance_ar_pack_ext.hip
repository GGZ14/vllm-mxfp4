// radiance_ar_pack_ext.hip: pybind surface for the rotated 6-bit all-reduce.
// Device code is in radiance_ar_pack.h so it can also be compiled without Python.
#include "radiance_ar_pack.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(radiance_ar_pack_ext, m) {
  m.def("all_reduce_pack", &all_reduce_pack);
  m.attr("MAX_BLOCKS") = RADIANCE_MAX_BLOCKS;
  m.attr("GROUP") = GROUP;
  m.attr("BITS") = BITS;
  m.attr("CHUNK_ELEMS") = CHUNK_ELEMS;
}
