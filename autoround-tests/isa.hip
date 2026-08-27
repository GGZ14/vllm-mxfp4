// Device-only translation unit: instantiate the shipped kernels so their ISA can be inspected.
#include "../radiance_autoround_kernels.h"
template __global__ void ar_int4_fp8_gemm_decode<8, 1, 1, true, 0>(
    const unsigned char *, const unsigned int *, const __half *, const float *, float *, int *,
    __bf16 *, int, int, int);
template __global__ void ar_int4_fp8_gemm_prefill<2, true, 0>(
    const unsigned char *, const unsigned int *, const __half *, const float *, __bf16 *, int,
    int, int);
template __global__ void ar_int4_fp8_gemm_prefill<4, true, 0>(
    const unsigned char *, const unsigned int *, const __half *, const float *, __bf16 *, int,
    int, int);
