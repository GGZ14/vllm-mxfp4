#include "escha_kernels.h"
template __global__ void escha_gemm_decode<8,1,1,2>(const __half*, const unsigned int*, float*, int*, __bf16*, int, int, int);
template __global__ void escha_gemm_decode<8,4,1,2>(const __half*, const unsigned int*, float*, int*, __bf16*, int, int, int);
template __global__ void escha_gemm_decode<8,1,4,3>(const __half*, const unsigned int*, float*, int*, __bf16*, int, int, int);
int main(){return 0;}
