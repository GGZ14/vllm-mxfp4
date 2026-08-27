// Does gfx1201 have a hardware f32->bf16 convert, and what does the plain (__bf16) cast cost?
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
typedef unsigned short u16;
__global__ void cast_plain(const float *a, __bf16 *o) { o[threadIdx.x] = (__bf16)a[threadIdx.x]; }
#if defined(AR_TRY_PK)
__global__ void cast_pk(const float *a, __bf16 *o) {
  // packed convert, if the builtin exists
  short2 v = __builtin_amdgcn_cvt_pk_bf16_f32(a[threadIdx.x * 2], a[threadIdx.x * 2 + 1]);
  ((short2 *)o)[threadIdx.x] = v;
}
#endif
int main() { return 0; }
