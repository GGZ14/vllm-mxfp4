// Gate the device trellis decoder against the verified NumPy reference.
//
// The vectors are REAL tiles from the checkpoint, not synthetic words: the trellis reads a
// circular bitstream, so only real data exercises the wrap the way the model does. Both bit rates
// are covered because K=2 takes the aligned fast path and K=3 the generic reader -- different code.
//
// The comparison is EXACT. Every step is integer or fp16 with no reassociation, so the device must
// reproduce the reference bit for bit; a tolerance here would hide precisely the kind of ordering
// bug that already cost a debugging round (see FORMAT.md).
#include "escha_kernels.h"
#include <vector>
#include <cstdio>

template <int K>
__global__ void decode_tile_kernel(const unsigned int *__restrict__ words, float *__restrict__ out) {
  const int lane = threadIdx.x;
  if (lane >= 32) return;
  unsigned int st[8];
  escha_states8<K>(words, lane, st);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int r, c;
    escha_tile_pos(lane, j, &r, &c);
    out[r * 16 + c] = __half2float(escha_decode(st[j]));
  }
}

int main() {
  FILE *f = fopen("vectors.bin", "rb");
  if (!f) { printf("vectors.bin missing (run gen_vectors.py)\n"); return 1; }
  unsigned int magic, n2, n3;
  if (fread(&magic, 4, 1, f) != 1 || magic != 0xE5C7A000u) { printf("bad magic\n"); return 1; }
  if (fread(&n2, 4, 1, f) != 1 || fread(&n3, 4, 1, f) != 1) return 1;
  printf("vectors: %u K=2 tiles, %u K=3 tiles\n", n2, n3);

  unsigned int *dW; float *dO;
  HIP_CHECK(hipMalloc(&dW, 24 * 4));
  HIP_CHECK(hipMalloc(&dO, 256 * 4));
  int bad = 0, total = 0;
  for (unsigned int c = 0; c < n2 + n3; ++c) {
    unsigned int K;
    if (fread(&K, 4, 1, f) != 1) break;
    const int nw = 256 * (int)K / 32;
    std::vector<unsigned int> w(nw);
    std::vector<float> want(256), got(256);
    if (fread(w.data(), 4, nw, f) != (size_t)nw) break;
    if (fread(want.data(), 4, 256, f) != 256) break;
    HIP_CHECK(hipMemcpy(dW, w.data(), nw * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dO, 0, 256 * 4));
    if (K == 2) hipLaunchKernelGGL((decode_tile_kernel<2>), dim3(1), dim3(32), 0, 0, dW, dO);
    else        hipLaunchKernelGGL((decode_tile_kernel<3>), dim3(1), dim3(32), 0, 0, dW, dO);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got.data(), dO, 256 * 4, hipMemcpyDeviceToHost));
    int mism = 0;
    for (int i = 0; i < 256; ++i) if (got[i] != want[i]) ++mism;
    total++;
    if (mism) { ++bad; if (bad <= 3) printf("  case %u (K=%u): %d/256 mismatched, "
                                            "got[0]=%.5f want[0]=%.5f\n", c, K, mism, got[0], want[0]); }
  }
  fclose(f);
  printf("decode gate: %d/%d tiles bit-exact -- %s\n", total - bad, total, bad ? "FAIL" : "PASS");
  hipFree(dW); hipFree(dO);
  return bad != 0;
}
