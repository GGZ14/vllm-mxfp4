// One-hot A isolates the weight path: with A[m][k] = (k == m), C[m][n] must equal W[m][n] exactly
// (times the per-token scale). Any error here is weight staging or decode, with no accumulation
// and no cancellation to hide behind.
#include "escha_kernels.h"
#include <vector>
#include <cstdio>
#include <cmath>
int main() {
  FILE *f = fopen("prefill_vectors.bin", "rb");
  unsigned int magic, ncase; fread(&magic,4,1,f); fread(&ncase,4,1,f);
  int K,M,N,Kdim; fread(&K,4,1,f); fread(&M,4,1,f); fread(&N,4,1,f); fread(&Kdim,4,1,f);
  const int words=(Kdim/16)*(N/16)*(256*K/32);
  std::vector<unsigned int> hw(words);
  fread(hw.data(),4,words,f); fclose(f);
  M = 64;                                        // one-hot over the first 64 k
  std::vector<unsigned char> hA((size_t)M*Kdim, 0);
  std::vector<float> hAs(M, 1.0f);
  for (int m=0;m<M;++m) hA[(size_t)m*Kdim+m] = 0x38;    // e4m3 1.0
  unsigned int *dW; unsigned char *dA; float *dAs; __bf16 *dC;
  HIP_CHECK(hipMalloc(&dW,(size_t)words*4)); HIP_CHECK(hipMalloc(&dA,hA.size()));
  HIP_CHECK(hipMalloc(&dAs,M*4)); HIP_CHECK(hipMalloc(&dC,(size_t)M*N*2));
  HIP_CHECK(hipMemcpy(dW,hw.data(),(size_t)words*4,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dAs,hAs.data(),M*4,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC,0,(size_t)M*N*2));
  const int BNF=EP_WN*2*16;
  hipLaunchKernelGGL((escha_gemm_prefill<2,2>), dim3((N+BNF-1)/BNF,(M+EP_BMF-1)/EP_BMF),
                     dim3(EP_NTHREADS),0,0,dA,dW,dAs,dC,M,N,Kdim);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned short> g((size_t)M*N);
  HIP_CHECK(hipMemcpy(g.data(),dC,g.size()*2,hipMemcpyDeviceToHost));
  FILE *o=fopen("onehot_out.bin","wb"); fwrite(g.data(),2,g.size(),o); fclose(o);
  printf("wrote onehot_out.bin  M=%d N=%d Kdim=%d K=%d\n", M,N,Kdim,K);
  return 0;
}
