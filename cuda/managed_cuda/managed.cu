/* -*- c++ -*- */
#include <stdio.h>
#include <cuda_runtime.h>

#define CHK_DO(o) if(cudaSuccess!=o){printf("%d failed\n",__LINE__);}

__global__ void gpukernel(int N, double *a, double *b, double*c)
{
  int i, j, k;
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  for(k=0; k<10; k++){
		c[i*N+j] += a[i*N+j] * b[i*N+j];
	  }
	}
  }
}

int main(int argc, char **argv)
{
  cudaStream_t s1;
  double *A1, *B1, *C1;
  cudaError_t cudaRet;
  int N, i, j, k;
  if(argc<2){
	N = 10;
  }else{
	N = atoi(argv[1]);
  }
  printf("N = %d\n", N);
  CHK_DO(cudaMallocManaged((void**)&A1, sizeof(double)*N*N, cudaMemAttachGlobal));
  CHK_DO(cudaMallocManaged((void**)&B1, sizeof(double)*N*N, cudaMemAttachGlobal));
  CHK_DO(cudaMallocManaged((void**)&C1, sizeof(double)*N*N, cudaMemAttachGlobal));

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  A1[i*N+j] = 1.0;
	  B1[i*N+j] = 2.0;
	  C1[i*N+j] = 0.0;
	}
  }
  CHK_DO(cudaStreamCreate(&s1));
  CHK_DO(cudaStreamSynchronize(s1));
  gpukernel<<<N,N,0,s1>>>(N,A1,B1,C1);

  CHK_DO(cudaStreamSynchronize(s1));
  CHK_DO(cudaStreamDestroy(s1));

  cudaFree(A1);
  cudaFree(B1);
  cudaFree(C1);
  return 0;
}
