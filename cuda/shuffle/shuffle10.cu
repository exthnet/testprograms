// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void gpukernel(float *B, float *A)
{
  int tid = threadIdx.x;
  float tmp = A[tid];
  if(tid>=16){
	tmp = __shfl_up_sync(0xffffffff, tmp, 2, warpSize);
  }
  if(tid<16){
	tmp = __shfl_up_sync(0xffffffff, tmp, 1, warpSize);
  }
  B[tid] = tmp;
}


int main(int argc, char **argv)
{
  int i, N;
  float *A, *B;
  float *dA, *dB;

  N = 32;
  A = (float*)malloc(sizeof(float)*N);
  B = (float*)malloc(sizeof(float)*N);

  for(i=0;i<N;i++){
	A[i] = (float)(i+1);
	B[i] = 0.0f;
  }

  cudaMalloc((void**)&dA, sizeof(float)*N);
  cudaMalloc((void**)&dB, sizeof(float)*N);

  printf("A\n");
  for(i=0; i<N; i++){
	printf(" %2.0f", A[i]);
  }
  printf("\n");
  printf("B (before)\n");
  for(i=0; i<N; i++){
	printf(" %2.0f", B[i]);
  }
  printf("\n");

  cudaMemcpy(dA, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  gpukernel<<<1,32>>>(dB, dA);
  cudaDeviceSynchronize();
  cudaMemcpy(B, dB, sizeof(float)*N, cudaMemcpyDeviceToHost);

  printf("B (after)\n");
  for(i=0; i<N; i++){
	printf(" %2.0f", B[i]);
  }
  printf("\n");

  cudaFree(dA); cudaFree(dB);
  free(A); free(B);
  return 0;
}
