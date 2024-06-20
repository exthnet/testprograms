// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>

__global__ void gpukernel(int N, float *C, float *A, float *B)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id<N)C[id] += A[id] * B[id];
}


int main(int argc, char **argv)
{
  int i, N, x;
  float *A, *B, *C;
  float *dA, *dB, *dC;

  x = 10;
  N = 128;
  A = (float*)malloc(sizeof(float)*N);
  B = (float*)malloc(sizeof(float)*N);
  C = (float*)malloc(sizeof(float)*N);

  for(i=0;i<N;i++){
	C[i] = 0.0f;	B[i] = 2.0f;
	A[i] = (float)(i+1)/(float)(N);
  }

  cudaMalloc((void**)&dA, sizeof(float)*N);
  cudaMalloc((void**)&dB, sizeof(float)*N);
  cudaMalloc((void**)&dC, sizeof(float)*N);

  printf("A\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", A[i]);
	}else{
	  printf(" %2.4f", A[i]);
	}
  }
  printf("\n");
  printf("B\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", B[i]);
	}else{
	  printf(" %2.4f", B[i]);
	}
  }
  printf("\n");
  printf("C (before)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", C[i]);
	}else{
	  printf(" %2.4f", C[i]);
	}
  }
  printf("\n");

  cudaMemcpy(dA, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float)*N, cudaMemcpyHostToDevice);

  dim3 grids;
  dim3 blocks;
  grids = dim3(4, 1, 1);
  blocks = dim3(64, 1 ,1);
  gpukernel<<<grids,blocks>>>(N, dC, dA, dB);

  cudaMemcpy(C, dC, sizeof(float)*N, cudaMemcpyDeviceToHost);

  printf("C (after)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.4f\n", C[i]);
	}else{
	  printf(" %2.4f", C[i]);
	}
  }
  printf("\n");

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  return 0;
}
