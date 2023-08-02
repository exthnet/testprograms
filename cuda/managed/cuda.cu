// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
//#include <cuda_rt.h>

__global__ void gpukernel(int N, double *C, double *A, double *B)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  C[id] = A[id] * B[id];
}


int main(int argc, char **argv)
{
  int i, j, N;
  double *A, *B, *C;
  double *dA, *dB, *dC;

  N = 4;
  A = (double*)malloc(sizeof(double)*N*N);
  B = (double*)malloc(sizeof(double)*N*N);
  C = (double*)malloc(sizeof(double)*N*N);

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  C[i*N+j] = 0.0f;
	  B[i*N+j] = 2.0f;
	  A[i*N+j] = (double)(i)/(double)(N)*1000.0 + (double)(j)/(double)(N);
	}
  }

  cudaMalloc((void**)&dA, sizeof(double)*N*N);
  cudaMalloc((void**)&dB, sizeof(double)*N*N);
  cudaMalloc((void**)&dC, sizeof(double)*N*N);

  printf("A\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", A[i*N+j]);
	}
	printf("\n");
  }
  printf("B\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", B[i*N+j]);
	}
	printf("\n");
  }
  printf("C (before)\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", C[i*N+j]);
	}
	printf("\n");
  }

  cudaMemcpy(dA, A, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(double)*N*N, cudaMemcpyHostToDevice);

  dim3 grids;
  dim3 blocks;
  grids = dim3(4, 1, 1);
  blocks = dim3(4, 1 ,1);
  gpukernel<<<grids,blocks>>>(N, dC, dA, dB);

  cudaMemcpy(C, dC, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

  printf("C (after)\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", C[i*N+j]);
	}
	printf("\n");
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  return 0;
}
