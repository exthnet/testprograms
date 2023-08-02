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

  N = 4;
  cudaMallocManaged((void**)&A, sizeof(double)*N*N);
  cudaMallocManaged((void**)&B, sizeof(double)*N*N);
  cudaMallocManaged((void**)&C, sizeof(double)*N*N);

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  C[i*N+j] = 0.0f;
	  B[i*N+j] = 2.0f;
	  A[i*N+j] = (double)(i)/(double)(N)*1000.0 + (double)(j)/(double)(N);
	}
  }

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

  dim3 grids;
  dim3 blocks;
  grids = dim3(4, 1, 1);
  blocks = dim3(4, 1 ,1);
  gpukernel<<<grids,blocks>>>(N, C, A, B);
  cudaDeviceSynchronize();

  printf("C (after)\n");
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  printf(" %2.4f", C[i*N+j]);
	}
	printf("\n");
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
