// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>

__global__ void cudakernel(int N, float *C, float *A, float *B)
{
  int i, j, k;
  i = blockIdx.x;
  j = threadIdx.x;
  for(k=0; k<N; k++){
	C[i*N+j] += A[i*N+k] * B[k*N+j];
  }
}

int main(int argc, char **argv)
{
  int i, j;
  int N;
  float *A, *B, *C;
  float *dA, *dB, *dC;

  N = 8;
  A = (float*)malloc(sizeof(float)*N*N);
  B = (float*)malloc(sizeof(float)*N*N);
  C = (float*)malloc(sizeof(float)*N*N);
  cudaMalloc((void**)&dA, sizeof(float)*N);
  cudaMalloc((void**)&dB, sizeof(float)*N);
  cudaMalloc((void**)&dC, sizeof(float)*N);

  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  C[i*N+j] = 0.0f;
	  A[i*N+j] = (float)(i) + (float)j/10.0f;
	  B[i*N+j] = (float)(i) + (float)j/10.0f;
	}
  }

  printf("A\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", A[i*N+j]);
	}
	printf("\n");
  }
  printf("B\n");
  for(i=0;i<N;i++) {
	for(j=0;j<N;j++){
	  printf(" %8.2f", B[i*N+j]);
	}
	printf("\n");
  }
  printf("C (before)\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", C[i*N+j]);
	}
	printf("\n");
  }

  cudaMemcpy(dA, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float)*N, cudaMemcpyHostToDevice);
  dim3 grids;
  dim3 blocks;
  grids = dim3(2, 1, 1);
  blocks = dim3(64, 1 ,1);
  cudakernel<<<grids,blocks>>>(N, dC, dA, dB);
  cudaMemcpy(C, dC, sizeof(float)*N, cudaMemcpyDeviceToHost);

  printf("C (after)\n");
  for(i=0;i<N;i++){
	for(j=0;j<N;j++){
	  printf(" %8.2f", C[i*N+j]);
	}
	printf("\n");
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  return 0;
}
