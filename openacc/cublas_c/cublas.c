#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

int main(int argc, char **argv)
{
  int i, j;
  int N;
  float *A, *B, *C;
  float *dA, *dB, *dC;
  float alpha=1.0f, beta=1.0f;
  cublasHandle_t handle;

  N = 8;
  A = (float*)malloc(sizeof(float)*N*N);
  B = (float*)malloc(sizeof(float)*N*N);
  C = (float*)malloc(sizeof(float)*N*N);
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

  cudaMalloc((void**)&dA, sizeof(float)*N*N);
  cudaMalloc((void**)&dB, sizeof(float)*N*N);
  cudaMalloc((void**)&dC, sizeof(float)*N*N);
  cudaMemcpy(dA, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
  cudaMemcpy(C, dC, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

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
