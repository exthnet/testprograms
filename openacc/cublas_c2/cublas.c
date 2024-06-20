#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <sys/time.h>

int main(int argc, char **argv)
{
  int i, j;
  int N;
  float *A, *B, *C;
  float *dA, *dB, *dC;
  float alpha=1.0f, beta=1.0f;
  cublasHandle_t handle;
  cudaStream_t stream;
  struct timeval tv1, tv2;

  if(argc!=2){
	printf("usage: %s N\n", argv[0]);
	return -1;
  }

  N = atoi(argv[1]);
  printf("N = %d\n", N);

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

  cudaMalloc((void**)&dA, sizeof(float)*N*N);
  cudaMalloc((void**)&dB, sizeof(float)*N*N);
  cudaMalloc((void**)&dC, sizeof(float)*N*N);
  cudaMemcpy(dA, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cublasCreate(&handle);
  cublasGetStream(handle, &stream);
  cudaStreamSynchronize(stream);
  gettimeofday(&tv1,NULL);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
  cublasGetStream(handle, &stream);
  cudaStreamSynchronize(stream);
  gettimeofday(&tv2,NULL);
  cudaMemcpy(C, dC, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);

  printf("TIME: %d %f\n", N, (tv2.tv_sec+tv2.tv_usec*1e-6)-(tv1.tv_sec+tv1.tv_usec*1e-6));
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  return 0;
}
