/* -*- c++ -*- */
#include <stdio.h>
#include <cuda_runtime.h>

#define CHK_DO(o) if(cudaSuccess!=o){printf("%d failed\n",__LINE__);}
#define ALLOCATE(x,y) x=(double*)malloc(sizeof(double)*N*N);CHK_DO(cudaMalloc((void**)&y, sizeof(double)*N*N));
//, cudaMemAttachGlobal))

__global__ void gpukernel(int N, double *a, double *b, double*c)
{
  int i, j, k;
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  for(k=0; k<2; k++){
		c[i*N+j] += a[i*N+j] * b[i*N+j];
	  }
	}
  }
}

int main(int argc, char **argv)
{
  cudaStream_t s1, s2, s3, s4;
  double *A1, *B1, *C1;
  double *A2, *B2, *C2;
  double *A3, *B3, *C3;
  double *A4, *B4, *C4;
  double *dA1, *dB1, *dC1;
  double *dA2, *dB2, *dC2;
  double *dA3, *dB3, *dC3;
  double *dA4, *dB4, *dC4;
  cudaError_t cudaRet;
  int N, M, i, j, k;
  double sum;
  if(argc<2){
	N = 10;
  }else{
	N = atoi(argv[1]);
  }
  printf("N = %d\n", N);
  if(argc<3){
	M = 100;
  }else{
	M = atoi(argv[2]);
  }
  printf("M = %d\n", M);

  ALLOCATE(A1,dA1);
  ALLOCATE(B1,dB1);
  ALLOCATE(C1,dC1);
  ALLOCATE(A2,dA2);
  ALLOCATE(B2,dB2);
  ALLOCATE(C2,dC2);
  ALLOCATE(A3,dA3);
  ALLOCATE(B3,dB3);
  ALLOCATE(C3,dC3);
  ALLOCATE(A4,dA4);
  ALLOCATE(B4,dB4);
  ALLOCATE(C4,dC4);

  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  A1[i*N+j] = 1.0;
	  B1[i*N+j] = 2.0;
	  C1[i*N+j] = 0.0;
	}
  }
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  A2[i*N+j] = 1.0;
	  B2[i*N+j] = 2.0;
	  C2[i*N+j] = 0.0;
	}
  }
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  A3[i*N+j] = 1.0;
	  B3[i*N+j] = 2.0;
	  C3[i*N+j] = 0.0;
	}
  }
  for(i=0; i<N; i++){
	for(j=0; j<N; j++){
	  A4[i*N+j] = 1.0;
	  B4[i*N+j] = 2.0;
	  C4[i*N+j] = 0.0;
	}
  }
  CHK_DO(cudaStreamCreate(&s1));
  CHK_DO(cudaStreamCreate(&s2));
  CHK_DO(cudaStreamCreate(&s3));
  CHK_DO(cudaStreamCreate(&s4));
  CHK_DO(cudaStreamSynchronize(s1));
  CHK_DO(cudaStreamSynchronize(s2));
  CHK_DO(cudaStreamSynchronize(s3));
  CHK_DO(cudaStreamSynchronize(s4));

  #pragma omp parallel sections
  {
  #pragma omp section
  {
  cudaMemcpyAsync(dA1, A1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(dB1, B1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(dC1, C1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  gpukernel<<<M,512,0,s1>>>(N,dA1,dB1,dC1);
  cudaMemcpyAsync(A1, dA1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(B1, dB1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(C1, dC1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(dA1, A1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(dB1, B1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  cudaMemcpyAsync(dC1, C1, sizeof(double)*N*N, cudaMemcpyHostToDevice, s1);
  gpukernel<<<M,512,0,s1>>>(N,dA1,dB1,dC1);
  cudaMemcpyAsync(A1, dA1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(B1, dB1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  cudaMemcpyAsync(C1, dC1, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s1);
  }
#pragma omp section
  {
  cudaMemcpyAsync(dA2, A2, sizeof(double)*N*N, cudaMemcpyHostToDevice, s2);
  cudaMemcpyAsync(dB2, B2, sizeof(double)*N*N, cudaMemcpyHostToDevice, s2);
  cudaMemcpyAsync(dC2, C2, sizeof(double)*N*N, cudaMemcpyHostToDevice, s2);
  gpukernel<<<M,512,0,s2>>>(N,dA2,dB2,dC2);
  cudaMemcpyAsync(A2, dA2, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s2);
  cudaMemcpyAsync(B2, dB2, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s2);
  cudaMemcpyAsync(C2, dC2, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s2);
  }
#pragma omp section
  {
  cudaMemcpyAsync(dA4, A4, sizeof(double)*N*N, cudaMemcpyHostToDevice, s4);
  cudaMemcpyAsync(dB4, B4, sizeof(double)*N*N, cudaMemcpyHostToDevice, s4);
  cudaMemcpyAsync(dC4, C4, sizeof(double)*N*N, cudaMemcpyHostToDevice, s4);
  gpukernel<<<M,512,0,s4>>>(N,dA4,dB4,dC4);
  cudaMemcpyAsync(A4, dA4, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s4);
  cudaMemcpyAsync(B4, dB4, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s4);
  cudaMemcpyAsync(C4, dC4, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s4);
  }
#pragma omp section
  {
  cudaMemcpyAsync(dA3, A3, sizeof(double)*N*N, cudaMemcpyHostToDevice, s3);
  cudaMemcpyAsync(dB3, B3, sizeof(double)*N*N, cudaMemcpyHostToDevice, s3);
  cudaMemcpyAsync(dC3, C3, sizeof(double)*N*N, cudaMemcpyHostToDevice, s3);
  gpukernel<<<M,512,0,s3>>>(N,dA3,dB3,dC3);
  cudaMemcpyAsync(A3, dA3, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s3);
  cudaMemcpyAsync(B3, dB3, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s3);
  cudaMemcpyAsync(C3, dC3, sizeof(double)*N*N, cudaMemcpyDeviceToHost, s3);
  }
}

  CHK_DO(cudaStreamSynchronize(s1));
  CHK_DO(cudaStreamSynchronize(s2));
  CHK_DO(cudaStreamSynchronize(s3));
  CHK_DO(cudaStreamSynchronize(s4));
  CHK_DO(cudaStreamDestroy(s1));
  CHK_DO(cudaStreamDestroy(s2));
  CHK_DO(cudaStreamDestroy(s3));
  CHK_DO(cudaStreamDestroy(s4));

  sum = 0.0;
  for(i=0; i<N; i++)for(j=0; j<N; j++)sum += C1[i*N+j];
  for(i=0; i<N; i++)for(j=0; j<N; j++)sum += C2[i*N+j];
  for(i=0; i<N; i++)for(j=0; j<N; j++)sum += C3[i*N+j];
  for(i=0; i<N; i++)for(j=0; j<N; j++)sum += C4[i*N+j];
  printf("sum = %f\n", sum);

  cudaFree(dA1);
  cudaFree(dB1);
  cudaFree(dC1);
  cudaFree(dA2);
  cudaFree(dB2);
  cudaFree(dC2);
  cudaFree(dA3);
  cudaFree(dB3);
  cudaFree(dC3);
  cudaFree(dA4);
  cudaFree(dB4);
  cudaFree(dC4);
  free(A1);
  free(B1);
  free(C1);
  free(A2);
  free(B2);
  free(C2);
  free(A3);
  free(B3);
  free(C3);
  free(A4);
  free(B4);
  free(C4);
  return 0;
}
