/* -*- c++ -*- */
#include <stdio.h>
#include <cuda_runtime.h>

#define CHK_DO(o) if(cudaSuccess!=o){printf("%d failed\n",__LINE__);}
#define ALLOCATE(x) CHK_DO(cudaMallocManaged((void**)&x, sizeof(double)*N*N))
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

  ALLOCATE(A1);
  ALLOCATE(B1);
  ALLOCATE(C1);
  ALLOCATE(A2);
  ALLOCATE(B2);
  ALLOCATE(C2);
  ALLOCATE(A3);
  ALLOCATE(B3);
  ALLOCATE(C3);
  ALLOCATE(A4);
  ALLOCATE(B4);
  ALLOCATE(C4);

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

  gpukernel<<<M,1024,0,s1>>>(N,A1,B1,C1);
  gpukernel<<<M,1024,0,s2>>>(N,A2,B2,C2);
  gpukernel<<<M,1024,0,s4>>>(N,A4,B4,C4);
  gpukernel<<<M,1024,0,s3>>>(N,A3,B3,C3);
  gpukernel<<<M,1024,0,s1>>>(N,A1,B1,C1);

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

  cudaFree(A1);
  cudaFree(B1);
  cudaFree(C1);
  cudaFree(A2);
  cudaFree(B2);
  cudaFree(C2);
  cudaFree(A3);
  cudaFree(B3);
  cudaFree(C3);
  cudaFree(A4);
  cudaFree(B4);
  cudaFree(C4);
  return 0;
}
