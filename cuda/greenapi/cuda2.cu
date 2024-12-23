// -*- c++ -*-
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>

#define frand() (double)(rand()%100)/10.0

__global__ void gpukernel
(int N, double *A, double *B, double *C){
  int i, j, k, ibegin, istep, jbegin, jstep;
  ibegin = blockIdx.y*blockDim.y + threadIdx.y;
  istep = blockDim.y * gridDim.y;
  jbegin = blockIdx.x*blockDim.x + threadIdx.x;
  jstep = blockDim.x * gridDim.x;
  for(i=ibegin; i<N; i+=istep){
	for(j=jbegin; j<N; j+=jstep){
	  for(k=0; k<N; k++){
		C[i*N+j] += A[i*N+k] * B[k*N+j];
	  }
	}
  }
}

double *A, *B, *C;
double *d_A, *d_B, *d_C; // device memory
int N = 100;
int blk = 10;
int main(int argc, char **argv){
  if(argc>1){
	N = atoi(argv[1]);
  }
  if(argc>2){
	blk = atoi(argv[2]);
  }
  printf("N = %d, blk = %d\n", N, blk);

  CUresult retDev;
  CUdevice dev, dev2;
  CUdevResource devRsc, devRsc2, devRsc3;
  unsigned int nGroups, nMin=16;
  CUdevResourceDesc devRscDesc, devRscDesc2;
  CUgreenCtx phCtx;
  retDev = cuInit(0);
  if(retDev!=CUDA_SUCCESS){printf("cuInit failed\n"); return -1;}
  retDev = cuDeviceGet(&dev, 0);
  if(retDev!=CUDA_SUCCESS){printf("cuDeviceGet failed\n"); return -1;}
  retDev = cuDeviceGetDevResource(dev, &devRsc, CU_DEV_RESOURCE_TYPE_SM);
  if(retDev!=CUDA_SUCCESS){printf("cuDeviceGetResource failed\n"); return -1;}
  retDev = cuDevSmResourceSplitByCount(&devRsc2, &nGroups, &devRsc, &devRsc3, 0, nMin);
  //retDev = cuDevSmResourceSplitByCount(NULL, &nGroups, &devRsc, NULL, 0, nMin);
  if(retDev!=CUDA_SUCCESS){printf("cuDevSmResourceSplitByCount failed\n");
	switch(retDev){
	case CUDA_SUCCESS:
	  printf("CUDA_SUCCESS\n"); break;
	case CUDA_ERROR_DEINITIALIZED:
	  printf("CUDA_ERROR_DEINITIALIZED\n"); break;
	case CUDA_ERROR_NOT_INITIALIZED:
	  printf("CUDA_ERROR_NOT_INITIALIZED\n"); break;
	case CUDA_ERROR_INVALID_DEVICE:
	  printf("CUDA_ERROR_INVALID_DEVICE\n"); break;
	case CUDA_ERROR_INVALID_VALUE:
	  printf("CUDA_ERROR_INVALID_VALUE\n"); break;
	case CUDA_ERROR_INVALID_RESOURCE_TYPE:
	  printf("CUDA_ERROR_INVALID_RESOURCE_TYPE\n"); break;
	case CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION:
	  printf("CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION\n"); break;
	}
  }
  retDev = cuDevResourceGenerateDesc(&devRscDesc, &devRsc2, 1);
  if(retDev!=CUDA_SUCCESS){printf("cuDeviceGetResource failed\n");}
  retDev = cuGreenCtxCreate(&phCtx, devRscDesc, dev2, CU_GREEN_CTX_DEFAULT_STREAM);
  if(retDev!=CUDA_SUCCESS){printf("cuGreenCtxCreate failed\n");}

  int x, y;
  A = (double*)malloc(sizeof(double)*N*N);
  B = (double*)malloc(sizeof(double)*N*N);
  C = (double*)malloc(sizeof(double)*N*N);
  for(y=0; y<N; y++){
    for(x=0; x<N; x++){
      A[y*N+x] = frand();
      B[y*N+x] = frand();
      C[y*N+x] = 0.0;
    }
  }
  cudaMalloc((void**)&d_A, sizeof(double)*N*N);
  cudaMalloc((void**)&d_B, sizeof(double)*N*N);
  cudaMalloc((void**)&d_C, sizeof(double)*N*N);
  cudaMemcpy(d_A, A, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(double)*N*N, cudaMemcpyHostToDevice);

  double d1, d2;
  cudaDeviceSynchronize();
  d1 = omp_get_wtime();
  gpukernel<<<dim3(blk,blk),dim3(8,8)>>>(N, d_A, d_B, d_C);
  cudaDeviceSynchronize();
  d2 = omp_get_wtime();
  printf("gpukernel %f sec\n", d2-d1);

  cudaMemcpy(C, d_C, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  // check result
  {
	int i;
	for(i=0; i<10; i++)printf(" %f", A[i]); printf("\n");
	for(i=0; i<10; i++)printf(" %f", B[i]); printf("\n");
	for(i=0; i<10; i++)printf(" %f", C[i]); printf("\n");
	double sum=0.0;
	for(y=0; y<N; y++){
	  for(x=0; x<N; x++){
		sum += C[y*N+x];
	  }
	}
	printf("sum %f\n", sum);
  }

  /*
  // A
  printf("A\n");
  for(y=0; y<N; y++){
    for(x=0; x<N; x++){
      printf(" %.2f", A[y*N+x]);
    }
	printf("\n");
  }
  // B
  printf("B\n");
  for(y=0; y<N; y++){
    for(x=0; x<N; x++){
      printf(" %.2f", B[y*N+x]);
    }
	printf("\n");
  }
  // C
  printf("C\n");
  for(y=0; y<N; y++){
    for(x=0; x<N; x++){
      printf(" %.2f", C[y*N+x]);
    }
	printf("\n");
  }
  */
  free(A); free(B); free(C);
  return 0;
}
