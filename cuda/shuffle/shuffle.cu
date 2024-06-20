// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>

__global__ void gpukernel(int N, double *C, double *A, double *B)
{
  int tid = threadIdx.x;
  double tmp = A[tid];
  //tmp = __shfl_down_sync(0xFFFFFFFF, tmp, 1, warpSize);
  //tmp = __shfl_down_sync(0x00000000, tmp, 1, warpSize);
  tmp = __shfl_up_sync(0xFFFFFFFF, tmp, 8, 16);
  /*
  if(tid<16){
	tmp = __shfl_down_sync(0x0000ffff, tmp, 1, warpSize);
  }else{
	tmp = __shfl_down_sync(0xffff0000, tmp, 2, warpSize);
  }
  */
  /*
  if(tid<16){
	tmp = __shfl_down_sync(0xffffffff, tmp, 1, warpSize);
  }else{
	tmp = __shfl_down_sync(0xffffffff, tmp, 2, warpSize);
  }
  */
  /*
  if(tid>=16){
	tmp = __shfl_down_sync(0xffffffff, tmp, 2, warpSize);
  }else{
	tmp = __shfl_down_sync(0xffffffff, tmp, 1, warpSize);
  }
  */
  /*
  if(tid>=16){
	tmp = __shfl_down_sync(0xffffffff, tmp, 2, warpSize);
  }
  if(tid<16){
	tmp = __shfl_down_sync(0xffffffff, tmp, 1, warpSize);
  }
  */
  /*
  if(tid<16){
	tmp = __shfl_down_sync(0x0000ffff, tmp, 1, warpSize);
  }
  if(tid>=16){
	tmp = __shfl_down_sync(0xffff0000, tmp, 2, warpSize);
  }
  */
  /*
  if(tid<16){
	tmp = __shfl_down_sync(0xffff0000, tmp, 1, warpSize);
  }
  if(tid>=16){
	tmp = __shfl_down_sync(0x0000ffff, tmp, 2, warpSize);
  }
  */
  C[tid] = tmp;
}


int main(int argc, char **argv)
{
  int i, N, x;
  double *A, *B, *C;
  double *dA, *dB, *dC;

  x = 32;
  N = 32;
  A = (double*)malloc(sizeof(double)*N);
  B = (double*)malloc(sizeof(double)*N);
  C = (double*)malloc(sizeof(double)*N);

  for(i=0;i<N;i++){
	C[i] = 0.0f;
	B[i] = 2.0f;
	A[i] = (double)(i+1);///(double)(N);
  }

  cudaMalloc((void**)&dA, sizeof(double)*N);
  cudaMalloc((void**)&dB, sizeof(double)*N);
  cudaMalloc((void**)&dC, sizeof(double)*N);

  printf("A\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.0f\n", A[i]);
	}else{
	  printf(" %2.0f", A[i]);
	}
  }
  printf("\n");
  printf("B\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.0f\n", B[i]);
	}else{
	  printf(" %2.0f", B[i]);
	}
  }
  printf("\n");
  printf("C (before)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.0f\n", C[i]);
	}else{
	  printf(" %2.0f", C[i]);
	}
  }
  printf("\n");

  cudaMemcpy(dA, A, sizeof(double)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(double)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(double)*N, cudaMemcpyHostToDevice);

  dim3 grids;
  dim3 blocks;
  grids = dim3(1, 1, 1);
  blocks = dim3(32, 1 ,1);
  cudaDeviceSynchronize();
  gpukernel<<<grids,blocks>>>(N, dC, dA, dB);
  cudaDeviceSynchronize();
  cudaMemcpy(C, dC, sizeof(double)*N, cudaMemcpyDeviceToHost);

  printf("C (after)\n");
  for(i=0; i<N; i++){
	if(i%x==x-1){
	  printf(" %2.0f\n", C[i]);
	}else{
	  printf(" %2.0f", C[i]);
	}
  }
  printf("\n");

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  return 0;
}
