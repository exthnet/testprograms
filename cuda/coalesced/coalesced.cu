// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

/*
64x64行列を32threadsで読み込むテスト
 */

__global__ void gpukernel1(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=0; y<64; y++){
	for(x=tid; x<64; x+=32){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

__global__ void gpukernel2(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid/16; y<64; y+=2){
	for(x=tid%16; x<64; x+=16){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

__global__ void gpukernel4(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid/8; y<64; y+=4){
	for(x=tid%8; x<64; x+=8){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

__global__ void gpukernel8(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid/4; y<64; y+=8){
	for(x=tid%4; x<64; x+=4){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

__global__ void gpukernel16(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid/2; y<64; y+=16){
	for(x=tid%2; x<64; x+=2){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

__global__ void gpukernel32(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid; y<64; y+=32){
	for(x=0; x<64; x+=1){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

template <int X>
__global__ void gpukernelX(double *mat, double *vec)
{
  int tid = threadIdx.x;
  double tmp;
  int x, y;
  tmp = 0.0;
  for(y=tid/X; y<64; y+=(32/X)){
	for(x=tid%X; x<64; x+=X){
	  tmp += mat[y*64+x];
	}
  }
  vec[tid] = tmp;
}

void gpukernel_driver(int x, double *dmat, double *dvec)
{
  switch(x){
  case 1:  gpukernelX< 1><<<1,32>>>(dmat, dvec); break;
  case 2:  gpukernelX< 2><<<1,32>>>(dmat, dvec); break;
  case 4:  gpukernelX< 4><<<1,32>>>(dmat, dvec); break;
  case 8:  gpukernelX< 8><<<1,32>>>(dmat, dvec); break;
  case 16: gpukernelX<16><<<1,32>>>(dmat, dvec); break;
  case 32: gpukernelX<32><<<1,32>>>(dmat, dvec); break;
  }
}

int main(int argc, char **argv)
{
  int i, N;
  double *mat, *vec;
  double *dmat, *dvec;
  double d;

  N = 64;
  mat = (double*)malloc(sizeof(double)*N*N);
  vec = (double*)malloc(sizeof(double)*N);

  for(i=0;i<N*N;i++){
	mat[i] = (double)(i+1);
  }
  for(i=0;i<N;i++){
	vec[i] = 0.0f;
  }

  cudaMalloc((void**)&dmat, sizeof(double)*N*N);
  cudaMalloc((void**)&dvec, sizeof(double)*N);

  cudaMemcpy(dmat, mat, sizeof(double)*N*N, cudaMemcpyHostToDevice);

  // 1
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel1<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 1, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 2
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel2<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 2, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 4
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel4<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 4, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 8
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel8<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 8, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 16
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel16<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 16, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 32
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel32<<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", 32, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);




  int X;

  // 1
  X=32;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<32><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 2
  X=16;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<16><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 4
  X=8;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<8><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 8
  X=4;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<4><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 16
  X=2;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<2><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // 32
  X=1;
  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernelX<1><<<1,32>>>(dmat, dvec);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;
  printf("%2d: time %f, ", X, d);
  cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
  d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);

  // driver
  for(X=32;X>=1;X>>=1){
	d=omp_get_wtime();
	for(i=0;i<100;i++){
	  cudaDeviceSynchronize();
	  gpukernel_driver(X,dmat, dvec);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;
	printf("%2d: time %f, ", X, d);
	cudaMemcpy(vec, dvec, sizeof(double)*N, cudaMemcpyDeviceToHost);
	d=0; for(i=0;i<32;i++)d+=vec[i]; printf("d=%f\n",d);
  }

  for(i=0;i<N;i++){
	printf(" %f", vec[i]);
  }
  printf("\n");

  cudaFree(dmat); cudaFree(dvec);
  free(mat); free(vec);
  return 0;
}
