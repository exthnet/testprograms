// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NEXEC 1000

__global__ void gpukernelW32(double *out, double *mat, double *vec)
{
  int y;
  double tmp = 0.0;
  for(y=0; y<32; y++){
	tmp = mat[y*32+threadIdx.x] * vec[threadIdx.x];
	for(int offset=16; offset>0; offset/=2){
	  tmp += __shfl_down_sync
		(0xffffffff, tmp, offset, 32);
	}
	if(threadIdx.x==0)out[threadIdx.x] += tmp;
  }
}

__global__ void gpukernelW1(double *out, double *mat, double *vec)
{
  int x;
  double tmp = 0.0;
  for(x=0; x<32; x++){
	tmp += mat[threadIdx.x*32+x] * vec[x];
  }
  out[threadIdx.x] += tmp;
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int N = 1000;
  int len = 32 * 32;
  int i, x;
  double *out, *mat, *vec;
  double *dout, *dmat, *dvec;
  double d;

  if(argc>1)N=atoi(argv[1]); printf("N=%d\n", N);
  out = (double*)malloc(sizeof(double)*32);
  mat = (double*)malloc(sizeof(double)*len);
  vec = (double*)malloc(sizeof(double)*32);

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&dmat, sizeof(double)*len);
  cudaMalloc((void**)&dvec, sizeof(double)*32);

  {
	x = 32;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	  vec[i] = sin((double)i/10.0);
	}
	for(i=0;i<len;i++){
	  mat[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW32<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW32<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);

  }

  {
	x = 1;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	  vec[i] = sin((double)i/10.0);
	}
	for(i=0;i<len;i++){
	  mat[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW1<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW1<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }

  {
	x = 32;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	  vec[i] = sin((double)i/10.0);
	}
	for(i=0;i<len;i++){
	  mat[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW32<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW32<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }

  {
	x = 1;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	  vec[i] = sin((double)i/10.0);
	}
	for(i=0;i<len;i++){
	  mat[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW1<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernelW1<<<1,32>>>(dout, dmat, dvec);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }


  cudaFree(dout); cudaFree(dmat); cudaFree(dvec);
  free(out); free(mat); free(vec);
  return 0;
}
