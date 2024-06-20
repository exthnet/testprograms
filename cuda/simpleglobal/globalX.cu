// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void gpukernel1(double *out, double *in, int N, int A, int B)
{
  int id;
  int i;
  int begin = (threadIdx.x/B)*(N/A)+(threadIdx.x%B);
  int end = ((threadIdx.x/B)+1)*(N/A);
  int step = B;
  double tmp = 0.0;
  for(id=0;id<1000;id++){
	for(i=begin;i<end;i+=step){
	  tmp += in[id*32+i];
	}
  }
  out[threadIdx.x] += tmp;
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int N = 1000;
  int len = N * 32;
  int i;
  double *out, *in;
  double *dout, *din;
  double d;

  if(argc>1)N=atoi(argv[1]); printf("N=%d\n", N);
  out = (double*)malloc(sizeof(double)*32);
  in = (double*)malloc(sizeof(double)*len);

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&din, sizeof(double)*len);

  int A, B, x;
  for(x=1;x<=32;x*=2){
	A = x; B = 32/x;
	for(i=0;i<32;i++){
	  out[i] = 0.0;
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<10;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, 32, A, B);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<10;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, 32, A, B);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }

  for(x=1;x<=32;x*=2){
	A = x; B = 32/x;
	for(i=0;i<32;i++){
	  out[i] = 0.0;
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<10;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, 32, A, B);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<10;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, 32, A, B);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }


  cudaFree(dout);  cudaFree(din);
  free(out); free(in);
  return 0;
}
