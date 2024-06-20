// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

__global__ void gpukernel1(double *out, double *in, int N)
{
  int i;
  int begin = (threadIdx.x/2)*(N/16)+(threadIdx.x%2);
  int end = ((threadIdx.x/2)+1)*(N/16);
  int step = 2;
  double tmp = 0.0;
  for(i=begin;i<end;i+=step){
	tmp += in[i];
  }
  out[threadIdx.x] = tmp;
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int N = 100;
  int len = N * 32;
  int i;
  double *out, *in;
  double *dout, *din;
  double d;

  if(argc>1)N=atoi(argv[1]); printf("N=%d, ", N);
  out = (double*)malloc(sizeof(double)*32);
  in = (double*)malloc(sizeof(double)*len);

  for(i=0;i<32;i++){
	out[i] = 0.0;
  }
  for(i=0;i<len;i++){
	in[i] = (double)(i+1);
  }

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&din, sizeof(double)*len);

  cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
  cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  gpukernel1<<<1,32>>>(dout, din, N);
  cudaDeviceSynchronize();
  cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

  d=omp_get_wtime();
  for(i=0;i<100;i++){
	cudaDeviceSynchronize();
	gpukernel1<<<1,32>>>(dout, din, N);
	cudaDeviceSynchronize();
  }
  d=omp_get_wtime()-d;

  printf("%2d: time %f msec(/100times), ", 16, d);
  d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);

  cudaFree(dout);  cudaFree(din);
  free(out); free(in);
  return 0;
}

