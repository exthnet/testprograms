// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NEXEC 1000

__global__ void gpukernel1(double *out, double *in, int iloop)
{
  int id;
  int y;
  double tmp = 0.0;
  for(id=0;id<NEXEC;id++){
	for(y=0; y<32; y++){
	  tmp += in[y*32000+threadIdx.x+id*32];
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernel2(double *out, double *in, int iloop)
{
  int id;
  int x;
  double tmp = 0.0;
  for(id=0;id<NEXEC;id++){
	for(x=0; x<32; x++){
	  tmp += in[threadIdx.x*32000+x+id*32];
	}
  }
  out[threadIdx.x] += tmp;
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int N = 1000;
  int len = N * 32 * 32;
  int i, x;
  double *out, *in;
  double *dout, *din;
  double d;

  if(argc>1)N=atoi(argv[1]); printf("N=%d\n", N);
  out = (double*)malloc(sizeof(double)*32);
  in = (double*)malloc(sizeof(double)*len);

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&din, sizeof(double)*len);

  {
	x = 1;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }

  {
	x = 2;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel2<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel2<<<1,32>>>(dout, din, NEXEC);
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
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel1<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	d=omp_get_wtime()-d;

	printf("%2d: time %f msec(/total), ", x, d);
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);
  }

  {
	x = 2;

	for(i=0;i<32;i++){
	  out[i] = 0.0;
	}
	for(i=0;i<len;i++){
	  in[i] = (double)(i+1);
	}

	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);

	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel2<<<1,32>>>(dout, din, NEXEC);
	  cudaDeviceSynchronize();
	}
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);

	d=omp_get_wtime();
	for(i=0;i<NEXEC;i++){
	  cudaDeviceSynchronize();
	  gpukernel2<<<1,32>>>(dout, din, NEXEC);
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
