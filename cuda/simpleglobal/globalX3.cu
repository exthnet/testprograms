// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NLOOP 1000
#define NEXEC 1000
#define IEXEC 1000

/*
W=32
  0  1  2  3 ... 31
W=16
  0  1  2  3 ... 15
 16 17 18 19 ... 31
W=8
  0  1  2  3 ...  7
  8  9 10 11 ... 15
 16 17 18 19 ... 23
 24 25 26 27 ... 31
W=4
  0  1  2  3
  4  5  6  7
  ...
 28 29 30 31
W=2
  0  1
  2  3
 ...
 30 31
W=1
  0
  1
  2
  3
 ...
 31
*/

__global__ void gpukernelW32(double *out, double *in, int iloop)
{
  int i, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y++){
	  tmp += in[y*32+threadIdx.x + i*32*32];
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW16a(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=2){
	  for(x=0; x<32; x+=16){
		tmp += in[(y+threadIdx.x/16)*32 + threadIdx.x%16+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW16b(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=16){
	  for(y=0; y<32; y+=2){
		tmp += in[(y+threadIdx.x/16)*32 + threadIdx.x%16+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW8a(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=4){
	  for(x=0; x<32; x+=8){
		tmp += in[(y+threadIdx.x/8)*32 + threadIdx.x%8+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW8b(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=8){
	  for(y=0; y<32; y+=4){
		tmp += in[(y+threadIdx.x/8)*32 + threadIdx.x%8+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW4a(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=8){
	  for(x=0; x<32; x+=4){
		tmp += in[(y+threadIdx.x/4)*32 + threadIdx.x%4+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW4b(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=4){
	  for(y=0; y<32; y+=8){
		tmp += in[(y+threadIdx.x/4)*32 + threadIdx.x%4+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW2a(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=16){
	  for(x=0; x<32; x+=2){
		tmp += in[(y+threadIdx.x/2)*32 + threadIdx.x%2+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW2b(double *out, double *in, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=2){
	  for(y=0; y<32; y+=16){
		tmp += in[(y+threadIdx.x/2)*32 + threadIdx.x%2+x + i*32*32];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

__global__ void gpukernelW1(double *out, double *in, int iloop)
{
  int i, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x++){
	  tmp += in[threadIdx.x*32+x + i*32*32];
	}
  }
  out[threadIdx.x] += tmp;
}

// general
template<int N>
__global__ void gpukernelWNa(double *out, double *in, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=32/N){
	  for(x=0; x<32; x+=N){
		tmp += in[(y+threadIdx.x/N)*32 + threadIdx.x%N+x + i*32*32];// * vec[x];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}
template<int N>
__global__ void gpukernelWNb(double *out, double *in, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=N){
	  for(y=0; y<32; y+=32/N){
		tmp += in[(y+threadIdx.x/N)*32 + threadIdx.x%N+x + i*32*32];// * vec[x];
	  }
	}
  }
  out[threadIdx.x] += tmp;
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int len = NLOOP * 32 * 32;
  int i, x;
  double *out, *in;
  double *dout, *din;
  double d;

  out = (double*)malloc(sizeof(double)*32);
  in = (double*)malloc(sizeof(double)*len);

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&din, sizeof(double)*len);

#define BENCH1(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<len;i++){\
	  in[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<<<1,32>>>(dout, din, ILOOP);				\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<<<1,32>>>(dout, din, ILOOP);				\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);\
  }

#define BENCH2(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<len;i++){\
	  in[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWN<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWN<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);\
  }
#define BENCH2a(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<len;i++){\
	  in[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNa<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNa<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);\
  }
#define BENCH2b(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<len;i++){\
	  in[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNb<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNb<KERNEL><<<1,32>>>(dout, din, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]; printf("d=%.2f\n",d);\
  }

  for(x=0;x<2;x++){
	BENCH1(gpukernelW32,  " W32", 1);
	BENCH1(gpukernelW16a, "W16r", 1);
	BENCH1(gpukernelW16b, "W16c", 1);
	BENCH1(gpukernelW8a,  " W8r", 1);
	BENCH1(gpukernelW8b,  " W8c", 1);
	BENCH1(gpukernelW4a,  " W4r", 1);
	BENCH1(gpukernelW4b,  " W4c", 1);
	BENCH1(gpukernelW2a,  " W2r", 1);
	BENCH1(gpukernelW2b,  " W2c", 1);
	BENCH1(gpukernelW1,   "  W1", 1);
  }
  for(x=0;x<2;x++){
	BENCH1(gpukernelW32,  " W32", IEXEC);
	BENCH1(gpukernelW16a, "W16r", IEXEC);
	BENCH1(gpukernelW16b, "W16c", IEXEC);
	BENCH1(gpukernelW8a,  " W8r", IEXEC);
	BENCH1(gpukernelW8b,  " W8c", IEXEC);
	BENCH1(gpukernelW4a,  " W4r", IEXEC);
	BENCH1(gpukernelW4b,  " W4c", IEXEC);
	BENCH1(gpukernelW2a,  " W2r", IEXEC);
	BENCH1(gpukernelW2b,  " W2c", IEXEC);
	BENCH1(gpukernelW1,   "  W1", IEXEC);
  }

  for(x=0;x<2;x++){
	BENCH2a(32,  "W32r", IEXEC);
	BENCH2b(32,  "W32c", IEXEC);
	BENCH2a(32,  "W32r", IEXEC);
	BENCH2b(32,  "W32c", IEXEC);
	BENCH2a(16,  "W16r", IEXEC);
	BENCH2b(16,  "W16c", IEXEC);
	BENCH2a(8,   " W8r", IEXEC);
	BENCH2b(8,   " W8c", IEXEC);
	BENCH2a(4,   " W4r", IEXEC);
	BENCH2b(4,   " W4c", IEXEC);
	BENCH2a(2,   " W2r", IEXEC);
	BENCH2b(2,   " W2c", IEXEC);
	BENCH2a(1,   " W1r", IEXEC);
	BENCH2b(1,   " W1c", IEXEC);
  }

  cudaFree(dout);  cudaFree(din);
  free(out); free(in);
  return 0;
}
