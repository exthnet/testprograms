// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NLOOP 1000
#define NEXEC 10
#define IEXEC NLOOP

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

__global__ void gpukernelW32(double *out, double *mat, double *vec, int iloop)
{
  int i, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y++){
	  tmp = 0.0;
	  tmp += mat[y*32 + threadIdx.x + i*32*32];// * vec[threadIdx.x];
	  if(threadIdx.x==0)out[y] += tmp;
	}
  }
}

__global__ void gpukernelW16r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=2){
	  tmp = 0.0;
	  for(x=0; x<32; x+=16){
		tmp += mat[(y+threadIdx.x/16)*32 + threadIdx.x%16+x + i*32*32];// * vec[threadIdx.x%16+x];
	  }
	  if(threadIdx.x%16==0)out[y+threadIdx.x/16] += tmp;
	}
  }
}

__global__ void gpukernelW16c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=16){
	  for(y=0; y<32; y+=2){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x/16)*32 + threadIdx.x%16+x + i*32*32];// * vec[threadIdx.x%16+x];
		if(threadIdx.x%16==0)out[y+threadIdx.x/16] += tmp;
	  }
	}
  }
}

__global__ void gpukernelW8r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=4){
	  tmp = 0.0;
	  for(x=0; x<32; x+=8){
		tmp += mat[(y+threadIdx.x/8)*32 + threadIdx.x%8+x + i*32*32];// * vec[threadIdx.x%8+x];
	  }
	  if(threadIdx.x%8==0)out[y+threadIdx.x/8] += tmp;
	}
  }
}

__global__ void gpukernelW8c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=8){
	  for(y=0; y<32; y+=4){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x/8)*32 + threadIdx.x%8+x + i*32*32];// * vec[threadIdx.x%8+x];
		if(threadIdx.x%8==0)out[y+threadIdx.x/8] += tmp;
	  }
	}
  }
}

__global__ void gpukernelW4r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=8){
	  tmp = 0.0;
	  for(x=0; x<32; x+=4){
		tmp += mat[(y+threadIdx.x/4)*32 + threadIdx.x%4+x + i*32*32];// * vec[threadIdx.x%4+x];
	  }
	  if(threadIdx.x%4==0)out[y+threadIdx.x/4] += tmp;
	}
  }
}

__global__ void gpukernelW4c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=4){
	  for(y=0; y<32; y+=8){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x/4)*32 + threadIdx.x%4+x + i*32*32];// * vec[threadIdx.x%4+x];
		if(threadIdx.x%4==0)out[y+threadIdx.x/4] += tmp;
	  }
	}
  }
}

__global__ void gpukernelW2r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=16){
	  tmp = 0.0;
	  for(x=0; x<32; x+=2){
		tmp += mat[(y+threadIdx.x/2)*32 + threadIdx.x%2+x + i*32*32];// * vec[threadIdx.x%2+x];
	  }
	  if(threadIdx.x%2==0)out[y+threadIdx.x/2] += tmp;
	}
  }
}

__global__ void gpukernelW2c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=2){
	  for(y=0; y<32; y+=16){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x/2)*32 + threadIdx.x%2+x + i*32*32];// * vec[threadIdx.x%2+x];
		if(threadIdx.x%2==0)out[y+threadIdx.x/2] += tmp;
	  }
	}
  }
}

__global__ void gpukernelW1(double *out, double *mat, double *vec, int iloop)
{
  int i, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	tmp = 0.0;
	for(x=0; x<32; x++){
	  tmp += mat[threadIdx.x*32+x + i*32*32];// * vec[x];
	}
	out[threadIdx.x] += tmp;
  }
}

// ######## ######## ######## ######## ######## ######## ######## ########

__global__ void gpukernelH32(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  tmp += mat[threadIdx.x*32 + x + i*32*32];// * vec[x];
	}
	out[threadIdx.x] += tmp;
  }
}

__global__ void gpukernelH16r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=16){
	  tmp = 0.0;
	  for(x=0; x<32; x+=2){
		tmp += mat[(y+threadIdx.x%16)*32 + threadIdx.x/16+x + i*32*32];// * vec[threadIdx.x/16+x];
	  }
	  if(threadIdx.x/16==0)out[y+threadIdx.x] += tmp;
	}
  }
}

__global__ void gpukernelH16c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=2){
	  for(y=0; y<32; y+=16){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x%16)*32 + threadIdx.x/16+x + i*32*32];// * vec[threadIdx.x/16+x];
		if(threadIdx.x/16==0)out[y+threadIdx.x] += tmp;
	  }
	}
  }
}

__global__ void gpukernelH8r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=8){
	  tmp = 0.0;
	  for(x=0; x<32; x+=4){
		tmp += mat[(y+threadIdx.x%8)*32 + threadIdx.x/8+x + i*32*32];// * vec[threadIdx.x/8+x];
	  }
	  if(threadIdx.x/8==0)out[y+threadIdx.x] += tmp;
	}
  }
}

__global__ void gpukernelH8c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=4){
	  for(y=0; y<32; y+=8){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x%8)*32 + threadIdx.x/8+x + i*32*32];// * vec[threadIdx.x/8+x];
		if(threadIdx.x/8==0)out[y+threadIdx.x] += tmp;
	  }
	}
  }
}

__global__ void gpukernelH4r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=4){
	  tmp = 0.0;
	  for(x=0; x<32; x+=8){
		tmp += mat[(y+threadIdx.x%4)*32 + threadIdx.x/4+x + i*32*32];// * vec[threadIdx.x/4+x];
	  }
	  if(threadIdx.x/4==0)out[y+threadIdx.x] += tmp;
	}
  }
}

__global__ void gpukernelH4c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=8){
	  for(y=0; y<32; y+=4){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x%4)*32 + threadIdx.x/4+x + i*32*32];// * vec[threadIdx.x/4+x];
		if(threadIdx.x/4==0)out[y+threadIdx.x] += tmp;
	  }
	}
  }
}

__global__ void gpukernelH2r(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=2){
	  tmp = 0.0;
	  for(x=0; x<32; x+=16){
		tmp += mat[(y+threadIdx.x%2)*32 + threadIdx.x/2+x + i*32*32];// * vec[threadIdx.x/2+x];
	  }
	  if(threadIdx.x/2==0)out[y+threadIdx.x] += tmp;
	}
  }
}

__global__ void gpukernelH2c(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=16){
	  for(y=0; y<32; y+=2){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x%2)*32 + threadIdx.x/2+x + i*32*32];// * vec[threadIdx.x/2+x];
		if(threadIdx.x/2==0)out[y+threadIdx.x] += tmp;
	  }
	}
  }
}

__global__ void gpukernelH1(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=1){
	  tmp = 0.0;
	  tmp += mat[y*32 + threadIdx.x + i*32*32];// * vec[threadIdx.x];
	  if(threadIdx.x==0)out[y] += tmp;
	}
  }
}

// ######## ######## ######## ######## ######## ######## ######## ########

// general
template<int N>
__global__ void gpukernelWNr(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=32/N){
	  tmp = 0.0;
	  for(x=0; x<32; x+=N){
		tmp += mat[(y+threadIdx.x/N)*32 + threadIdx.x%N+x + i*32*32];// * vec[threadIdx.x%N+x];
	  }
	  if(threadIdx.x%N==0)out[y+threadIdx.x/N] += tmp;
	}
  }
}
template<int N>
__global__ void gpukernelWNc(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=N){
	  for(y=0; y<32; y+=32/N){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x/N)*32 + threadIdx.x%N+x + i*32*32];// * vec[threadIdx.x%N+x];
		if(threadIdx.x%N==0)out[y+threadIdx.x/N] += tmp;
	  }
	}
  }
}

template<int N>
__global__ void gpukernelWNr2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=threadIdx.x/N; y<32; y+=32/N){
	  tmp = 0.0;
	  for(x=threadIdx.x%N; x<32; x+=N){
		tmp += mat[y*32 + x + i*32*32];// * vec[x];
	  }
	  if(threadIdx.x%N==0)out[y] += tmp;
	}
  }
}
template<int N>
__global__ void gpukernelWNc2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=threadIdx.x%N; x<32; x+=N){
	  for(y=threadIdx.x/N; y<32; y+=32/N){
		tmp = 0.0;
		tmp += mat[y*32 + x + i*32*32];// * vec[x];
		if(threadIdx.x%N==0)out[y] += tmp;
	  }
	}
  }
}

// ######## ######## ######## ######## ######## ######## ######## ########

template<int N>
__global__ void gpukernelHNr(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=N){
	  tmp = 0.0;
	  for(x=0; x<32; x+=32/N){
		tmp += mat[(y+threadIdx.x%N)*32 + threadIdx.x/N+x + i*32*32];// * vec[threadIdx.x/N+x];
	  }
	  if(threadIdx.x/N==0)out[y+threadIdx.x%N] += tmp;
	}
  }
}

template<int N>
__global__ void gpukernelHNc(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=0; x<32; x+=32/N){
	  for(y=0; y<32; y+=N){
		tmp = 0.0;
		tmp += mat[(y+threadIdx.x%N)*32 + threadIdx.x/N+x + i*32*32];// * vec[threadIdx.x/N+x];
		if(threadIdx.x/N==0)out[y+threadIdx.x%N] += tmp;
	  }
	}
  }
}

template<int N>
__global__ void gpukernelHNr2(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=threadIdx.x%N; y<32; y+=N){
	  tmp = 0.0;
	  for(x=threadIdx.x/N; x<32; x+=32/N){
		tmp += mat[y*32 + x + i*32*32];// * vec[x];
	  }
	  if(threadIdx.x/N==0)out[y] += tmp;
	}
  }
}

template<int N>
__global__ void gpukernelHNc2(double *out, double *mat, double *vec, int iloop)
{
  int i, y, x;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(x=threadIdx.x/N; x<32; x+=32/N){
	  for(y=threadIdx.x%N; y<32; y+=N){
		tmp = 0.0;
		tmp += mat[y*32 + x + i*32*32];// * vec[x];
		if(threadIdx.x/N==0)out[y] += tmp;
	  }
	}
  }
}

// ######## ######## ######## ######## ######## ######## ######## ########

int main(int argc, char **argv)
{
  int len = NLOOP * 32 * 32;
  int i, x;
  double *out, *mat, *vec;
  double *dout, *dmat, *dvec;
  double d;

  out = (double*)malloc(sizeof(double)*32);
  mat = (double*)malloc(sizeof(double)*len);
  vec = (double*)malloc(sizeof(double)*32);

  cudaMalloc((void**)&dout, sizeof(double)*32);
  cudaMalloc((void**)&dmat, sizeof(double)*len);
  cudaMalloc((void**)&dvec, sizeof(double)*32);

#define BENCH1(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<<<1,32>>>(dout, dmat, dvec, ILOOP);				\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<<<1,32>>>(dout, dmat, dvec, ILOOP);				\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]*(double)(i+1)/32.0; printf("d=%.2f\n",d);\
  }

#define BENCH2(KERNEL,N,NAME,ILOOP)				\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]*(double)(i+1)/32.0; printf("d=%.2f\n",d);\
  }

#define BENCH2r(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNr<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNr<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]*(double)(i+1)/32.0; printf("d=%.2f\n",d);\
  }

#define BENCH2c(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNc<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNc<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]*(double)(i+1)/32.0; printf("d=%.2f\n",d);\
  }

#define BENCH2r2(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<32;i++){\
	  out[i] = 0.0;\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/10.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*32, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNr2<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  gpukernelWNr2<KERNEL><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<32;i++)d+=out[i]*(double)(i+1)/32.0; printf("d=%.2f\n",d);	\
  }


  for(x=0;x<2;x++){
	BENCH1(gpukernelW32,  " W32", 1);
	BENCH1(gpukernelW16r, "W16r", 1);
	BENCH1(gpukernelW16c, "W16c", 1);
	BENCH1(gpukernelW8r,  " W8r", 1);
	BENCH1(gpukernelW8c,  " W8c", 1);
	BENCH1(gpukernelW4r,  " W4r", 1);
	BENCH1(gpukernelW4c,  " W4c", 1);
	BENCH1(gpukernelW2r,  " W2r", 1);
	BENCH1(gpukernelW2c,  " W2c", 1);
	BENCH1(gpukernelW1,   "  W1", 1);
  }
  for(x=0;x<2;x++){
	BENCH1(gpukernelW32,  " W32", IEXEC);
	BENCH1(gpukernelW16r, "W16r", IEXEC);
	BENCH1(gpukernelW16c, "W16c", IEXEC);
	BENCH1(gpukernelW8r,  " W8r", IEXEC);
	BENCH1(gpukernelW8c,  " W8c", IEXEC);
	BENCH1(gpukernelW4r,  " W4r", IEXEC);
	BENCH1(gpukernelW4c,  " W4c", IEXEC);
	BENCH1(gpukernelW2r,  " W2r", IEXEC);
	BENCH1(gpukernelW2c,  " W2c", IEXEC);
	BENCH1(gpukernelW1,   "  W1", IEXEC);
  }

  for(x=0;x<2;x++){
	BENCH2r(32,  "W32r", IEXEC);
	BENCH2c(32,  "W32c", IEXEC);
	BENCH2r(32,  "W32r", IEXEC);
	BENCH2c(32,  "W32c", IEXEC);
	BENCH2r(16,  "W16r", IEXEC);
	BENCH2c(16,  "W16c", IEXEC);
	BENCH2r(8,   " W8r", IEXEC);
	BENCH2c(8,   " W8c", IEXEC);
	BENCH2r(4,   " W4r", IEXEC);
	BENCH2c(4,   " W4c", IEXEC);
	BENCH2r(2,   " W2r", IEXEC);
	BENCH2c(2,   " W2c", IEXEC);
	BENCH2r(1,   " W1r", IEXEC);
	BENCH2c(1,   " W1c", IEXEC);
  }

  for(x=0;x<2;x++){
	BENCH2r2(32,  "W32r2", IEXEC);
	BENCH2r2(32,  "W32r2", IEXEC);
	BENCH2r2(16,  "W16r2", IEXEC);
	BENCH2r2(8,   " W8r2", IEXEC);
	BENCH2r2(4,   " W4r2", IEXEC);
	BENCH2r2(2,   " W2r2", IEXEC);
	BENCH2r2(1,   " W1r2", IEXEC);
  }

  for(x=0;x<2;x++){
	BENCH2(gpukernelWNr, 32, "W32r", IEXEC);
	BENCH2(gpukernelWNc, 32, "W32c", IEXEC);
	BENCH2(gpukernelWNr, 32, "W32r", IEXEC);
	BENCH2(gpukernelWNc, 32, "W32c", IEXEC);
	BENCH2(gpukernelWNr, 16, "W16r", IEXEC);
	BENCH2(gpukernelWNc, 16, "W16c", IEXEC);
	BENCH2(gpukernelWNr,  8, " W8r", IEXEC);
	BENCH2(gpukernelWNc,  8, " W8c", IEXEC);
	BENCH2(gpukernelWNr,  4, " W4r", IEXEC);
	BENCH2(gpukernelWNc,  4, " W4c", IEXEC);
	BENCH2(gpukernelWNr,  2, " W2r", IEXEC);
	BENCH2(gpukernelWNc,  2, " W2c", IEXEC);
	BENCH2(gpukernelWNr,  1, " W1r", IEXEC);
	BENCH2(gpukernelWNc,  1, " W1c", IEXEC);
  }

  // W-major
  printf("W-major 0\n");
  for(x=0;x<2;x++){
	BENCH1(gpukernelW32,  " W32", IEXEC);
	BENCH1(gpukernelW16r, "W16r", IEXEC);
	BENCH1(gpukernelW16c, "W16c", IEXEC);
	BENCH1(gpukernelW8r,  " W8r", IEXEC);
	BENCH1(gpukernelW8c,  " W8c", IEXEC);
	BENCH1(gpukernelW4r,  " W4r", IEXEC);
	BENCH1(gpukernelW4c,  " W4c", IEXEC);
	BENCH1(gpukernelW2r,  " W2r", IEXEC);
	BENCH1(gpukernelW2c,  " W2c", IEXEC);
	BENCH1(gpukernelW1,   "  W1", IEXEC);
  }

  printf("W-major 1\n");
  for(x=0;x<2;x++){
	BENCH2(gpukernelWNr, 32, "W32r", IEXEC);
	BENCH2(gpukernelWNc, 32, "W32c", IEXEC);
	BENCH2(gpukernelWNr, 16, "W16r", IEXEC);
	BENCH2(gpukernelWNc, 16, "W16c", IEXEC);
	BENCH2(gpukernelWNr,  8, " W8r", IEXEC);
	BENCH2(gpukernelWNc,  8, " W8c", IEXEC);
	BENCH2(gpukernelWNr,  4, " W4r", IEXEC);
	BENCH2(gpukernelWNc,  4, " W4c", IEXEC);
	BENCH2(gpukernelWNr,  2, " W2r", IEXEC);
	BENCH2(gpukernelWNc,  2, " W2c", IEXEC);
	BENCH2(gpukernelWNr,  1, " W1r", IEXEC);
	BENCH2(gpukernelWNc,  1, " W1c", IEXEC);
  }

  printf("W-major 2\n");
  for(x=0;x<2;x++){
	BENCH2(gpukernelWNr2, 32, "W32r2", IEXEC);
	BENCH2(gpukernelWNc2, 32, "W32c2", IEXEC);
	BENCH2(gpukernelWNr2, 16, "W16r2", IEXEC);
	BENCH2(gpukernelWNc2, 16, "W16c2", IEXEC);
	BENCH2(gpukernelWNr2,  8, " W8r2", IEXEC);
	BENCH2(gpukernelWNc2,  8, " W8c2", IEXEC);
	BENCH2(gpukernelWNr2,  4, " W4r2", IEXEC);
	BENCH2(gpukernelWNc2,  4, " W4c2", IEXEC);
	BENCH2(gpukernelWNr2,  2, " W2r2", IEXEC);
	BENCH2(gpukernelWNc2,  2, " W2c2", IEXEC);
	BENCH2(gpukernelWNr2,  1, " W1r2", IEXEC);
	BENCH2(gpukernelWNc2,  1, " W1c2", IEXEC);
  }

  // H-major
  printf("H-major 0\n");
  for(x=0;x<2;x++){
	BENCH1(gpukernelH32,  " H32", IEXEC);
	BENCH1(gpukernelH16r, "H16r", IEXEC);
	BENCH1(gpukernelH16c, "H16c", IEXEC);
	BENCH1(gpukernelH8r,  " H8r", IEXEC);
	BENCH1(gpukernelH8c,  " H8c", IEXEC);
	BENCH1(gpukernelH4r,  " H4r", IEXEC);
	BENCH1(gpukernelH4c,  " H4c", IEXEC);
	BENCH1(gpukernelH2r,  " H2r", IEXEC);
	BENCH1(gpukernelH2c,  " H2c", IEXEC);
	BENCH1(gpukernelH1,   "  H1", IEXEC);
  }

  printf("H-major 1\n");
  for(x=0;x<2;x++){
	BENCH2(gpukernelHNr, 32, "H32r", IEXEC);
	BENCH2(gpukernelHNc, 32, "H32c", IEXEC);
	BENCH2(gpukernelHNr, 16, "H16r", IEXEC);
	BENCH2(gpukernelHNc, 16, "H16c", IEXEC);
	BENCH2(gpukernelHNr,  8, " H8r", IEXEC);
	BENCH2(gpukernelHNc,  8, " H8c", IEXEC);
	BENCH2(gpukernelHNr,  4, " H4r", IEXEC);
	BENCH2(gpukernelHNc,  4, " H4c", IEXEC);
	BENCH2(gpukernelHNr,  2, " H2r", IEXEC);
	BENCH2(gpukernelHNc,  2, " H2c", IEXEC);
	BENCH2(gpukernelHNr,  1, " H1r", IEXEC);
	BENCH2(gpukernelHNc,  1, " H1c", IEXEC);
  }

  printf("H-major 2\n");
  for(x=0;x<2;x++){
	BENCH2(gpukernelHNr2, 32, "H32r2", IEXEC);
	BENCH2(gpukernelHNc2, 32, "H32c2", IEXEC);
	BENCH2(gpukernelHNr2, 16, "H16r2", IEXEC);
	BENCH2(gpukernelHNc2, 16, "H16c2", IEXEC);
	BENCH2(gpukernelHNr2,  8, " H8r2", IEXEC);
	BENCH2(gpukernelHNc2,  8, " H8c2", IEXEC);
	BENCH2(gpukernelHNr2,  4, " H4r2", IEXEC);
	BENCH2(gpukernelHNc2,  4, " H4c2", IEXEC);
	BENCH2(gpukernelHNr2,  2, " H2r2", IEXEC);
	BENCH2(gpukernelHNc2,  2, " H2c2", IEXEC);
	BENCH2(gpukernelHNr2,  1, " H1r2", IEXEC);
	BENCH2(gpukernelHNc2,  1, " H1c2", IEXEC);
  }

  cudaFree(dout); cudaFree(dmat); cudaFree(dvec);
  free(out); free(mat); free(vec);
  return 0;
}
