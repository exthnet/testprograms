// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NLOOP 1000
#define NEXEC 10
#define IEXEC NLOOP

//#define KERNEL32

// ######## ######## ######## ######## ######## ######## ######## ########
// general WNr, WNc, WNr2, WNc2

template<int N>
__global__ void gpukernelWNr(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i++){
	for(y=0; y<32; y+=32/N){
	  tmp = 0.0;
	  for(x=0; x<32; x+=N){
		tmp += mat[(y+threadIdx.x/N)*32 + threadIdx.x%N+x + i*32*32] * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y+threadIdx.x/N + i*32] += tmp;
	}
  }
}

template<int N, int M>
__global__ void gpukernelAMWNr(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i+=M){
	for(y=0; y<32; y+=32/N){
	  tmp = 0.0;
	  for(x=0; x<32; x+=N){
		tmp += mat[(y+(threadIdx.x%32)/N)*32 + threadIdx.x%N+x + (i+threadIdx.x/32)*32*32];// * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y+(threadIdx.x%32)/N + i*32] += tmp;
	}
  }
}
template<int N, int M>
__global__ void gpukernelAMWNr2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/32;i<iloop;i+=M){
	for(y=(threadIdx.x%32)/N; y<32; y+=32/N){
	  tmp = 0.0;
	  for(x=threadIdx.x%N; x<32; x+=N){
		tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y + i*32] += tmp;
	}
  }
}
template<int N, int M>
__global__ void gpukernelAMWNr3(double *out, double *mat, double *vec, int iloop, int w, int h)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/32;i<iloop;i+=M){
	for(y=(threadIdx.x%32)/N; y<h; y+=32/N){
	  tmp = 0.0;
	  for(x=threadIdx.x%N; x<w; x+=N){
		tmp += mat[y*w + x + i*w*h];// * vec[threadIdx.x%N+x];
	  }
	  /*
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y + i*h] += tmp;
	  */
	  atomicAdd(&out[y + i*h], tmp);
	}
  }
}
template<int N, int M>
__global__ void gpukernelAMWNc2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/32;i<iloop;i+=M){
	for(x=threadIdx.x%N; x<32; x+=N){
	  for(y=(threadIdx.x%32)/N; y<32; y+=32/N){
		tmp = 0.0;
		tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*32] += tmp;
	  }
	}
  }
}
template<int N, int M>
__global__ void gpukernelAMWNc3(double *out, double *mat, double *vec, int iloop, int w, int h)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/32;i<iloop;i+=M){
	for(x=threadIdx.x%N; x<w; x+=N){
	  for(y=(threadIdx.x%32)/N; y<h; y+=32/N){
		tmp = 0.0;
		tmp += mat[y*w + x + i*w*h];// * vec[threadIdx.x%N+x];
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*h] += tmp;
	  }
	}
  }
}

template<int N, int M>
__global__ void gpukernelBMWNr(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i+=M*32/N){
	for(y=0; y<32; y+=1){
	  tmp = 0.0;
	  for(x=0; x<32; x+=N){
		tmp += mat[y*32 + threadIdx.x%N+x + (i+threadIdx.x/N)*32*32];// * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y + i*32] += tmp;
	}
  }
}
template<int N, int M>
__global__ void gpukernelBMWNr2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/N;i<iloop;i+=M*32/N){
	for(y=0; y<32; y+=1){
	  tmp = 0.0;
	  for(x=threadIdx.x%N; x<32; x+=N){
		tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y + i*32] += tmp;
	}
  }
}
template<int N, int M>
__global__ void gpukernelBMWNr3(double *out, double *mat, double *vec, int iloop, int w, int h)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/N;i<iloop;i+=M*32/N){
	for(y=0; y<h; y+=1){
	  tmp = 0.0;
	  for(x=threadIdx.x%N; x<w; x+=N){
		tmp += mat[y*w + x + i*w*h];// * vec[threadIdx.x%N+x];
	  }
	  /*
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y + i*h] += tmp;
	  */
	  atomicAdd(&out[y + i*h], tmp);
	}
  }
}
template<int N, int M>
__global__ void gpukernelBMWNc2(double *out, double *mat, double *vec, int iloop)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/N;i<iloop;i+=M*32/N){
	for(x=threadIdx.x%N; x<32; x+=N){
	  for(y=0; y<32; y+=1){
		tmp = 0.0;
		tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*32] += tmp;
	  }
	}
  }
}
template<int N, int M>
__global__ void gpukernelBMWNc3(double *out, double *mat, double *vec, int iloop, int w, int h)
{
  int i, x, y;
  double tmp = 0.0;
  for(i=threadIdx.x/N;i<iloop;i+=M*32/N){
	for(x=threadIdx.x%N; x<w; x+=N){
	  for(y=0; y<h; y+=1){
		tmp = 0.0;
		tmp += mat[y*w + x + i*w*h];// * vec[threadIdx.x%N+x];
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*h] += tmp;
	  }
	}
  }
}

template<int N, int M, int L>
__global__ void gpukernelCLMWNr2(double *out, double *mat, double *vec, int iloop)
{
	// complete
	int i, x, y;
	double tmp = 0.0;
	for(i=threadIdx.x/(N*L);i<iloop;i+=(32*M)/(N*L)){
	  for(y=(threadIdx.x%(N*L))/N; y<32; y+=L){
		tmp = 0.0;
		for(x=threadIdx.x%N; x<32; x+=N){
		  tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*32] += tmp;
	  }
	}
}
template<int N, int M, int L>
__global__ void gpukernelCLMWNr3(double *out, double *mat, double *vec, int iloop, int w, int h)
{
	// complete
	int i, x, y;
	double tmp = 0.0;
	for(i=threadIdx.x/(N*L);i<iloop;i+=(32*M)/(N*L)){
	  for(y=(threadIdx.x%(N*L))/N; y<h; y+=L){
		tmp = 0.0;
		for(x=threadIdx.x%N; x<w; x+=N){
		  tmp += mat[y*w + x + i*w*h];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*h] += tmp;
	  }
	}
}

template<int N, int M, int L>
__global__ void gpukernelCLMWNr(double *out, double *mat, double *vec, int iloop)
{
#if 0
  int i, x, y;
  double tmp = 0.0;
  for(i=0;i<iloop;i+=M/L){
	for(y=0; y<32; y+=32/N*L){
	  tmp = 0.0;
	  for(x=0; x<32; x+=N){
		tmp += mat[(y+threadIdx.x/(32*(M/L)))*32 + threadIdx.x%N+x + (i+threadIdx.x/(N*L))*32*32];// * vec[threadIdx.x%N+x];
	  }
	  for(int offset=N/2; offset>0; offset/=2){
		tmp += __shfl_down_sync
		  (0xffffffff, tmp, offset, 32);
	  }
	  if(threadIdx.x%N==0)out[y+threadIdx.x/(32*(M/L)) + (i+threadIdx.x/(N*L))*32] += tmp;
	}
  }
#endif
#if 0
  if(N==32){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=M/L){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x%(32*L)/32)*32 + threadIdx.x%N+x + (i+threadIdx.x/(32*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+threadIdx.x%(32*L)/32 + (i+threadIdx.x/(32*L))*32] += tmp;
	  }
	}
  }else if(N==16){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=(32*M)/(N*L)){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x%(N*L)/N)*32 + threadIdx.x%N+x + (i+threadIdx.x/(N*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+threadIdx.x%(N*L)/N + (i+threadIdx.x/(N*L))*32] += tmp;
	  }
	}
  }
#endif

#if 0 // correct, notice for overrun
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=(32*M)/(N*L)){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+(threadIdx.x%(N*L))/N)*32 + threadIdx.x%N+x + (i+threadIdx.x/(N*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+(threadIdx.x%(N*L))/N + (i+threadIdx.x/(N*L))*32] += tmp;
	  }
	}
#endif

#if 1
	// complete
	int i, x, y;
	double tmp = 0.0;
	for(i=threadIdx.x/(N*L);i<iloop;i+=(32*M)/(N*L)){
	  for(y=(threadIdx.x%(N*L))/N; y<32; y+=L){
		tmp = 0.0;
		for(x=threadIdx.x%N; x<32; x+=N){
		  tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + i*32] += tmp;
	  }
	}
#endif

#if 0
	// open loop: no performance differences
	int i, x, y;
	double tmp = 0.0;
	for(i=threadIdx.x/(N*L);i<iloop;i+=(32*M)/(N*L)){
	  for(y=(threadIdx.x%(N*L))/N; y<32; y+=L){
		tmp = 0.0;
		for(x=threadIdx.x%N; x<32; x+=N){
		  tmp += mat[y*32 + x + i*32*32];// * vec[threadIdx.x%N+x];
		}
		if(N>=32)tmp += __shfl_down_sync(0xffffffff, tmp, 16, 32);
		if(N>=16)tmp += __shfl_down_sync(0xffffffff, tmp,  8, 32);
		if(N>= 8)tmp += __shfl_down_sync(0xffffffff, tmp,  4, 32);
		if(N>= 4)tmp += __shfl_down_sync(0xffffffff, tmp,  2, 32);
		if(N>= 2)tmp += __shfl_down_sync(0xffffffff, tmp,  1, 32);
		if(threadIdx.x%N==0)out[y + i*32] += tmp;
	  }
	}
#endif

#if 0
  if(M==2&&L==1){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=2){
	  for(y=0; y<32; y+=1){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y)*32 + threadIdx.x%N+x + (i+threadIdx.x/32)*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y + (i+threadIdx.x/32)*32] += tmp;
	  }
	}
  }
  if(M==2&&L==2){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=1){
	  for(y=0; y<32; y+=2){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x/32)*32 + threadIdx.x%N+x + (i)*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+(threadIdx.x/32) + i*32] += tmp;
	  }
	}
  }
  // M=4, 128threads/block
  if(M==4&&L==1){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=M/L){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x%(32*L)/32)*32 + threadIdx.x%N+x + (i+threadIdx.x/(32*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+threadIdx.x%(32*L)/32 + (i+threadIdx.x/(32*1))*32] += tmp;
	  }
	}
  }
  if(M==4&&L==2){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=M/L){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x%(32*L)/32)*32 + threadIdx.x%N+x + (i+threadIdx.x/(32*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+threadIdx.x%(32*L)/32 + (i+threadIdx.x/(32*L))*32] += tmp;
	  }
	}
  }
  if(M==4&&L==4){
	int i, x, y;
	double tmp = 0.0;
	for(i=0;i<iloop;i+=M/L){
	  for(y=0; y<32; y+=L){
		tmp = 0.0;
		for(x=0; x<32; x+=N){
		  tmp += mat[(y+threadIdx.x%(32*L)/32)*32 + threadIdx.x%N+x + (i+threadIdx.x/(32*L))*32*32];// * vec[threadIdx.x%N+x];
		}
		for(int offset=N/2; offset>0; offset/=2){
		  tmp += __shfl_down_sync
			(0xffffffff, tmp, offset, 32);
		}
		if(threadIdx.x%N==0)out[y+threadIdx.x%(32*L)/32 + (i+threadIdx.x/(32*L))*32] += tmp;
	  }
	}
  }
#endif
}

// ######## ######## ######## ######## ######## ######## ######## ########

void swap(int *a, int *b)
{
  int x = *a;
  *a = *b;
  *b = x;
}

int main(int argc, char **argv)
{
  int len = 32 * 32 * NLOOP;
  int i, x;
  double *out, *mat, *vec;
  double *dout, *dmat, *dvec;
  double d;
  int w, h;

  w = h = 32;
  if(argc==3){
	w = atoi(argv[1]);
	h = atoi(argv[2]);
	printf("w=%d, h=%d\n",w,h); fflush(stdout);
  }
  len = w * h * NLOOP;

  out = (double*)malloc(sizeof(double)*h*NLOOP);
  mat = (double*)malloc(sizeof(double)*len);
  vec = (double*)malloc(sizeof(double)*w);

  cudaMalloc((void**)&dout, sizeof(double)*h*NLOOP);
  cudaMalloc((void**)&dmat, sizeof(double)*len);
  cudaMalloc((void**)&dvec, sizeof(double)*w);

#define BENCH1(KERNEL,NAME,ILOOP)						\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<<<1,32>>>(dout, dmat, dvec, ILOOP);				\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
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
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout);\
  }

#define BENCH2(KERNEL,N,NAME,ILOOP)				\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N><<<1,32>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
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
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout); \
  }

#define BENCH30(KERNEL,N,M,NAME,ILOOP)			\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M><<<1,32*M>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M><<<1,32*M>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout);\
  }

#define BENCH31(KERNEL,N,M,NAME,ILOOP,W,H)		\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M><<<1,32*M>>>(dout, dmat, dvec, ILOOP,W,H);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M><<<1,32*M>>>(dout, dmat, dvec, ILOOP,W,H);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout);\
  }

#define BENCH40(KERNEL,N,M,L,NAME,ILOOP)			\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M,L><<<1,32*M>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M,L><<<1,32*M>>>(dout, dmat, dvec, ILOOP);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout);\
  }

#define BENCH41(KERNEL,N,M,L,NAME,ILOOP,W,H)		\
  {\
	for(i=0;i<h*NLOOP;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<w;i++){\
	  vec[i] = sin((double)i/10.0);\
	}\
	for(i=0;i<len;i++){\
	  mat[i] = (double)(i+1)/100.0;				\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*h*NLOOP, cudaMemcpyHostToDevice);\
	cudaMemcpy(dmat, mat, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dvec, vec, sizeof(double)*w, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M,L><<<1,32*M>>>(dout, dmat, dvec, ILOOP,W,H);	\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*h*NLOOP, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  KERNEL<N,M,L><<<1,32*M>>>(dout, dmat, dvec, ILOOP,W,H);	\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", NAME, d);\
	d=0.0; for(i=0;i<h*NLOOP;i++)d+=out[i]*(double)(i+1)/10000.0; printf("d=%.2f\n",d); fflush(stdout);\
  }

  // W-major
  /*
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
  */

  /*
for n in 32 16 8 4 2 1
do
echo "printf(\"W-major A${n}\\n\");"
for m in `seq 1 32`
do
echo "BENCH3(gpukernelAMWNr, ${n},  ${m}, \"A${m}MW${n}r\", IEXEC);"
done
done

for n in 32 16 8 4 2 1
do
echo "printf(\"W-major B${n}\\n\");"
for m in `seq 1 32`
do
echo "BENCH3(gpukernelBMWNr, ${n},  ${m}, \"B${m}MW${n}r\", IEXEC);"
done
done
   */

  BENCH30(gpukernelAMWNr, 32,  1, "A1MW32r", IEXEC);


  // 32x32 only kernel
#ifdef KERNEL32
  for(x=0;x<1;x++){
printf("W-major A32\n");
BENCH30(gpukernelAMWNr2, 32,  1, "AM01WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  2, "AM02WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  3, "AM03WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  4, "AM04WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  5, "AM05WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  6, "AM06WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  7, "AM07WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  8, "AM08WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  9, "AM09WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  10, "AM10WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  11, "AM11WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  12, "AM12WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  13, "AM13WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  14, "AM14WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  15, "AM15WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  16, "AM16WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  17, "AM17WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  18, "AM18WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  19, "AM19WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  20, "AM20WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  21, "AM21WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  22, "AM22WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  23, "AM23WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  24, "AM24WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  25, "AM25WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  26, "AM26WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  27, "AM27WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  28, "AM28WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  29, "AM29WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  30, "AM30WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  31, "AM31WN32r", IEXEC);
BENCH30(gpukernelAMWNr2, 32,  32, "AM32WN32r", IEXEC);
printf("W-major A16\n");
BENCH30(gpukernelAMWNr2, 16,  1, "AM01WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  2, "AM02WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  3, "AM03WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  4, "AM04WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  5, "AM05WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  6, "AM06WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  7, "AM07WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  8, "AM08WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  9, "AM09WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  10, "AM10WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  11, "AM11WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  12, "AM12WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  13, "AM13WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  14, "AM14WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  15, "AM15WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  16, "AM16WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  17, "AM17WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  18, "AM18WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  19, "AM19WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  20, "AM20WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  21, "AM21WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  22, "AM22WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  23, "AM23WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  24, "AM24WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  25, "AM25WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  26, "AM26WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  27, "AM27WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  28, "AM28WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  29, "AM29WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  30, "AM30WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  31, "AM31WN16r", IEXEC);
BENCH30(gpukernelAMWNr2, 16,  32, "AM32WN16r", IEXEC);
printf("W-major A8\n");
BENCH30(gpukernelAMWNr2, 8,  1, "AM01WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  2, "AM02WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  3, "AM03WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  4, "AM04WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  5, "AM05WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  6, "AM06WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  7, "AM07WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  8, "AM08WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  9, "AM09WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  10, "AM10WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  11, "AM11WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  12, "AM12WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  13, "AM13WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  14, "AM14WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  15, "AM15WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  16, "AM16WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  17, "AM17WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  18, "AM18WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  19, "AM19WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  20, "AM20WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  21, "AM21WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  22, "AM22WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  23, "AM23WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  24, "AM24WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  25, "AM25WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  26, "AM26WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  27, "AM27WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  28, "AM28WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  29, "AM29WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  30, "AM30WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  31, "AM31WN08r", IEXEC);
BENCH30(gpukernelAMWNr2, 8,  32, "AM32WN08r", IEXEC);
printf("W-major A4\n");
BENCH30(gpukernelAMWNr2, 4,  1, "AM01WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  2, "AM02WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  3, "AM03WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  4, "AM04WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  5, "AM05WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  6, "AM06WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  7, "AM07WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  8, "AM08WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  9, "AM09WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  10, "AM10WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  11, "AM11WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  12, "AM12WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  13, "AM13WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  14, "AM14WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  15, "AM15WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  16, "AM16WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  17, "AM17WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  18, "AM18WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  19, "AM19WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  20, "AM20WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  21, "AM21WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  22, "AM22WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  23, "AM23WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  24, "AM24WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  25, "AM25WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  26, "AM26WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  27, "AM27WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  28, "AM28WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  29, "AM29WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  30, "AM30WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  31, "AM31WN04r", IEXEC);
BENCH30(gpukernelAMWNr2, 4,  32, "AM32WN04r", IEXEC);
printf("W-major A2\n");
BENCH30(gpukernelAMWNr2, 2,  1, "AM01WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  2, "AM02WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  3, "AM03WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  4, "AM04WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  5, "AM05WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  6, "AM06WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  7, "AM07WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  8, "AM08WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  9, "AM09WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  10, "AM10WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  11, "AM11WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  12, "AM12WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  13, "AM13WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  14, "AM14WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  15, "AM15WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  16, "AM16WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  17, "AM17WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  18, "AM18WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  19, "AM19WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  20, "AM20WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  21, "AM21WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  22, "AM22WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  23, "AM23WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  24, "AM24WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  25, "AM25WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  26, "AM26WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  27, "AM27WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  28, "AM28WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  29, "AM29WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  30, "AM30WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  31, "AM31WN02r", IEXEC);
BENCH30(gpukernelAMWNr2, 2,  32, "AM32WN02r", IEXEC);
printf("W-major A1\n");
BENCH30(gpukernelAMWNr2, 1,  1, "AM01WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  2, "AM02WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  3, "AM03WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  4, "AM04WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  5, "AM05WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  6, "AM06WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  7, "AM07WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  8, "AM08WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  9, "AM09WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  10, "AM10WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  11, "AM11WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  12, "AM12WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  13, "AM13WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  14, "AM14WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  15, "AM15WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  16, "AM16WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  17, "AM17WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  18, "AM18WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  19, "AM19WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  20, "AM20WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  21, "AM21WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  22, "AM22WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  23, "AM23WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  24, "AM24WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  25, "AM25WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  26, "AM26WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  27, "AM27WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  28, "AM28WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  29, "AM29WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  30, "AM30WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  31, "AM31WN01r", IEXEC);
BENCH30(gpukernelAMWNr2, 1,  32, "AM32WN01r", IEXEC);
  }

  for(x=0;x<1;x++){
printf("W-major B32\n");
BENCH30(gpukernelBMWNr2, 32,  1, "BM01W32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  2, "BM02WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  3, "BM03WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  4, "BM04WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  5, "BM05WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  6, "BM06WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  7, "BM07WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  8, "BM08WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  9, "BM09WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  10, "BM10WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  11, "BM11WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  12, "BM12WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  13, "BM13WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  14, "BM14WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  15, "BM15WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  16, "BM16WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  17, "BM17WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  18, "BM18WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  19, "BM19WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  20, "BM20WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  21, "BM21WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  22, "BM22WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  23, "BM23WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  24, "BM24WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  25, "BM25WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  26, "BM26WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  27, "BM27WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  28, "BM28WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  29, "BM29WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  30, "BM30WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  31, "BM31WN32r", IEXEC);
BENCH30(gpukernelBMWNr2, 32,  32, "BM32WN32r", IEXEC);
printf("W-major B16\n");
BENCH30(gpukernelBMWNr2, 16,  1, "BM01W16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  2, "BM02WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  3, "BM03WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  4, "BM04WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  5, "BM05WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  6, "BM06WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  7, "BM07WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  8, "BM08WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  9, "BM09WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  10, "BM10WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  11, "BM11WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  12, "BM12WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  13, "BM13WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  14, "BM14WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  15, "BM15WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  16, "BM16WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  17, "BM17WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  18, "BM18WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  19, "BM19WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  20, "BM20WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  21, "BM21WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  22, "BM22WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  23, "BM23WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  24, "BM24WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  25, "BM25WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  26, "BM26WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  27, "BM27WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  28, "BM28WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  29, "BM29WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  30, "BM30WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  31, "BM31WN16r", IEXEC);
BENCH30(gpukernelBMWNr2, 16,  32, "BM32WN16r", IEXEC);
printf("W-major B8\n");
BENCH30(gpukernelBMWNr2, 8,  1, "BM01WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  2, "BM02WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  3, "BM03WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  4, "BM04WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  5, "BM05WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  6, "BM06WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  7, "BM07WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  8, "BM08WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  9, "BM09WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  10, "BM10WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  11, "BM11WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  12, "BM12WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  13, "BM13WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  14, "BM14WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  15, "BM15WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  16, "BM16WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  17, "BM17WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  18, "BM18WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  19, "BM19WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  20, "BM20WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  21, "BM21WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  22, "BM22WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  23, "BM23WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  24, "BM24WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  25, "BM25WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  26, "BM26WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  27, "BM27WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  28, "BM28WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  29, "BM29WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  30, "BM30WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  31, "BM31WN08r", IEXEC);
BENCH30(gpukernelBMWNr2, 8,  32, "BM32WN08r", IEXEC);
printf("W-major B4\n");
BENCH30(gpukernelBMWNr2, 4,  1, "BM01WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  2, "BM02WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  3, "BM03WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  4, "BM04WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  5, "BM05WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  6, "BM06WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  7, "BM07WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  8, "BM08WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  9, "BM09WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  10, "BM10WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  11, "BM11WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  12, "BM12WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  13, "BM13WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  14, "BM14WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  15, "BM15WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  16, "BM16WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  17, "BM17WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  18, "BM18WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  19, "BM19WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  20, "BM20WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  21, "BM21WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  22, "BM22WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  23, "BM23WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  24, "BM24WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  25, "BM25WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  26, "BM26WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  27, "BM27WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  28, "BM28WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  29, "BM29WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  30, "BM30WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  31, "BM31WN04r", IEXEC);
BENCH30(gpukernelBMWNr2, 4,  32, "BM32WN04r", IEXEC);
printf("W-major B2\n");
BENCH30(gpukernelBMWNr2, 2,  1, "BM01WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  2, "BM02WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  3, "BM03WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  4, "BM04WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  5, "BM05WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  6, "BM06WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  7, "BM07WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  8, "BM08WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  9, "BM09WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  10, "BM10WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  11, "BM11WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  12, "BM12WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  13, "BM13WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  14, "BM14WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  15, "BM15WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  16, "BM16WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  17, "BM17WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  18, "BM18WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  19, "BM19WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  20, "BM20WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  21, "BM21WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  22, "BM22WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  23, "BM23WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  24, "BM24WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  25, "BM25WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  26, "BM26WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  27, "BM27WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  28, "BM28WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  29, "BM29WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  30, "BM30WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  31, "BM31WN02r", IEXEC);
BENCH30(gpukernelBMWNr2, 2,  32, "BM32WN02r", IEXEC);
printf("W-major B1\n");
BENCH30(gpukernelBMWNr2, 1,  1, "BM01WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  2, "BM02WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  3, "BM03WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  4, "BM04WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  5, "BM05WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  6, "BM06WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  7, "BM07WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  8, "BM08WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  9, "BM09WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  10, "BM10WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  11, "BM11WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  12, "BM12WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  13, "BM13WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  14, "BM14WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  15, "BM15WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  16, "BM16WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  17, "BM17WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  18, "BM18WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  19, "BM19WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  20, "BM20WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  21, "BM21WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  22, "BM22WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  23, "BM23WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  24, "BM24WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  25, "BM25WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  26, "BM26WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  27, "BM27WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  28, "BM28WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  29, "BM29WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  30, "BM30WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  31, "BM31WN01r", IEXEC);
BENCH30(gpukernelBMWNr2, 1,  32, "BM32WN01r", IEXEC);
  }
#endif

  // free size kernel
#ifndef KERNEL32
  for(x=0;x<1;x++){
printf("W-major A32\n");
BENCH31(gpukernelAMWNr3, 32,  1, "AM01WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  2, "AM02WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  3, "AM03WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  4, "AM04WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  5, "AM05WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  6, "AM06WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  7, "AM07WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  8, "AM08WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  9, "AM09WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  10, "AM10WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  11, "AM11WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  12, "AM12WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  13, "AM13WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  14, "AM14WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  15, "AM15WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  16, "AM16WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  17, "AM17WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  18, "AM18WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  19, "AM19WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  20, "AM20WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  21, "AM21WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  22, "AM22WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  23, "AM23WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  24, "AM24WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  25, "AM25WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  26, "AM26WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  27, "AM27WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  28, "AM28WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  29, "AM29WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  30, "AM30WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  31, "AM31WN32r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 32,  32, "AM32WN32r", IEXEC, w, h);
printf("W-major A16\n");
BENCH31(gpukernelAMWNr3, 16,  1, "AM01WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  2, "AM02WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  3, "AM03WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  4, "AM04WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  5, "AM05WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  6, "AM06WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  7, "AM07WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  8, "AM08WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  9, "AM09WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  10, "AM10WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  11, "AM11WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  12, "AM12WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  13, "AM13WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  14, "AM14WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  15, "AM15WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  16, "AM16WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  17, "AM17WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  18, "AM18WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  19, "AM19WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  20, "AM20WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  21, "AM21WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  22, "AM22WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  23, "AM23WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  24, "AM24WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  25, "AM25WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  26, "AM26WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  27, "AM27WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  28, "AM28WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  29, "AM29WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  30, "AM30WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  31, "AM31WN16r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 16,  32, "AM32WN16r", IEXEC, w, h);
printf("W-major A8\n");
BENCH31(gpukernelAMWNr3, 8,  1, "AM01WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  2, "AM02WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  3, "AM03WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  4, "AM04WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  5, "AM05WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  6, "AM06WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  7, "AM07WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  8, "AM08WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  9, "AM09WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  10, "AM10WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  11, "AM11WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  12, "AM12WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  13, "AM13WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  14, "AM14WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  15, "AM15WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  16, "AM16WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  17, "AM17WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  18, "AM18WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  19, "AM19WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  20, "AM20WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  21, "AM21WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  22, "AM22WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  23, "AM23WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  24, "AM24WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  25, "AM25WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  26, "AM26WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  27, "AM27WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  28, "AM28WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  29, "AM29WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  30, "AM30WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  31, "AM31WN08r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 8,  32, "AM32WN08r", IEXEC, w, h);
printf("W-major A4\n");
BENCH31(gpukernelAMWNr3, 4,  1, "AM01WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  2, "AM02WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  3, "AM03WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  4, "AM04WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  5, "AM05WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  6, "AM06WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  7, "AM07WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  8, "AM08WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  9, "AM09WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  10, "AM10WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  11, "AM11WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  12, "AM12WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  13, "AM13WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  14, "AM14WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  15, "AM15WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  16, "AM16WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  17, "AM17WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  18, "AM18WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  19, "AM19WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  20, "AM20WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  21, "AM21WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  22, "AM22WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  23, "AM23WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  24, "AM24WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  25, "AM25WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  26, "AM26WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  27, "AM27WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  28, "AM28WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  29, "AM29WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  30, "AM30WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  31, "AM31WN04r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 4,  32, "AM32WN04r", IEXEC, w, h);
printf("W-major A2\n");
BENCH31(gpukernelAMWNr3, 2,  1, "AM01WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  2, "AM02WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  3, "AM03WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  4, "AM04WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  5, "AM05WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  6, "AM06WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  7, "AM07WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  8, "AM08WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  9, "AM09WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  10, "AM10WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  11, "AM11WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  12, "AM12WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  13, "AM13WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  14, "AM14WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  15, "AM15WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  16, "AM16WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  17, "AM17WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  18, "AM18WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  19, "AM19WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  20, "AM20WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  21, "AM21WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  22, "AM22WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  23, "AM23WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  24, "AM24WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  25, "AM25WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  26, "AM26WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  27, "AM27WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  28, "AM28WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  29, "AM29WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  30, "AM30WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  31, "AM31WN02r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 2,  32, "AM32WN02r", IEXEC, w, h);
printf("W-major A1\n");
BENCH31(gpukernelAMWNr3, 1,  1, "AM01WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  2, "AM02WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  3, "AM03WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  4, "AM04WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  5, "AM05WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  6, "AM06WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  7, "AM07WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  8, "AM08WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  9, "AM09WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  10, "AM10WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  11, "AM11WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  12, "AM12WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  13, "AM13WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  14, "AM14WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  15, "AM15WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  16, "AM16WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  17, "AM17WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  18, "AM18WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  19, "AM19WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  20, "AM20WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  21, "AM21WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  22, "AM22WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  23, "AM23WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  24, "AM24WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  25, "AM25WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  26, "AM26WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  27, "AM27WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  28, "AM28WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  29, "AM29WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  30, "AM30WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  31, "AM31WN01r", IEXEC, w, h);
BENCH31(gpukernelAMWNr3, 1,  32, "AM32WN01r", IEXEC, w, h);
  }

  for(x=0;x<1;x++){
printf("W-major B32\n");
BENCH31(gpukernelBMWNr3, 32,  1, "BM01W32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  2, "BM02WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  3, "BM03WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  4, "BM04WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  5, "BM05WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  6, "BM06WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  7, "BM07WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  8, "BM08WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  9, "BM09WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  10, "BM10WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  11, "BM11WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  12, "BM12WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  13, "BM13WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  14, "BM14WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  15, "BM15WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  16, "BM16WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  17, "BM17WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  18, "BM18WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  19, "BM19WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  20, "BM20WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  21, "BM21WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  22, "BM22WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  23, "BM23WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  24, "BM24WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  25, "BM25WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  26, "BM26WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  27, "BM27WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  28, "BM28WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  29, "BM29WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  30, "BM30WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  31, "BM31WN32r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 32,  32, "BM32WN32r", IEXEC, w, h);
printf("W-major B16\n");
BENCH31(gpukernelBMWNr3, 16,  1, "BM01W16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  2, "BM02WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  3, "BM03WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  4, "BM04WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  5, "BM05WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  6, "BM06WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  7, "BM07WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  8, "BM08WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  9, "BM09WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  10, "BM10WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  11, "BM11WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  12, "BM12WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  13, "BM13WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  14, "BM14WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  15, "BM15WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  16, "BM16WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  17, "BM17WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  18, "BM18WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  19, "BM19WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  20, "BM20WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  21, "BM21WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  22, "BM22WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  23, "BM23WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  24, "BM24WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  25, "BM25WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  26, "BM26WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  27, "BM27WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  28, "BM28WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  29, "BM29WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  30, "BM30WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  31, "BM31WN16r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 16,  32, "BM32WN16r", IEXEC, w, h);
printf("W-major B8\n");
BENCH31(gpukernelBMWNr3, 8,  1, "BM01WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  2, "BM02WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  3, "BM03WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  4, "BM04WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  5, "BM05WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  6, "BM06WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  7, "BM07WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  8, "BM08WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  9, "BM09WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  10, "BM10WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  11, "BM11WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  12, "BM12WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  13, "BM13WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  14, "BM14WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  15, "BM15WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  16, "BM16WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  17, "BM17WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  18, "BM18WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  19, "BM19WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  20, "BM20WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  21, "BM21WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  22, "BM22WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  23, "BM23WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  24, "BM24WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  25, "BM25WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  26, "BM26WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  27, "BM27WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  28, "BM28WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  29, "BM29WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  30, "BM30WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  31, "BM31WN08r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 8,  32, "BM32WN08r", IEXEC, w, h);
printf("W-major B4\n");
BENCH31(gpukernelBMWNr3, 4,  1, "BM01WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  2, "BM02WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  3, "BM03WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  4, "BM04WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  5, "BM05WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  6, "BM06WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  7, "BM07WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  8, "BM08WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  9, "BM09WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  10, "BM10WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  11, "BM11WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  12, "BM12WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  13, "BM13WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  14, "BM14WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  15, "BM15WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  16, "BM16WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  17, "BM17WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  18, "BM18WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  19, "BM19WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  20, "BM20WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  21, "BM21WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  22, "BM22WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  23, "BM23WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  24, "BM24WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  25, "BM25WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  26, "BM26WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  27, "BM27WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  28, "BM28WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  29, "BM29WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  30, "BM30WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  31, "BM31WN04r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 4,  32, "BM32WN04r", IEXEC, w, h);
printf("W-major B2\n");
BENCH31(gpukernelBMWNr3, 2,  1, "BM01WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  2, "BM02WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  3, "BM03WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  4, "BM04WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  5, "BM05WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  6, "BM06WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  7, "BM07WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  8, "BM08WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  9, "BM09WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  10, "BM10WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  11, "BM11WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  12, "BM12WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  13, "BM13WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  14, "BM14WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  15, "BM15WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  16, "BM16WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  17, "BM17WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  18, "BM18WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  19, "BM19WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  20, "BM20WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  21, "BM21WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  22, "BM22WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  23, "BM23WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  24, "BM24WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  25, "BM25WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  26, "BM26WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  27, "BM27WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  28, "BM28WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  29, "BM29WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  30, "BM30WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  31, "BM31WN02r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 2,  32, "BM32WN02r", IEXEC, w, h);
printf("W-major B1\n");
BENCH31(gpukernelBMWNr3, 1,  1, "BM01WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  2, "BM02WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  3, "BM03WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  4, "BM04WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  5, "BM05WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  6, "BM06WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  7, "BM07WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  8, "BM08WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  9, "BM09WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  10, "BM10WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  11, "BM11WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  12, "BM12WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  13, "BM13WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  14, "BM14WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  15, "BM15WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  16, "BM16WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  17, "BM17WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  18, "BM18WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  19, "BM19WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  20, "BM20WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  21, "BM21WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  22, "BM22WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  23, "BM23WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  24, "BM24WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  25, "BM25WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  26, "BM26WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  27, "BM27WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  28, "BM28WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  29, "BM29WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  30, "BM30WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  31, "BM31WN01r", IEXEC, w, h);
BENCH31(gpukernelBMWNr3, 1,  32, "BM32WN01r", IEXEC, w, h);
  }
#endif

  // CLMWNr
  // N M L
  // M WARPs/TB = 32*M threads / TB
  // N*L threads : 1 GEMV, N threads/line

  // BENCH40: 32x32 only kernel
  // BENCH41: free size kernel

#ifdef KERNEL32
  BENCH40(gpukernelCLMWNr2, 32,   2,  1, "CL01M02WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   2,  2, "CL02M02WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   4,  1, "CL01M04WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   4,  2, "CL02M04WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   4,  4, "CL04M04WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   8,  1, "CL01M08WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   8,  2, "CL02M08WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   8,  4, "CL04M08WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,   8,  8, "CL08M08WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  16,  1, "CL01M16WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  16,  2, "CL02M16WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  16,  4, "CL04M16WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  16,  8, "CL08M16WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  16, 16, "CL16M16WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32,  1, "CL01M32WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32,  2, "CL02M32WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32,  4, "CL04M32WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32,  8, "CL08M32WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32, 16, "CL16M32WN32r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 32,  32, 32, "CL32M32WN32r", IEXEC);

  BENCH40(gpukernelCLMWNr2, 16,   2,  1, "CL01M02WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   2,  2, "CL02M02WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   4,  1, "CL01M04WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   4,  2, "CL02M04WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   4,  4, "CL04M04WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   8,  1, "CL01M08WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   8,  2, "CL02M08WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   8,  4, "CL04M08WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,   8,  8, "CL08M08WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  16,  1, "CL01M16WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  16,  2, "CL02M16WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  16,  4, "CL04M16WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  16,  8, "CL08M16WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  16, 16, "CL16M16WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32,  1, "CL01M32WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32,  2, "CL02M32WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32,  4, "CL04M32WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32,  8, "CL08M32WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32, 16, "CL16M32WN16r", IEXEC);
  BENCH40(gpukernelCLMWNr2, 16,  32, 32, "CL32M32WN16r", IEXEC);

  BENCH40(gpukernelCLMWNr2,  8,   2,  1, "CL01M02WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   2,  2, "CL02M02WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   4,  1, "CL01M04WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   4,  2, "CL02M04WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   4,  4, "CL04M04WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   8,  1, "CL01M08WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   8,  2, "CL02M08WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   8,  4, "CL04M08WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,   8,  8, "CL08M08WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  16,  1, "CL01M16WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  16,  2, "CL02M16WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  16,  4, "CL04M16WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  16,  8, "CL08M16WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  16, 16, "CL16M16WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32,  1, "CL01M32WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32,  2, "CL02M32WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32,  4, "CL04M32WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32,  8, "CL08M32WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32, 16, "CL16M32WN08r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  8,  32, 32, "CL32M32WN08r", IEXEC);

  BENCH40(gpukernelCLMWNr2,  4,   2,  1, "CL01M02WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   2,  2, "CL02M02WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   4,  1, "CL01M04WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   4,  2, "CL02M04WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   4,  4, "CL04M04WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   8,  1, "CL01M08WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   8,  2, "CL02M08WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   8,  4, "CL04M08WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,   8,  8, "CL08M08WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  16,  1, "CL01M16WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  16,  2, "CL02M16WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  16,  4, "CL04M16WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  16,  8, "CL08M16WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  16, 16, "CL16M16WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32,  1, "CL01M32WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32,  2, "CL02M32WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32,  4, "CL04M32WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32,  8, "CL08M32WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32, 16, "CL16M32WN04r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  4,  32, 32, "CL32M32WN04r", IEXEC);

  BENCH40(gpukernelCLMWNr2,  2,   2,  1, "CL01M02WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   2,  2, "CL02M02WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   4,  1, "CL01M04WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   4,  2, "CL02M04WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   4,  4, "CL04M04WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   8,  1, "CL01M08WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   8,  2, "CL02M08WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   8,  4, "CL04M08WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,   8,  8, "CL08M08WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  16,  1, "CL01M16WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  16,  2, "CL02M16WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  16,  4, "CL04M16WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  16,  8, "CL08M16WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  16, 16, "CL16M16WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32,  1, "CL01M32WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32,  2, "CL02M32WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32,  4, "CL04M32WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32,  8, "CL08M32WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32, 16, "CL16M32WN02r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  2,  32, 32, "CL32M32WN02r", IEXEC);

  BENCH40(gpukernelCLMWNr2,  1,   2,  1, "CL01M02WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   2,  2, "CL02M02WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   4,  1, "CL01M04WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   4,  2, "CL02M04WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   4,  4, "CL04M04WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   8,  1, "CL01M08WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   8,  2, "CL02M08WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   8,  4, "CL04M08WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,   8,  8, "CL08M08WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  16,  1, "CL01M16WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  16,  2, "CL02M16WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  16,  4, "CL04M16WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  16,  8, "CL08M16WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  16, 16, "CL16M16WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32,  1, "CL01M32WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32,  2, "CL02M32WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32,  4, "CL04M32WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32,  8, "CL08M32WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32, 16, "CL16M32WN01r", IEXEC);
  BENCH40(gpukernelCLMWNr2,  1,  32, 32, "CL32M32WN01r", IEXEC);
#endif

#ifndef KERNEL32
  BENCH41(gpukernelCLMWNr3, 32,   2,  1, "CL01M02WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   2,  2, "CL02M02WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   4,  1, "CL01M04WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   4,  2, "CL02M04WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   4,  4, "CL04M04WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   8,  1, "CL01M08WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   8,  2, "CL02M08WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   8,  4, "CL04M08WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,   8,  8, "CL08M08WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  16,  1, "CL01M16WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  16,  2, "CL02M16WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  16,  4, "CL04M16WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  16,  8, "CL08M16WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  16, 16, "CL16M16WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32,  1, "CL01M32WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32,  2, "CL02M32WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32,  4, "CL04M32WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32,  8, "CL08M32WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32, 16, "CL16M32WN32r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 32,  32, 32, "CL32M32WN32r", IEXEC, w, h);

  BENCH41(gpukernelCLMWNr3, 16,   2,  1, "CL01M02WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   2,  2, "CL02M02WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   4,  1, "CL01M04WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   4,  2, "CL02M04WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   4,  4, "CL04M04WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   8,  1, "CL01M08WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   8,  2, "CL02M08WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   8,  4, "CL04M08WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,   8,  8, "CL08M08WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  16,  1, "CL01M16WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  16,  2, "CL02M16WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  16,  4, "CL04M16WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  16,  8, "CL08M16WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  16, 16, "CL16M16WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32,  1, "CL01M32WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32,  2, "CL02M32WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32,  4, "CL04M32WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32,  8, "CL08M32WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32, 16, "CL16M32WN16r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3, 16,  32, 32, "CL32M32WN16r", IEXEC, w, h);

  BENCH41(gpukernelCLMWNr3,  8,   2,  1, "CL01M02WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   2,  2, "CL02M02WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   4,  1, "CL01M04WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   4,  2, "CL02M04WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   4,  4, "CL04M04WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   8,  1, "CL01M08WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   8,  2, "CL02M08WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   8,  4, "CL04M08WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,   8,  8, "CL08M08WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  16,  1, "CL01M16WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  16,  2, "CL02M16WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  16,  4, "CL04M16WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  16,  8, "CL08M16WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  16, 16, "CL16M16WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32,  1, "CL01M32WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32,  2, "CL02M32WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32,  4, "CL04M32WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32,  8, "CL08M32WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32, 16, "CL16M32WN08r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  8,  32, 32, "CL32M32WN08r", IEXEC, w, h);

  BENCH41(gpukernelCLMWNr3,  4,   2,  1, "CL01M02WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   2,  2, "CL02M02WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   4,  1, "CL01M04WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   4,  2, "CL02M04WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   4,  4, "CL04M04WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   8,  1, "CL01M08WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   8,  2, "CL02M08WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   8,  4, "CL04M08WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,   8,  8, "CL08M08WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  16,  1, "CL01M16WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  16,  2, "CL02M16WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  16,  4, "CL04M16WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  16,  8, "CL08M16WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  16, 16, "CL16M16WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32,  1, "CL01M32WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32,  2, "CL02M32WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32,  4, "CL04M32WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32,  8, "CL08M32WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32, 16, "CL16M32WN04r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  4,  32, 32, "CL32M32WN04r", IEXEC, w, h);

  BENCH41(gpukernelCLMWNr3,  2,   2,  1, "CL01M02WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   2,  2, "CL02M02WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   4,  1, "CL01M04WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   4,  2, "CL02M04WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   4,  4, "CL04M04WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   8,  1, "CL01M08WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   8,  2, "CL02M08WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   8,  4, "CL04M08WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,   8,  8, "CL08M08WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  16,  1, "CL01M16WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  16,  2, "CL02M16WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  16,  4, "CL04M16WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  16,  8, "CL08M16WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  16, 16, "CL16M16WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32,  1, "CL01M32WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32,  2, "CL02M32WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32,  4, "CL04M32WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32,  8, "CL08M32WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32, 16, "CL16M32WN02r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  2,  32, 32, "CL32M32WN02r", IEXEC, w, h);

  BENCH41(gpukernelCLMWNr3,  1,   2,  1, "CL01M02WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   2,  2, "CL02M02WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   4,  1, "CL01M04WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   4,  2, "CL02M04WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   4,  4, "CL04M04WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   8,  1, "CL01M08WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   8,  2, "CL02M08WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   8,  4, "CL04M08WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,   8,  8, "CL08M08WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  16,  1, "CL01M16WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  16,  2, "CL02M16WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  16,  4, "CL04M16WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  16,  8, "CL08M16WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  16, 16, "CL16M16WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32,  1, "CL01M32WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32,  2, "CL02M32WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32,  4, "CL04M32WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32,  8, "CL08M32WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32, 16, "CL16M32WN01r", IEXEC, w, h);
  BENCH41(gpukernelCLMWNr3,  1,  32, 32, "CL32M32WN01r", IEXEC, w, h);
#endif

  cudaFree(dout); cudaFree(dmat); cudaFree(dvec);
  free(out); free(mat); free(vec);
  return 0;
}
