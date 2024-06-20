// -*- c++ -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

#define NEXEC 1
// number of warp
#define NW 3

__global__ void gpukernel_32a(double *out, double *in, int N, int *head)
{
  int n;
  int y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/32; n<N; n+=NW){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  tmp += in[y*32 + threadIdx.x%32 + h*32*32];
	}
	if(threadIdx.x%32==0)out[n] += tmp;
  }
}
__global__ void gpukernel_32b(double *out, double *in, int N, int *head)
{
  int n;
  int x;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/32; n<N; n+=NW){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x++){
	  tmp += in[(threadIdx.x%32)*32 + x + h*32*32];
	}
	if(threadIdx.x%32==0)out[n] += tmp;
  }
}

__global__ void gpukernel_16aa(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/16; n<N; n+=NW*2){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  for(x=0; x<32; x+=16){
		tmp += in[y*32 + threadIdx.x%16+x + h*32*32];
	  }
	}
	if(threadIdx.x%16==0)out[n] += tmp;
  }
}
__global__ void gpukernel_16ab(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/16; n<N; n+=NW*2){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=16){
	  for(y=0; y<32; y++){
		tmp += in[y*32 + threadIdx.x%16+x + h*32*32];
	  }
	}
	if(threadIdx.x%16==0)out[n] += tmp;
  }
}
__global__ void gpukernel_16ba(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/16; n<N; n+=NW*2){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y+=16){
	  for(x=0; x<32; x+=1){
		tmp += in[(y+threadIdx.x%16)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%16==0)out[n] += tmp;
  }
}
__global__ void gpukernel_16bb(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/16; n<N; n+=NW*2){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  for(y=0; y<32; y+=16){
		tmp += in[(y+threadIdx.x%16)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%16==0)out[n] += tmp;
  }
}

__global__ void gpukernel_8aa(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/8; n<N; n+=NW*4){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  for(x=0; x<32; x+=8){
		tmp += in[y*32 + threadIdx.x%8+x + h*32*32];
	  }
	}
	if(threadIdx.x%8==0)out[n] += tmp;
  }
}
__global__ void gpukernel_8ab(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/8; n<N; n+=NW*4){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=8){
	  for(y=0; y<32; y++){
		tmp += in[y*32 + threadIdx.x%8+x + h*32*32];
	  }
	}
	if(threadIdx.x%8==0)out[n] += tmp;
  }
}
__global__ void gpukernel_8ba(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/8; n<N; n+=NW*4){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y+=8){
	  for(x=0; x<32; x+=1){
		tmp += in[(y+threadIdx.x%8)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%8==0)out[n] += tmp;
  }
}
__global__ void gpukernel_8bb(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/8; n<N; n+=NW*4){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  for(y=0; y<32; y+=8){
		tmp += in[(y+threadIdx.x%8)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%8==0)out[n] += tmp;
  }
}

__global__ void gpukernel_4aa(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/4; n<N; n+=NW*8){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  for(x=0; x<32; x+=4){
		tmp += in[y*32 + threadIdx.x%4+x + h*32*32];
	  }
	}
	if(threadIdx.x%4==0)out[n] += tmp;
  }
}
__global__ void gpukernel_4ab(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/4; n<N; n+=NW*8){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=4){
	  for(y=0; y<32; y++){
		tmp += in[y*32 + threadIdx.x%4+x + h*32*32];
	  }
	}
	if(threadIdx.x%4==0)out[n] += tmp;
  }
}
__global__ void gpukernel_4ba(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/4; n<N; n+=NW*8){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y+=4){
	  for(x=0; x<32; x+=1){
		tmp += in[(y+threadIdx.x%4)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%4==0)out[n] += tmp;
  }
}
__global__ void gpukernel_4bb(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/4; n<N; n+=NW*8){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  for(y=0; y<32; y+=4){
		tmp += in[(y+threadIdx.x%4)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%4==0)out[n] += tmp;
  }
}

__global__ void gpukernel_2aa(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/2; n<N; n+=NW*16){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  for(x=0; x<32; x+=2){
		tmp += in[y*32 + threadIdx.x%2+x + h*32*32];
	  }
	}
	if(threadIdx.x%2==0)out[n] += tmp;
  }
}
__global__ void gpukernel_2ab(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/2; n<N; n+=NW*16){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=2){
	  for(y=0; y<32; y++){
		tmp += in[y*32 + threadIdx.x%2+x + h*32*32];
	  }
	}
	if(threadIdx.x%2==0)out[n] += tmp;
  }
}
__global__ void gpukernel_2ba(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/2; n<N; n+=NW*16){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y+=2){
	  for(x=0; x<32; x+=1){
		tmp += in[(y+threadIdx.x%2)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%2==0)out[n] += tmp;
  }
}
__global__ void gpukernel_2bb(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x/2; n<N; n+=NW*16){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  for(y=0; y<32; y+=2){
		tmp += in[(y+threadIdx.x%2)*32 + x + h*32*32];
	  }
	}
	if(threadIdx.x%2==0)out[n] += tmp;
  }
}

__global__ void gpukernel_1a(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x; n<N; n+=NW*32){
	h = head[n];
	tmp = 0.0;
	for(y=0; y<32; y++){
	  for(x=0; x<32; x+=1){
		tmp += in[y*32 + x + h*32*32];
	  }
	}
	out[n] += tmp;
  }
}
__global__ void gpukernel_1b(double *out, double *in, int N, int *head)
{
  int n;
  int x, y;
  double tmp = 0.0;
  int h;
  for(n=threadIdx.x; n<N; n+=NW*32){
	h = head[n];
	tmp = 0.0;
	for(x=0; x<32; x+=1){
	  for(y=0; y<32; y++){
		tmp += in[y*32 + x + h*32*32];
	  }
	}
	out[n] += tmp;
  }
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
  int rule = 0;
  int N = 320000;
  int len = 32 * 32 * N;
  int i, x;
  double *out, *in;
  double *dout, *din;
  double d;
  int *head, *dhead;

  if(argc>1)rule=atoi(argv[1]); printf("rule=%d\n", rule);
  //if(argc>1)N=atoi(argv[1]); printf("N=%d\n", N);
  out = (double*)malloc(sizeof(double)*32*N);
  in = (double*)malloc(sizeof(double)*len);
  head = (int*)malloc(sizeof(int)*N);

  cudaMalloc((void**)&dout, sizeof(double)*32*N);
  cudaMalloc((void**)&din, sizeof(double)*len);
  cudaMalloc((void**)&dhead, sizeof(int)*N);

#define BENCH(FUNCNAME,X) \
  {\
	for(i=0;i<32*N;i++){\
	  out[i] = 0.0;\
	}\
	for(i=0;i<len;i++){\
	  in[i] = (double)(i+1)/1000.0;\
	}\
	if(rule==0)for(i=0;i<N;i++)head[i] = i;\
	if(rule==1)for(i=0;i<N;i++)head[i] = N-1-i;\
	if(rule==2){\
	  for(i=0;i<N;i++)head[i] = i;\
	  for(i=0;i<N;i++)swap(&head[rand()%N], &head[rand()%N]);\
	}\
\
	cudaMemcpy(dout, out, sizeof(double)*32*N, cudaMemcpyHostToDevice);\
	cudaMemcpy(din, in, sizeof(double)*len, cudaMemcpyHostToDevice);\
	cudaMemcpy(dhead, head, sizeof(int)*N, cudaMemcpyHostToDevice);\
\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  FUNCNAME<<<1,32*NW>>>(dout, din, N, dhead);\
	  cudaDeviceSynchronize();\
	}\
	cudaMemcpy(out, dout, sizeof(double)*32*N, cudaMemcpyDeviceToHost);\
\
	d=omp_get_wtime();\
	for(i=0;i<NEXEC;i++){\
	  cudaDeviceSynchronize();\
	  FUNCNAME<<<1,32*NW>>>(dout, din, N, dhead);\
	  cudaDeviceSynchronize();\
	}\
	d=omp_get_wtime()-d;\
\
	printf("%s: time %f msec(/total), ", X, d);\
	d=0.0; for(i=0;i<32*N;i++)d+=out[i]; printf("d=%.2f\n",d);\
  }

  BENCH(gpukernel_1a ,  "  1a ");
  BENCH(gpukernel_1b ,  "  1b ");
  BENCH(gpukernel_32a,  " 32a ");
  BENCH(gpukernel_32b,  " 32b ");
  BENCH(gpukernel_16aa, " 16aa");
  BENCH(gpukernel_16ab, " 16ab");
  BENCH(gpukernel_16ba, " 16ba");
  BENCH(gpukernel_16bb, " 16bb");
  BENCH(gpukernel_8aa,  "  8aa");
  BENCH(gpukernel_8ab,  "  8ab");
  BENCH(gpukernel_8ba,  "  8ba");
  BENCH(gpukernel_8bb,  "  8bb");
  BENCH(gpukernel_4aa,  "  4aa");
  BENCH(gpukernel_4ab,  "  4ab");
  BENCH(gpukernel_4ba,  "  4ba");
  BENCH(gpukernel_4bb,  "  4bb");
  BENCH(gpukernel_2aa,  "  2aa");
  BENCH(gpukernel_2ab,  "  2ab");
  BENCH(gpukernel_2ba,  "  2ba");
  BENCH(gpukernel_2bb,  "  2bb");
  BENCH(gpukernel_1a,   "  1a ");
  BENCH(gpukernel_1b,   "  1b ");

  cudaFree(dout);  cudaFree(din);
  free(out); free(in);
  return 0;
}
