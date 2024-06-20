/* -*- C++ -*- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cublas_v2.h>

/*
  matrix multiply A = B*C on CUDA
*/

#ifdef _DOUBLE
#define real double
#endif
#ifdef _SINGLE
#define real float
#endif

#include "time_cpu_struct.h"
#include "time_gpu_struct.h"
struct SGpuTime gputimeH2D_1;
struct SGpuTime gputimeH2D_2;
struct SGpuTime gputimeKernel;
struct SGpuTime gputimeD2H;

#define TIME_INITIALIZE();							\
  TimeInitialize_GPU(&gputimeH2D_1,"gputimeH2D_1");\
  TimeInitialize_GPU(&gputimeH2D_2,"gputimeH2D_2");\
  TimeInitialize_GPU(&gputimeKernel,"gputimeKernel");\
  TimeInitialize_GPU(&gputimeD2H,"gputimeD2H");
#define DO_H2D_1(o); TimeBegin_GPU(&gputimeH2D_1); ret=o; TimeEnd1_GPU(&gputimeH2D_1);
#define DO_H2D_2(o); TimeBegin_GPU(&gputimeH2D_2); ret=o; TimeEnd1_GPU(&gputimeH2D_2);
#define DO_KERNEL3(o,n); TimeBegin_GPU(&gputimeKernel); ret=o; TimeEnd2_GPU(&gputimeKernel, n);
#define DO_KERNEL2(o,n); TimeBegin_GPU(&gputimeKernel); ret=o; TimeEnd2_GPU(&gputimeKernel, 1.0e-9*2.0*nelem);
#define DO_KERNEL1(o); TimeBegin_GPU(&gputimeKernel); ret=o; TimeEnd1_GPU(&gputimeKernel);
#define DO_D2H(o); TimeBegin_GPU(&gputimeD2H); ret=o; TimeEnd1_GPU(&gputimeD2H);

int SIZE = 16;
int RANDSEED = 0;
real *g_A, *g_B, *g_C;
real *g_dbg;
int KERNEL = 0;
real *d_A, *d_B, *d_C;
real *d_dbg;
cublasStatus_t stat;
cublasHandle_t handle;
int BLOCKS=256, THREADS=32;
int LOOPS=1;
int SMLIMIT=0;

int checkArgs(int argc, char** argv)
{
  int i;
  for(i=1; i<argc; i++){
	if(strcmp(argv[i], "-size")==0){
	  SIZE = atoi(argv[++i]);
	}
	if(strcmp(argv[i], "-kernel")==0){
	  KERNEL = atoi(argv[++i]);
	}
	if(strcmp(argv[i], "-blocks")==0){
	  BLOCKS = atoi(argv[++i]);
	  printf(" %d block(s)\n", BLOCKS);
	}
	if(strcmp(argv[i], "-threads")==0){
	  THREADS = atoi(argv[++i]);
	  printf(" %d thread(s)\n", THREADS);
	}
	if(strcmp(argv[i], "-loops")==0){
	  LOOPS = atoi(argv[++i]);
	  printf(" %d loop(s)\n", LOOPS);
	}
	if(strcmp(argv[i], "-smlimit")==0){
	  SMLIMIT = atoi(argv[++i]);
	  printf(" SMLIMIT %d\n", SMLIMIT);
	}
  }
  return 0;
}

real frand()
{
  //return (real)(rand()%1000) / 1000.0;
  return (real)((double)(rand()%10) / 5.0);
}

int gpu_h2d(void *d, void *h, int elemsize, int count)
{
  cudaMemcpy(d,h,elemsize*count,cudaMemcpyHostToDevice);
  return 0;
}

int gpu_d2h(void *h, void *d, int elemsize, int count)
{
  cudaMemcpy(h,d,elemsize*count,cudaMemcpyDeviceToHost);
  return 0;
}

// 逐次
__global__ void gpu_kernel_1(int size, real *_a, real *_b, real *_c)
{
  int i, j, k;
  for(j=0; j<size; j++){
	for(i=0; i<size; i++){
	  real sum = 0.0f;
	  for(k=0; k<size; k++){
		sum += _b[j*size+k] * _c[k*size+i];
	  }
	  _a[j*size+i] = sum;
	}
  }
}

// 行単位並列化
__global__ void gpu_kernel_2(int size, real *_a, real *_b, real *_c)
{
  int i, j, k;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int jBegin = id;
  int jEnd = size;
  int jStep = gridDim.x*blockDim.x;
  for(j=jBegin; j<jEnd; j+=jStep){
	for(i=0; i<size; i++){
	  real tmp = 0.0;
	  for(k=0; k<size; k++){
		tmp += _b[j*size+k] * _c[k*size+i];
	  }
	  _a[j*size+i] = tmp;
	}
  }
}

// 行単位並列化：blockとthreadを明確に分離
__global__ void gpu_kernel_3(int size, real *_a, real *_b, real *_c)
{
  __shared__ real sm[32];
  int i, j, k;
  for(j=blockIdx.x; j<size; j+=gridDim.x){
	__syncthreads();
	for(i=0; i<size; i++){
	  sm[threadIdx.x] = 0.0;
	  for(k=threadIdx.x; k<size; k+=blockDim.x){
		sm[threadIdx.x] += _b[j*size+k] * _c[k*size+i];
	  }
	  __syncthreads();
	  if(threadIdx.x==0){
		for(k=0; k<32; k++){
		  _a[j*size+i] += sm[k];
		}
	  }
	  __syncthreads();
	}
  }
}

// SM blocking
// SXthreads, Xblocks
__global__ void gpu_kernel_4(int size, real *_a, real *_b, real *_c, real *_dbg)
{
  const int SX = 32;
  __shared__ real smA[SX*SX];
  __shared__ real smB[SX*SX];
  __shared__ real smC[SX*SX];
  int x, y, z;
  int i, j, k, l;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bs = SX;

  for(y=bid*bs; y<size; y+=bs*gridDim.x){
	for(x=0; x<size; x+=bs){
	  for(i=0; i<SX; i++){
		smA[i*SX+tid] = 0.0;
	  }
	  for(z=0; z<size/SX; z++){
		for(i=0; i<SX; i++){
		  smB[i*SX+tid] = _b[(y+i)*size +z*SX+tid];
		  smC[i*SX+tid] = _c[(i+z*SX)*size +x+tid];
		}
		__syncthreads();
		{
		  for(i=0; i<SX; i++){
			//for(j=0; j<SX; j++)
			j = tid;
			{
			  real tmp = 0.0;
			  for(k=0; k<SX; k++){
				tmp += smB[i*SX+k] * smC[k*SX+j];
			  }
			  smA[i*SX+j] += tmp;
			}
		  }
		}
		__syncthreads();
	  }
	  for(i=0; i<SX; i++){
		_a[(y+i)*size+x+tid] = smA[i*SX+tid];
	  }
	  __syncthreads();
	}
  }
}
// SM blocking
template <int SX>
__global__ void gpu_kernel_5(int size, real * __restrict__ _a, real * __restrict__  _b, real * __restrict__  _c)
{
  int x, y, z;
  int k, l;
  int tidX = threadIdx.x;
  int tidY = threadIdx.y;
  int bid = blockIdx.x;
  int bs1 = SX;
  int bs2 = bs1*gridDim.x;

  for(y=bid*bs1; y<size; y+=bs2){
	for(x=0; x<size; x+=bs1){
#if 1
	  real tmp = 0.0;
	  __shared__ real smB[SX*SX];
	  __shared__ real smC[SX*SX];
	  for(z=0; z<size; z+=SX){
		smB[tidY*SX +tidX] = _b[(y+tidY)*size +z+tidX];
		smC[tidY*SX +tidX] = _c[(z+tidY)*size +x+tidX];
		__syncthreads();
#pragma unroll
		for(k=0; k<SX; k++){
		  tmp += smB[tidY*SX+k] * smC[k*SX+tidX];
		}
		__syncthreads();
	  }
#else
	  real tmp = 0.0;
	  __shared__ real smB[SX][SX];
	  __shared__ real smC[SX][SX];
	  for(z=0; z<size; z+=SX){
		smB[tidY][tidX] = _b[(y+tidY)*size +z+tidX];
		smC[tidY][tidX] = _c[(z+tidY)*size +x+tidX];
		__syncthreads();
#pragma unroll
		for(k=0; k<SX; k++){
		  tmp += smB[tidY][k] * smC[k][tidX];
		}
		__syncthreads();
	  }
#endif
	  _a[(y+tidY)*size +x+tidX] = tmp;
	}
  }
}

int gpu_kernel(int kernel, int size, real *_a, real *_b, real *_c, real *_dbg)
{
  switch(kernel){
  default:
  case 0:
	printf("kernel %d is undefined\n", kernel);
	return -1;
	break;
  case 1:
	gpu_kernel_1<<<1,1>>>(size,_a,_b,_c);
	break;
  case 2:
	//gpu_kernel_2<<<240,256>>>(size,_a,_b,_c); // (448/32)*10=240
	gpu_kernel_2<<<624,256>>>(size,_a,_b,_c); // (2496/32)*8=624
	break;
  case 3:
	gpu_kernel_3<<<624,32>>>(size,_a,_b,_c);
	break;
  case 4:
	gpu_kernel_4<<<624,32>>>(size,_a,_b,_c, _dbg);
	break;
  case 5:
	//gpu_kernel_5<<<624,dim3(32,32,1)>>>(size,_a,_b,_c);
	{
	  switch(THREADS){
	  case 32:	gpu_kernel_5<32><<<BLOCKS,dim3(32,32,1)>>>(size,_a,_b,_c);	break;
	  case 24:	gpu_kernel_5<24><<<BLOCKS,dim3(24,24,1)>>>(size,_a,_b,_c);	break;
	  case 16:	gpu_kernel_5<16><<<BLOCKS,dim3(16,16,1)>>>(size,_a,_b,_c);	break;
	  case  8:	gpu_kernel_5< 8><<<BLOCKS,dim3( 8, 8,1)>>>(size,_a,_b,_c);	break;
	  case  4:	gpu_kernel_5< 4><<<BLOCKS,dim3( 4, 4,1)>>>(size,_a,_b,_c);	break;
	  default:	gpu_kernel_5< 2><<<BLOCKS,dim3( 2, 2,1)>>>(size,_a,_b,_c);	break;
	  }
	}
	break;
  case 99:
	{
	  real alpha = 1.0;
	  real beta = 0.0;
#ifdef _DOUBLE
	  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, d_B, SIZE, d_C, size, &beta, d_A, size);
#else
	  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, d_B, SIZE, d_C, size, &beta, d_A, size);
#endif
	}
	break;
  }
  return 0;
}

// ******** ******** ******** ********
// main
// ******** ******** ******** ********
int main(int argc, char** argv)
{
  int i;
  if(checkArgs(argc,argv))return -1;

  printf("size %d\n", SIZE);
  srand(RANDSEED);

  TIME_INITIALIZE();

  //printf("initialize...");
  int t,g;

  g_A = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_B = (real*)malloc(sizeof(real)*SIZE*SIZE);
  g_C = (real*)malloc(sizeof(real)*SIZE*SIZE);
  cudaMalloc((void**)&d_A, sizeof(real)*SIZE*SIZE);
  cudaMalloc((void**)&d_B, sizeof(real)*SIZE*SIZE);
  cudaMalloc((void**)&d_C, sizeof(real)*SIZE*SIZE);
  g_dbg = (real*)malloc(sizeof(real)*1000);
  cudaMalloc((void**)&d_dbg, sizeof(real)*1000);
  for(i=0;i<1000;i++)g_dbg[i] = 0.0;

  for(g=0; g<SIZE; g++){
	for(t=0; t<SIZE; t++){
	  g_A[g*SIZE+t] = 0.0;
	  g_B[g*SIZE+t] = frand();
	  g_C[g*SIZE+t] = frand();
	}
  }
  /*
  printf("done\n");
  printf("====\n");
  for(g=0; g<SIZE; g++){
	for(t=0; t<SIZE; t++){
	  printf(" %.2f", g_B[g*SIZE+t]);
	}
	printf("\n");
  }
  printf("====\n");
  */
  {
	int i, j;
	FILE *F;
	F = fopen("B.txt", "w");
	for(i=0; i<SIZE; i++){
	  for(j=0; j<SIZE; j++){
		fprintf(F, " %.2f", g_B[i*SIZE+j]);
	  }
	  fprintf(F, "\n");
	}
	fclose(F);
  }
  {
	int i, j;
	FILE *F;
	F = fopen("C.txt", "w");
	for(i=0; i<SIZE; i++){
	  for(j=0; j<SIZE; j++){
		fprintf(F, " %.2f", g_C[i*SIZE+j]);
	  }
	  fprintf(F, "\n");
	}
	fclose(F);
  }

  struct timeval tBegin, tEnd;
  struct timezone tz;
  real dSec;
  real dBegin, dEnd;
  int ret;
  if(KERNEL!=99){
	gpu_h2d(d_A, g_A, sizeof(real), SIZE*SIZE);
	gpu_h2d(d_B, g_B, sizeof(real), SIZE*SIZE);
	gpu_h2d(d_C, g_C, sizeof(real), SIZE*SIZE);
	gpu_h2d(d_dbg, g_dbg, sizeof(real), 1000);
  }
  if(KERNEL==99){
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS){
	  printf("CUBLAS initialization failed\n");
	}
	stat = cublasSetMatrix(SIZE,SIZE,sizeof(real),g_A,SIZE,d_A,SIZE);
	if(stat != CUBLAS_STATUS_SUCCESS){
	  printf("CUBLAS SetMatrix A failed\n");
	  return -1;
	}
	stat = cublasSetMatrix(SIZE,SIZE,sizeof(real),g_B,SIZE,d_B,SIZE);
	if(stat != CUBLAS_STATUS_SUCCESS){
	  printf("CUBLAS SetMatrix B failed\n");
	  return -1;
	}
	stat = cublasSetMatrix(SIZE,SIZE,sizeof(real),g_C,SIZE,d_C,SIZE);
	if(stat != CUBLAS_STATUS_SUCCESS){
	  printf("CUBLAS SetMatrix C failed\n");
	  return -1;
	}
	if(SMLIMIT>0){
	  printf("cublasSetSmCountTarget %d\n", SMLIMIT);
	  stat = cublasSetSmCountTarget(handle, SMLIMIT);
	  if(stat != CUBLAS_STATUS_SUCCESS){
		printf("cublasSetSmCountTarget failed\n");
		return -1;
	  }
	}

  }
  gettimeofday(&tBegin, &tz);
  for(i=0;i<LOOPS;i++){DO_KERNEL3(gpu_kernel(KERNEL, SIZE, d_A, d_B, d_C, d_dbg), 1.0e-9*SIZE*SIZE*SIZE*2);}
  if(ret!=0)return -1;
  gettimeofday(&tEnd, &tz);
  if(KERNEL!=99){
	gpu_d2h(g_A, d_A, sizeof(real), SIZE*SIZE);
	gpu_d2h(g_dbg, d_dbg, sizeof(real), 1000);
  }else{
	stat = cublasGetMatrix(SIZE,SIZE,sizeof(real),d_A,SIZE,g_A,SIZE);
	if(stat != CUBLAS_STATUS_SUCCESS){
	  printf("CUBLAS GetMatrix failed\n");
	}
  }
  dBegin= tBegin.tv_sec + (double)tBegin.tv_usec*1.0e-6;
  dEnd= tEnd.tv_sec + (double)tEnd.tv_usec*1.0e-6;
  dSec= dEnd - dBegin;
  printf("performance: size %d^2, %f sec, %f Mflops\n", SIZE, dSec, (double)(LOOPS*2.0*(double)SIZE*(double)SIZE*(double)SIZE)/dSec/1000.0/1000.0);
  TimePrintf_GPU(&gputimeKernel);

  printf("random sampling:");
  for(i=0; i<10; i++){
	int n = rand()%(SIZE*SIZE);
	printf(" %.2f", g_A[n]);
  }
  printf("\n");

  {
	int i, j;
	FILE *F;
	F = fopen("dbg.txt", "w");
	for(i=0; i<1000; i++){
	  fprintf(F, " %.2f", g_dbg[i]);
	  if((i+1)%8==0)fprintf(F, "\n");
	}
	fprintf(F, "\n");
	fclose(F);
  }

  {
	int i, j;
	FILE *F;
	char filename[16];
	snprintf(filename, 16, "A_cuda_k%d.txt", KERNEL);
	F = fopen(filename, "w");
	for(i=0; i<SIZE; i++){
	  for(j=0; j<SIZE; j++){
		fprintf(F, " %.2f", g_A[i*SIZE+j]);
	  }
	  fprintf(F, "\n");
	}
	fclose(F);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  if(KERNEL!=99){
  }else{
	cublasDestroy(handle);
  }
  free(g_A);
  free(g_B);
  free(g_C);

  return 0;
}
