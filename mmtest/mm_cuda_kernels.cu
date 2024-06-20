/* -*- C++ -*- */
#include <stdio.h>
#include <cuda.h>

extern "C"
int gpu_h2d(void *d, void *h, int elemsize, int count)
{
  cudaMemcpy(d,h,elemsize*count,cudaMemcpyHostToDevice);
  return 0;
}

extern "C"
int gpu_d2h(void *h, void *d, int elemsize, int count)
{
  cudaMemcpy(h,d,elemsize*count,cudaMemcpyDeviceToHost);
  return 0;
}

// 逐次
__global__ void gpu_kernel_1(int size, double *_a, double *_b, double *_c)
{
  int i, j, k;
  for(j=0; j<size; j++){
	for(i=0; i<size; i++){
	  double sum = 0.0f;
	  for(k=0; k<size; k++){
		sum += _b[j*size+k] * _c[k*size+i];
	  }
	  _a[j*size+i] = sum;
	}
  }
}

// 行単位並列化
__global__ void gpu_kernel_2(int size, double *_a, double *_b, double *_c)
{
  int i, j, k;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int jBegin = id;
  int jEnd = size;
  int jStep = gridDim.x*blockDim.x;
  for(j=jBegin; j<jEnd; j+=jStep){
	for(i=0; i<size; i++){
	  double sum = 0.0f;
	  for(k=0; k<size; k++){
		sum += _b[j*size+k] * _c[k*size+i];
	  }
	  _a[j*size+i] = sum;
	}
  }
}

// SM blocking
// 32threads, Xblocks
__global__ void gpu_kernel_3(int size, double *_a, double *_b, double *_c)
{
  __shared__ double smA[32*32];
  __shared__ double smB[32*32];
  __shared__ double smC[32*32];
  int x, y;
  int i, j, k, l;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bs = 32;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  int jBegin = id;
  int jEnd = size;
  int jStep = gridDim.x*blockDim.x;
  for(y=bid*bs; y<size; y+=bs){
	for(x=0; x<size; x+=bs){

	  for(i=0; i<32; i++){
		smA[i*32+tid] = 0.0;
		smB[i*32+tid] = _b[(y+i)*32+x+tid];
		smC[tid*32+i] = _c[(y+i)*32+x+tid];
	  }
	  if(tid==0){
		for(i=0; i<32; i++){
		  for(j=0; j<32; j++){
			for(k=0; k<32; k++){
			  smA[i*32+j] += smB[i*32+k] * smC[i*32+k];
			}
		  }
		}
	  }
	  for(i=0; i<32; i++){
		_a[(i+1)*32+x+tid] = smA[i*32+tid];
	  }
	}
  }
}

extern "C"
int gpu_kernel(int kernel, int size, double *_a, double *_b, double *_c)
{
  switch(kernel){
  case 0:
	DO_KERNEL(gpu_kernel_1<<<1,1>>>(size,_a,_b,_c));
	break;
  case 1:
	//DO_KERNEL(gpu_kernel_2<<<240,256>>>(size,_a,_b,_c)); // (448/32)*10
	DO_KERNEL(gpu_kernel_2<<<624,256>>>(size,_a,_b,_c)); // (2496/32)*8
	break;
  case 2:
	DO_KERNEL(gpu_kernel_3<<<624,32>>>(size,_a,_b,_c));
	break;
  default:
	printf("kernel %d is undefined\n", kernel);
	break;
  }
  return 0;
}
