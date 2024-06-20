// -*- c++ -*-
__global__ void gpukernel(int N, float *C, float *A, float *B)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  C[id] += A[id] * B[id];
}

extern "C" void gpukernel_wrapper(int N, float *C, float *A, float *B)
{
  dim3 grids;
  dim3 blocks;
  grids = dim3(2, 1, 1);
  blocks = dim3(64, 1 ,1);
  gpukernel<<<grids,blocks>>>(N, C, A, B);
}
