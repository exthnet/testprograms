// -*- c++ -*-
#include <stdio.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv)
{
  cudaError_t err;
  cudaDeviceProp dev;
  int nDevices;
  int i;

  err = cudaGetDeviceCount(&nDevices);
  /*
  if(err!=cudaSuccess){
	printf("cudaGetDeviceCount failed\n");
	return err;
  }
  */
  printf("%d GPU(s) found\n", nDevices);
  for(i=0; i<nDevices; i++){
	err = cudaGetDeviceProperties(&dev, i);
	printf("%d: %s %d\n", i, dev.name, dev.uuid);
  }
  return 0;
}

