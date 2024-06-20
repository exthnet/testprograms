#ifndef _MM_CUDA_KERNELS_H
#define _MM_CUDA_KERNELS_H

int gpu_h2d(void *d, void *h, int elemsize, int count);
int gpu_d2h(void *h, void *d, int elemsize, int count);
int gpu_kernel(int kernel, int size, double *_a, double *_b, double *_c);

#endif
