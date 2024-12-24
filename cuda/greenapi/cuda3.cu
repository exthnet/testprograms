#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cassert>
#include <cstdint>

#define CUDA_RT(call)                                                   \
    do {                                                                \
        cudaError_t _err = (call);                                      \
        if ( cudaSuccess != _err ) {                                    \
            fprintf(stderr, "CUDA error in file '%s' at line %i: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return _err;                                                \
        } } while (0)


#define CUDA_DRV(call)                                                  \
    do {                                                                \
        CUresult _status = (call);                                      \
        if ( CUDA_SUCCESS != _status) {                                 \
            fprintf(stderr, "CUDA error in file '%s' at line %i: %i\n", \
                    __FILE__, __LINE__, _status);                       \
            return _status;                                             \
        } } while (0)

__device__ int temp_result;

extern "C" __global__ void timewaster(const int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    volatile float result = 0;
    for (int i = 0; i < num_iterations; i++) {
        result += sqrtf(i + idx) + sinf(idx * i);
    }
    if (idx == 0) {
        temp_result = (int)result;
    }
}

__global__ void kernel()
{
    temp_result += threadIdx.x * blockIdx.x;
}

int main()
{
    CUgreenCtx gctx[2];
    CUdevResourceDesc desc[2];
    CUdevResource input;
    CUdevResource resources[2];
    CUstream streamA;
    CUstream streamB;

    unsigned int nbGroups = 1;
    unsigned int minCount = 0;

    // Initialize device 0
    CUDA_RT(cudaInitDevice(0, 0, 0));
    // Preload
    timewaster<<<1, 512>>>(1);
    kernel<<<1, 512>>>();

    // Query input SMs
    CUDA_DRV(cuDeviceGetDevResource((CUdevice)0, &input, CU_DEV_RESOURCE_TYPE_SM));
    // We want 3/4 the device for our green context
    minCount = (unsigned int)((float)input.sm.smCount * 0.75f);

    // Split my resources
    CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCount));

    // Create a descriptor/ctx for the main 3/4 partion
    CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
    CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)0, CU_GREEN_CTX_DEFAULT_STREAM));
    // ... and one for the remainder 1/4 partition
    CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
    CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)0, CU_GREEN_CTX_DEFAULT_STREAM));
    // Create streams that we will use from here on out
    CUDA_DRV(cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
    CUDA_DRV(cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

    timewaster<<<1000, 512, 0, (cudaStream_t)streamA>>>(1000);
    kernel<<<1, 512, 0, (cudaStream_t)streamB>>>();

    CUDA_RT(cudaStreamSynchronize((cudaStream_t)streamA));
    CUDA_RT(cudaStreamSynchronize((cudaStream_t)streamB));

    return (0);
}
