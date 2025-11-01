
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA kernel! (block %d, thread %d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA visible devices in sysroot-linked build (runtime check happens on Jetson): %d\n", deviceCount);

    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
