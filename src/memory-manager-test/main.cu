#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cstdint>

#include <cstdio>
#define CUDACHECK() \
        do { auto e = cudaGetLastError(); if (e) { printf("%s:%d, %s(%d), %s\n", __FILE__, __LINE__, cudaGetErrorName(e), e , cudaGetErrorString(e)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while (false)


using count_t = unsigned long long;

__global__
void kernel(int * pointer, size_t size, count_t * out) {
    count_t mycount = 0;
    int TID = blockDim.x * blockIdx.x + threadIdx.x;
    int TS = gridDim.x * blockDim.x;

    for (int i = TID; i < size; i+=TS) {
        mycount++;
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(out, mycount); }
}

int main() {
    int * a = nullptr;
    count_t * b = nullptr;
    size_t tile_size = (128 * 1024 * 1024) / sizeof(*a);
    size_t num_tiles = 1000;

    cudaMallocManaged(&a, tile_size * num_tiles * sizeof(*a)); CUDACHECK();

    cudaSetDevice(0);

    cudaMalloc(&b, sizeof(*b));
    cudaMemset(b, 0, sizeof(*b));

    cudaStream_t s[2];
    cudaStreamCreate(&s[0]); CUDACHECK();
    cudaStreamCreate(&s[1]); CUDACHECK();

    cudaEvent_t e[2];
    cudaEventCreate(&e[0]); CUDACHECK();
    cudaEventCreate(&e[1]); CUDACHECK();

    cudaMemPrefetchAsync(&a[0], tile_size * sizeof(*a), 0, s[1]); CUDACHECK();
    cudaEventRecord(e[0], s[1]);  CUDACHECK();

    for (int i = 0; i < num_tiles; i++) { 
        cudaEventSynchronize(e[0]); CUDACHECK();
        cudaEventSynchronize(e[1]); CUDACHECK();

        kernel<<<160, 1024, 0, s[0]>>>(&a[tile_size * i], tile_size, b); 
        cudaEventRecord(e[0], s[0]); CUDACHECK(); 

        if (i < num_tiles-1) {
            cudaMemPrefetchAsync(&a[tile_size * (i+1)], tile_size * sizeof(*a), 0, s[1]); CUDACHECK();
            cudaEventRecord(e[1], s[1]); CUDACHECK();
        } 

        std::swap(s[0], s[1]);
        std::swap(e[0], e[1]);
    }

    count_t hb = 0;
    cudaMemcpy(&hb, b, sizeof(*b), cudaMemcpyDeviceToHost);

    printf("hb: %lld\n", hb);

    cudaDeviceReset(); CUDACHECK();

    return 0;
}