#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define bufSize 16

__global__ void kernel(float * src, float * dst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ts = gridDim.x * blockDim.x;

    for (int i = tid; i < bufSize; i += ts) {
        dst[i] = src[i] * 2.0f;
    }
}

int main() {
    int GPUs;
    cudaGetDeviceCount(&GPUs);

    if (GPUs < 2) {
        return 0;
    }

    std::vector<cudaDeviceProp> prop(GPUs);
    for (int i = 0; i < GPUs; i++) {
        cudaGetDeviceProperties(&prop[i], i);
    }

    for (int i = 0; i < GPUs; i++) {
        for (int j = 0; j < GPUs; j++) {
            if (i == j) { continue; }
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess) {
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    std::vector<float*> g(GPUs);
    std::vector<float*> h(1);

    for (int i = 0; i < g.size(); i++) {
        cudaSetDevice(i);
        cudaMalloc(&g[i], sizeof(float) * bufSize);
    }

    for (int i = 0; i < h.size(); i++) {
        cudaSetDevice(i);
        cudaMallocHost(&h[i], sizeof(float) * bufSize);
    }

    for (int i = 0; i < bufSize; i++) {
        h[0][i] = (float)i;
    }

    for (int i = 0; i < bufSize; i++) {
        printf("%f ", h[0][i]);
    }
    printf("\n");

    cudaMemcpy(g[0], h[0], sizeof(float) * bufSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < GPUs; i++) {
        if (i != GPUs - 1) {
            kernel<<<1, 32>>>(g[i], g[i+1]);
        } else {
            kernel<<<1, 32>>>(g[i], g[0]);
        }
    }

    cudaMemcpy(h[0], g[0], sizeof(float) * bufSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < bufSize; i++) {
        printf("%f ", h[0][i]);
    }
    printf("\n");

    cudaDeviceReset();
    return 0; 
 }