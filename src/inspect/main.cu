/*
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <math.h>
#include <cstring>
 
__global__ void add(float *x, float *y, float *z) {
    *z = *x + *y;
}
 
int main() {
    float *x, *y;
    float *dz[4];
    float hz[4] = {0,};

    cudaMallocManaged(&x, sizeof(float));
    cudaMallocManaged(&y, sizeof(float));

    *x = 1.0f; *y = 2.0f;

    cudaStream_t str[4];

    for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&str[i]);
        cudaMalloc(&dz[i], sizeof(float));
        cudaMemset(&dz[i], 0, sizeof(float));
    }

    // initialize x and y arrays on the host
    for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
        cudaMemPrefetchAsync(x, sizeof(float), i);
        cudaMemPrefetchAsync(y, sizeof(float), i);
        add<<<1, 1, 0, str[i]>>>(x, y, dz[i]);
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(&hz[i], dz[i], sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < 4; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(str[i]);
    }

    for (int i = 0; i < 4; i++) {
        printf("%f ", hz[i]);
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(str[i]);
        cudaFree(dz[i]);
    }

    return 0;
}
*/
#include <string>

#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../memory.cuh"
#include "../common.h"
#include "../meta.h"

#define CUDACHECK() \
        do { auto e = cudaGetLastError(); if (e) { printf("%s:%d, %s(%d), %s\n", __FILE__, __LINE__, cudaGetErrorName(e), e , cudaGetErrorString(e)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while (false)

// Forward declaration
struct Grid;
struct Device;
struct Managed;

// Types
using Vertex = uint32_t;
using Count = unsigned long long;
using Graph = std::vector<std::vector<Grid>>;
using Devices = std::vector<Device>;

//
struct Grid {
    CudaManagedMemory<Vertex> row, ptr, col;
};

// Only in GPU
struct Device {
    struct StreamMemory {
        CudaMemory<Count> count;
    };

    struct GlobalMemory {
    };

    int deviceID;

    std::vector<cudaStream_t> stream;
    std::vector<StreamMemory> streamMemory;
    GlobalMemory globalMemory;

    void init(int const _deviceID, int const _streams) {
        this->deviceID = _deviceID;
        this->stream.resize(_streams);
        this->streamMemory.resize(_streams);

        cudaSetDevice(this->deviceID); CUDACHECK();
        cudaDeviceReset(); CUDACHECK();
        for (auto & s : this->stream) {
            cudaStreamCreate(&s); CUDACHECK();
        }

        for (auto & smem : this->streamMemory) {
            smem.count.malloc(1); CUDACHECK();
            smem.count.zerofill(); CUDACHECK();
        }
    }

    ~Device() {
        cudaSetDevice(this->deviceID); CUDACHECK();
        for (auto & s : this->stream) {
            cudaStreamDestroy(s);
        }
    }
};


struct Managed {
    Graph graph;

    void init(fs::path const & folderPath) {
        meta_t meta;
        meta.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));

        std::ifstream f;
        this->graph.resize(meta.info.count.row);
        for (auto & g : this->graph) {
            g.resize(meta.info.count.row);
        }

        auto loader = [](std::ifstream & f, fs::path const & path, CudaManagedMemory<Vertex> & mem){
            f.open(path);
            f.seekg(0, std::ios::end);
            auto const fileSize = f.tellg();
            f.seekg(0, std::ios::beg);
            mem.mallocByte(fileSize); CUDACHECK();
            f.read((char*)mem.data(), fileSize);
            f.close();
        };

        for (auto i = 0; i < meta.grid.each.size(); i++) {
            auto const baseString = folderPath.string() + std::string(meta.grid.each[i].name) + ".";

            auto const pathRow = fs::path(baseString + meta.extension.row);
            auto const pathPtr = fs::path(baseString + meta.extension.ptr);
            auto const pathCol = fs::path(baseString + meta.extension.col);

            if (!(fs::exists(pathRow) && fs::exists(pathPtr) && fs::exists(pathCol))) {
                printf("Not exists: %s\n", meta.grid.each[i].name.c_str());
                exit(EXIT_FAILURE);
            }

            size_t const rowIndex = meta.grid.each[i].index.row;
            size_t const colIndex = meta.grid.each[i].index.col;

            std::ifstream f;
            loader(f, pathRow, this->graph[rowIndex][colIndex].row);
            loader(f, pathPtr, this->graph[rowIndex][colIndex].ptr);
            loader(f, pathCol, this->graph[rowIndex][colIndex].col);
        }
    }
};

__global__
void edgeCount(Grid const g, Count * count) {
    Count mycount = 0;

    //uint32_t ts = gridDim.x * blockDim.x;
    //uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = blockIdx.x; i < g.row.count(); i += gridDim.x) {
        for (auto j = g.ptr[i] + threadIdx.x; j < g.ptr[i+1]; j += blockDim.x) {
            mycount++;
        }
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(count, mycount); }
}

int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <streams> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount); CUDACHECK();
    printf("Found %d devices\n", deviceCount);

    auto p = [](int src, int dst){
        cudaSetDevice(src); CUDACHECK();
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, src, dst); CUDACHECK();
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(dst, 0); CUDACHECK();
            printf("Peer Access Enabled: GPU%d -> GPU%d\n", src, dst);
        }
    };

    for (auto i = 0; i < deviceCount; i++) {
        for (auto j = 0; j < i; j++) {
            p(i, j);
            p(j, i);
        }
    }

    Devices devices(deviceCount);

    for (auto i = 0; i < deviceCount; i++){
        devices[i].init(i, 1);
        printf("Set GPU%d Memory\n", i);
    }

    Managed managed;
    managed.init(pathFolder);
    printf("Set Unified Memory\n");

    cudaDeviceSynchronize();

    int gpu = 0;
    for (auto & row : managed.graph) {
        for (auto & grid : row) {
            auto & count = devices[gpu].streamMemory[0].count;
            auto & stream = devices[gpu].stream[0];

            cudaSetDevice(gpu); CUDACHECK();
            //grid.row.prefetchAsync(gpu, devices[gpu].stream[0]); CUDACHECK();
            //grid.ptr.prefetchAsync(gpu, devices[gpu].stream[0]); CUDACHECK();
            //grid.col.prefetchAsync(gpu, devices[gpu].stream[0]); CUDACHECK();
            edgeCount<<<blockCount, threadCount, 0, stream>>>(grid, count.data());

            gpu = (gpu == (devices.size()-1)) ? 0 : gpu + 1;
        }
    }

    for (int i = 0; i < devices.size(); i++) {
        cudaSetDevice(i); CUDACHECK();
        cudaDeviceSynchronize(); CUDACHECK();
    }

    std::vector<Count> hostResult(deviceCount);
    for (int i = 0; i < devices.size(); i++) {
        cudaSetDevice(i); CUDACHECK();
        for (int j = 0; j < devices[i].stream.size(); j++) {
            devices[i].streamMemory[j].count.copyD2H(&hostResult[i], devices[i].stream[j]); CUDACHECK();
        }
    }

    for (int i = 0; i < devices.size(); i++) {
        cudaSetDevice(i); CUDACHECK();
        cudaDeviceSynchronize(); CUDACHECK();
    }

    for (int i = 1; i < hostResult.size(); i++) {
        hostResult[0] += hostResult[i];
    }

    printf("hCount: %lld\n", hostResult.front());

    return 0;
}