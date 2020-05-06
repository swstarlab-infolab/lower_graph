#include <string>

#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <queue>
#include <future>
#include <chrono>

#include "../cudaMemory.cuh"
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
protected:
    struct _stream {
        cudaStream_t stream;
        cudaEvent_t event;
        struct {
            CudaMemory<Count> edge, selfloop;
        } memory;
    };

    struct _global {
    };

public:
    int deviceID;

    std::vector<_stream> stream;
    _global global;

    std::vector<Grid> worklist;

    void init(int const _deviceID, int const _streams) {
        this->deviceID = _deviceID;
        this->stream.resize(_streams);

        cudaSetDevice(this->deviceID); CUDACHECK();
        cudaDeviceReset(); CUDACHECK();

        for (auto & s : this->stream) {
            cudaStreamCreate(&s.stream); CUDACHECK();
            cudaEventCreate(&s.event); CUDACHECK();
            s.memory.edge.malloc(1); CUDACHECK();
            s.memory.edge.zerofill(); CUDACHECK();
            s.memory.selfloop.malloc(1); CUDACHECK();
            s.memory.selfloop.zerofill(); CUDACHECK();
        }
    }

    ~Device() {
        cudaSetDevice(this->deviceID);
        /*
        for (auto & s : this->stream) {
            cudaEventDestroy(s.event); CUDACHECK();
            cudaStreamDestroy(s.stream); CUDACHECK();
        }
        */
        cudaDeviceReset();
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

struct Result {
    Count edge, selfloop;
};

__global__
void countEdge(Grid const g, Count * count) {
    Count mycount = 32;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    //uint32_t ts = gridDim.x * blockDim.x;

    for (size_t i = blockIdx.x; i < g.ptr.count() - 1; i += gridDim.x) {
        for (auto j = g.ptr[i] + threadIdx.x; j < g.ptr[i+1]; j += blockDim.x) {
            mycount++;
        }
    }


    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(count, mycount); }
}

__global__
void countSelfloop(Grid const g, Count * count) {
    Count mycount = 0;

    for (size_t i = blockIdx.x; i < g.ptr.count() - 1; i += gridDim.x) {
        for (auto j = g.ptr[i] + threadIdx.x; j < g.ptr[i+1]; j += blockDim.x) {
            if (g.row[i] == g.col[j]) {
                mycount++;
            }
        }
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(count, mycount); }
}

Result eachGPURoutine(Managed const & man, Device & dev, int const blocks, int const threads) {
    if (dev.worklist.size() == 0) {
        return Result{0,};
    }


    auto & nowWork = dev.worklist.front();

    printf("%d %d %d\n", nowWork.row.byte(), nowWork.ptr.byte(), nowWork.col.byte());

    cudaSetDevice(dev.deviceID);
    countEdge<<<blocks, threads>>>(nowWork, dev.stream[0].memory.edge.data());
    countSelfloop<<<blocks, threads>>>(nowWork, dev.stream[0].memory.selfloop.data());

/*

    printf("Start Prefetch\n");
    nowWork.row.prefetch(dev.deviceID, dev.stream[0].stream);
    nowWork.ptr.prefetch(dev.deviceID, dev.stream[0].stream);
    nowWork.col.prefetch(dev.deviceID, dev.stream[0].stream);

    printf("Prefetch event record\n");
    cudaEventRecord(dev.stream[0].event, dev.stream[1].stream);

    for (size_t i = 0; i < dev.worklist.size(); i++) {
        auto & nowWork = dev.worklist[i];

        cudaEventSynchronize(dev.stream[0].event);
        cudaEventSynchronize(dev.stream[1].event);
        printf("event sync record\n");

        countEdge<<<blocks, threads, 0, dev.stream[0].stream>>>(nowWork, dev.stream[0].memory.edge.data());
        countSelfloop<<<blocks, threads, 0, dev.stream[0].stream>>>(nowWork, dev.stream[0].memory.selfloop.data());

        cudaDeviceSynchronize();

        cudaEventRecord(dev.stream[0].event, dev.stream[0].stream);
        printf("kernel event record\n");

        if (i < dev.worklist.size() - 1) {
            cudaStreamSynchronize(dev.stream[1].stream);
            auto & nextWork = dev.worklist[i+1];

            nextWork.row.prefetch(dev.deviceID, dev.stream[1].stream);
            nextWork.ptr.prefetch(dev.deviceID, dev.stream[1].stream);
            nextWork.col.prefetch(dev.deviceID, dev.stream[1].stream);
            cudaEventRecord(dev.stream[1].event, dev.stream[1].stream);
            printf("prefetch event record if\n");
        }

        std::swap(dev.stream[0].stream, dev.stream[1].stream);
        std::swap(dev.stream[0].event, dev.stream[1].event);
        printf("swap\n");
    }
    */

    cudaDeviceSynchronize();
    printf("sync all\n");

    std::vector<Result> result(dev.stream.size());

    for (int i = 0; i < result.size(); i++) {
        dev.stream[i].memory.edge.copyD2H(&result[i].edge);
        dev.stream[i].memory.selfloop.copyD2H(&result[i].selfloop);
    }

    for (int i = 1; i < result.size(); i++) {
        result.front().edge += result[i].edge;
        result.front().selfloop += result[i].selfloop;
    }

    return result.front();
};

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const blockCount = strtol(argv[2], nullptr, 10);
    auto const threadCount = strtol(argv[3], nullptr, 10);

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount); CUDACHECK();
    //printf("Found %d devices\n", deviceCount);

    auto p = [](int src, int dst){
        cudaSetDevice(src); CUDACHECK();
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, src, dst); CUDACHECK();
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(dst, 0); CUDACHECK();
            //printf("Peer Access Enabled: GPU%d -> GPU%d\n", src, dst);
        }
    };


    for (auto i = 0; i < deviceCount; i++) {
        for (auto j = 0; j < i; j++) {
            p(i, j);
            p(j, i);
        }
    }

    Devices devices(deviceCount);

    for (auto i = 0; i < devices.size(); i++){
        devices[i].init(i, 2);
    }

    Managed managed;
    managed.init(pathFolder);

    cudaDeviceSynchronize();

    int gpu = 0;
    for (auto & row : managed.graph) {
        for (auto & grid : row) {
            if (grid.row.count() != 0) {
                devices[gpu].worklist.push_back(std::move(grid));
            }
        }
        gpu = (gpu == (devices.size()-1)) ? 0 : gpu + 1;
    }

    std::vector<std::future<Result>> futures(devices.size());

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < futures.size(); i++) {
        futures[i] = std::async(std::launch::async, eachGPURoutine, std::ref(managed), std::ref(devices[i]), blockCount, threadCount);
    }

    Result hResult = {0,};
    for (int i = 0; i < devices.size(); i++) {
        auto dResult = futures[i].get();
        hResult.edge += dResult.edge;
        hResult.selfloop += dResult.selfloop;
    }

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;

    printf("edges:%lld\n", hResult.edge);
    printf("loops:%lld\n", hResult.selfloop);
    printf("elapsed:%lf\n", elapsed.count());

    return 0;
}