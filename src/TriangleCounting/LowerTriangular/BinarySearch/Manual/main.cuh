#ifndef __MAIN_CUH__
#define __MAIN_CUH__

#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <GridCSR/CUDA/memory.cuh>
#include <GridCSR/GridCSR.h>

#define CUDACHECK() \
        do { auto e = cudaGetLastError(); if (e) { printf("%s:%d, %s(%d), %s\n", __FILE__, __LINE__, cudaGetErrorName(e), e , cudaGetErrorString(e)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while (false)

// Forward declaration
struct Grid;
struct Grids;
struct DeviceMemory;

// Types
using LookupType = uint32_t;
using Lookup = CudaMemory<LookupType>;

using CountType = unsigned long long;
using Count = CudaMemory<CountType>;

struct Grid {
    int location; // where is grid?
    CudaMemory<GridCSR::Vertex> row, ptr, col;
    void loadToGPU(int const destGPUID) {
        switch (location) {
        case -1:
            if () {
                cudaMemcpy()
            } else {

            }
            // Its on CPU
            break;
        case 0: case 1: case 2: case 3:
            cudaSetDevice(location);
            // Its on GPU
            break;
        }
    }

    void free() {
        this->state = GridState::onMemory;
        this->row.free();
        this->ptr.free();
        this->col.free();
    }
};

struct Graph {
    std::vector<std::vector<Grid>> grids;
    void init(GridCSR::FS::path const & folderPath);
};

struct DeviceMemory {
    struct StreamMemory {
        struct {
            CudaMemory<Lookup> G0, G2, temp;
        } lookup;

        CudaMemory<char> cub;

        CudaMemory<Count> count;
    };

    int deviceID;

    std::vector<cudaStream_t> stream;
    std::vector<StreamMemory> streamMemory;

    void init(int const _deviceID, int const _streams, int const _gridWidth);

    ~DeviceMemory();
};

__global__
void genLookupTemp(Grid const g, Lookup lookup_temp);

__global__
void resetLookupTemp(Grid const g, Lookup lookup_temp);

__global__
void kernel(
        Grid const g0, Grid const g1, Grid const g2,
        Lookup const lookup0,
        Lookup const lookup2,
        Count count);

#endif