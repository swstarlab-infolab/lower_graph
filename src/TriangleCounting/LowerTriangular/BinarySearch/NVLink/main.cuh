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
struct grid_t;
struct managed_t;
struct device_t;

// Types
using lookup_t = uint32_t;
using count_t = unsigned long long;
using graph_t = std::vector<std::vector<grid_t>>;
using devices_t = std::vector<device_t>;

struct grid_t {
    CudaMemory<GridCSR::Vertex> row, ptr, col;
};

struct managed_t {
    graph_t graph;
    void init(GridCSR::FS::path const & folderPath);
};

struct device_t {
    struct StreamMemory {
        struct {
            CudaMemory<lookup_t> G0, G2, temp;
        } lookup;

        CudaMemory<char> cub;
        CudaMemory<count_t> count;
    };

    int deviceID;

    std::vector<cudaStream_t> stream;
    std::vector<StreamMemory> streamMemory;

    void init(int const _deviceID, int const _streams, int const _gridWidth);
    ~device_t();
};

__global__
void genLookupTemp(grid_t const g, CudaMemory<lookup_t> lookup_temp);

__global__
void resetLookupTemp(grid_t const g, CudaMemory<lookup_t> lookup_temp);

__global__
void kernel(
        grid_t const g0, grid_t const g1, grid_t const g2,
        CudaMemory<lookup_t> const lookup0,
        CudaMemory<lookup_t> const lookup2,
        CudaMemory<count_t> count);

#endif