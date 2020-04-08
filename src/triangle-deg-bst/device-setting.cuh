#ifndef __DEVICE_SETTING_CUH__
#define __DEVICE_SETTING_CUH__

#include "device-memory.cuh"

#include <vector>
#include <cstdint>

#include "../common.h"
#include "../meta.h"

#define CUDACHECK() \
        do { auto e = cudaGetLastError(); if (e) { printf("%s:%d, %s(%d), %s\n", __FILE__, __LINE__, cudaGetErrorName(e), e , cudaGetErrorString(e)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while (false)

#define EXP_BITMAP0 12
#define EXP_BITMAP1 5

/*
#ifndef CUDA_BLOCKS
#define CUDA_BLOCKS 160
#endif

#ifndef CUDA_THREADS 
#define CUDA_THREADS 1024
#endif

#ifndef CUDA_STREAMS 
#define CUDA_STREAMS 4
#endif
*/


using bitmap_t = uint32_t;
using count_t = unsigned long long;
using lookup_t = uint32_t;

struct device_setting_t {
    struct {
        struct {
            uint32_t index;
            cudaDeviceProp info;
        } meta;

        struct {
            std::vector<cudaStream_t> stream;
            uint32_t block, thread;
        } setting;
    } gpu;

    struct {
    private:
        struct __mem_per_stream_t {
            device_memory_t<count_t> count;
        };

        struct __mem_global_t {
            device_memory_t<vertex_t> row, ptr, col;
        };

    public:
        std::vector<__mem_per_stream_t> stream;
        std::vector<std::vector<__mem_global_t>> graph;
        meta_t graph_meta;
    } mem;

    void init(
        uint32_t const gpuIndex,
        uint32_t const stream,
        uint32_t const block,
        uint32_t const thread,
        fs::path const & folderPath);

    ~device_setting_t();

private:
    void load_meta(fs::path const & folderPath);
    void load_graph(fs::path const & folderPath);
};

#endif