#ifndef __DEVICE_SETTING_CUH__
#define __DEVICE_SETTING_CUH__

#include "device-memory.cuh"

#include <vector>
#include <cstdint>

#include "../common.h"
#include "../meta.h"

#define CUDACHECK() \
        do { auto e = cudaGetLastError(); if (e) { printf("%s:%d, %s(%d), %s\n", __FILE__, __LINE__, cudaGetErrorName(e), e , cudaGetErrorString(e)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while (false)

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
            struct {
                device_memory_t<lookup_t> G0, G2, temp;
            } lookup;

            struct {
                void * ptr;
                uint64_t byte;
            } cub;

            device_memory_t<count_t> count;
        };

    public:
        std::vector<__mem_per_stream_t> stream;
    } mem;

    void init(
        uint32_t const gpuIndex,
        uint32_t const stream,
        uint32_t const block,
        uint32_t const thread,
        uint32_t const gridWidth);

    ~device_setting_t();
};

struct unified_setting_t {
private:
    struct __mem_global_t {
        unified_memory_t<vertex_t> row, ptr, col;
    };
public:
    meta_t meta;
    std::vector<std::vector<__mem_global_t>> graph;

    void load_graph(fs::path const & folderPath);
};

vertex_t getGridWidth(fs::path const & folderPath);

#endif