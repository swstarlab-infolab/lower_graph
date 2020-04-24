#include "device-setting.cuh"

#include "../meta.h"

#include <cuda_runtime.h>
#include <fstream>
#include <cmath>

#include <cub/device/device_scan.cuh>

void device_setting_t::init(
    uint32_t const gpuIndex,
    uint32_t const stream,
    uint32_t const block,
    uint32_t const thread,
    uint32_t const gridWidth)
{
    auto & meta = this->gpu.meta;
    meta.index = gpuIndex;

    cudaSetDevice(meta.index); CUDACHECK();
    cudaDeviceSynchronize();

    cudaDeviceReset(); CUDACHECK();
    cudaDeviceSynchronize();

    cudaSetDevice(meta.index); CUDACHECK();
    cudaGetDeviceProperties(&meta.info, meta.index); CUDACHECK();

    auto & setting = this->gpu.setting;
    setting.stream.resize(stream);
    for (auto & s : setting.stream) {
        cudaStreamCreate(&s); CUDACHECK();
    }
    setting.block = block;
    setting.thread = thread;

    this->mem.stream.resize(setting.stream.size());

    for (auto & s : this->mem.stream) {
        s.lookup.G0.alloc(gridWidth + 1); CUDACHECK();
        s.lookup.G2.alloc(gridWidth + 1); CUDACHECK();
        s.lookup.temp.alloc(gridWidth + 1); CUDACHECK();

        s.lookup.G0.zerofill(); CUDACHECK();
        s.lookup.G2.zerofill(); CUDACHECK();
        s.lookup.temp.zerofill(); CUDACHECK();

        s.count.alloc(1); CUDACHECK();
        s.count.zerofill(); CUDACHECK();

        cub::DeviceScan::ExclusiveSum(
            s.cub.ptr,
            s.cub.byte,
            s.lookup.temp.ptr,
            s.lookup.G0.ptr,
            s.lookup.G0.count); CUDACHECK();

        cudaMalloc(&s.cub.ptr, s.cub.byte); CUDACHECK();
    }
}

device_setting_t::~device_setting_t() {
    cudaSetDevice(this->gpu.meta.index); CUDACHECK();

    for (auto & s : this->gpu.setting.stream) {
        cudaStreamDestroy(s); CUDACHECK();
    }

    cudaDeviceReset(); CUDACHECK();
}

void unified_setting_t::load_graph(fs::path const & folderPath) {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);

    auto peerEnable = [](int src, int dst){
        cudaSetDevice(src);
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, src, dst);
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(dst, 0);
            printf("Peer Access Enabled: GPU%d -> GPU%d\n", src, dst);
        }
    };

    for (auto i = 0; i < devCount; i++) {
        for (auto j = 0; j < i; j++) {
            peerEnable(i, j);
            peerEnable(j, i);
        }
    }

    auto & m = this->meta;
    auto & g = this->graph;

    m.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));

    if (m.info.count.row != m.info.count.col) {
        printf("Not appropriate grid count! Abort process.\n");
        exit(EXIT_FAILURE);
    }

    if (m.info.width.row != m.info.width.col) {
        printf("Not appropriate grid width! Abort process.\n");
        exit(EXIT_FAILURE);
    }

    auto const & gridCount = m.info.count.row;

    g.resize(gridCount);
    for (auto & gr : g) {
        gr.resize(gridCount);
    }

    auto loader = [](std::ifstream & f, fs::path const & p, decltype(g.front().front().row) & d){
        f.open(p);

        f.seekg(0, std::ios::end);
        auto const fileSize = f.tellg();
        f.seekg(0, std::ios::beg);

        auto const vertexCount = fileSize / sizeof(vertex_t);
        d.alloc(vertexCount); CUDACHECK();
        f.read((char*)d.ptr, fileSize);
        f.close();
    };

    // if decltype is directely used on the site, the weird '__T1' error happened
    using mGridEachFront = decltype(m.grid.each.front());

    for (auto i = 0; i < m.grid.each.size(); i++) {
        auto const basicString = folderPath.string() + std::string(m.grid.each[i].name) + ".";

        auto const pathRow = fs::path(basicString + m.extension.row);
        auto const pathPtr = fs::path(basicString + m.extension.ptr);
        auto const pathCol = fs::path(basicString + m.extension.col);

        if (!(fs::exists(pathRow) && fs::exists(pathPtr) && fs::exists(pathCol))) {
            printf("Not exists: %s\n", m.grid.each[i].name.c_str());
            exit(EXIT_FAILURE);
        }

        size_t const rowIndex = m.grid.each[i].index.row;
        size_t const colIndex = m.grid.each[i].index.col;

        std::ifstream f;

        loader(f, pathRow, this->graph[rowIndex][colIndex].row);
        loader(f, pathPtr, this->graph[rowIndex][colIndex].ptr);
        loader(f, pathCol, this->graph[rowIndex][colIndex].col);
    }
}

vertex_t getGridWidth(fs::path const & folderPath) {
    meta_t m;
    m.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));
    return m.info.width.row;
}