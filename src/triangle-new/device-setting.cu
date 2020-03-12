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
    fs::path const & folderPath)
{
    this->load_meta(folderPath);

    auto const & gridWidth = this->mem.graph_meta.info.width.row;

    {
        auto & meta = this->gpu.meta;
        meta.index = gpuIndex;
        cudaSetDevice(meta.index); CUDACHECK();
        cudaGetDeviceProperties(&meta.info, meta.index); CUDACHECK();
    }

    {
        auto & setting = this->gpu.setting;
        setting.stream.resize(stream);
        for (auto & s : setting.stream) {
            cudaStreamCreate(&s); CUDACHECK();
        }
        setting.block = block;
        setting.thread = thread;

    }

    {
        this->mem.stream.resize(this->gpu.setting.stream.size());

        for (auto & s : this->mem.stream) {
            s.bitmap.lv0.alloc(this->gpu.setting.block * (gridWidth >> EXP_BITMAP0)); CUDACHECK();
            s.bitmap.lv1.alloc(this->gpu.setting.block * (gridWidth >> EXP_BITMAP1)); CUDACHECK();

            s.bitmap.lv0.zerofill(); CUDACHECK();
            s.bitmap.lv1.zerofill(); CUDACHECK();

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
                s.lookup.G0.byte()); CUDACHECK();

            cudaMalloc(&s.cub.ptr, s.cub.byte); CUDACHECK();
        }
    }

    this->load_graph(folderPath);
}

void device_setting_t::load_meta(fs::path const & folderPath) {
    auto & m = this->mem.graph_meta;

    m.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));

    if (m.info.count.row != m.info.count.col) {
        printf("Not appropriate grid count! Abort process.\n");
        exit(EXIT_FAILURE);
    }

    if (m.info.width.row != m.info.width.col) {
        printf("Not appropriate grid width! Abort process.\n");
        exit(EXIT_FAILURE);
    }
}

void device_setting_t::load_graph(fs::path const & folderPath) {
    auto const & m = this->mem.graph_meta;

    auto const & gridCount = m.info.count.row;

    auto & g = this->mem.graph;
    g.resize(gridCount);
    for (auto & r : g) {
        r.resize(gridCount);
    }

    for (auto in : m.grid.each) {
        auto const basicString = folderPath.string() + std::string(in.name) + ".";

        auto const pathRow = fs::path(basicString + m.extension.row);
        auto const pathPtr = fs::path(basicString + m.extension.ptr);
        auto const pathCol = fs::path(basicString + m.extension.col);

        if (!(fs::exists(pathRow) && fs::exists(pathPtr) && fs::exists(pathCol))) {
            printf("Not exists: %s\n", in.name.c_str());
            exit(EXIT_FAILURE);
        }

        size_t const rowIndex = in.index.row;
        size_t const colIndex = in.index.col;


        auto loader = [](std::ifstream & f, fs::path const & p, decltype(g.front().front().row) & d){
            f.open(p);

            f.seekg(0, std::ios::end);
            auto const fileSize = f.tellg();
            f.seekg(0, std::ios::beg);

            auto const vertexCount = fileSize / sizeof(vertex_t);
            std::vector<vertex_t> _temp(vertexCount);

            f.read((char*)_temp.data(), fileSize);
            f.close();

            d.alloc(vertexCount); CUDACHECK();
            d.copy_h2d(_temp.data()); CUDACHECK();

            _temp.clear();
        };

        std::ifstream f;
        loader(f, pathRow, g[rowIndex][colIndex].row);
        loader(f, pathPtr, g[rowIndex][colIndex].ptr);
        loader(f, pathCol, g[rowIndex][colIndex].col);
    }
}

device_setting_t::~device_setting_t() {
    cudaSetDevice(this->gpu.meta.index); CUDACHECK();

    for (auto & s : this->gpu.setting.stream) {
        //cudaStreamDestroy(s); CUDACHECK();
    }

    cudaDeviceReset(); CUDACHECK();
}