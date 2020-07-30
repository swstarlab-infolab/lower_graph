#include "device-setting.cuh"

#include <GridCSR/GridCSR.h>

#include <cuda_runtime.h>
#include <fstream>
#include <cmath>

void device_setting_t::init(
    uint32_t const gpuIndex,
    uint32_t const stream,
    uint32_t const block,
    uint32_t const thread,
    GridCSR::FS::path const & folderPath)
{
    auto & meta = this->gpu.meta;
    meta.index = gpuIndex;
    cudaSetDevice(meta.index); CUDACHECK();
    cudaDeviceSynchronize();
    cudaDeviceReset(); CUDACHECK();
    cudaDeviceSynchronize();

    this->load_meta(folderPath);
    this->load_graph(folderPath);

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
        s.count.alloc(1); CUDACHECK();
        s.count.zerofill(); CUDACHECK();
    }
}

void device_setting_t::load_meta(GridCSR::FS::path const & folderPath) {
    auto & m = this->mem.graph_meta;

    m.Load(GridCSR::FS::path(folderPath.string() + "meta.json"));

    if (m.info.count.row != m.info.count.col) {
        printf("Not appropriate grid count! Abort process.\n");
        exit(EXIT_FAILURE);
    }

    if (m.info.width.row != m.info.width.col) {
        printf("Not appropriate grid width! Abort process.\n");
        exit(EXIT_FAILURE);
    }
}

#include <tbb/parallel_for_each.h>

void device_setting_t::load_graph(GridCSR::FS::path const & folderPath) {
    auto const & m = this->mem.graph_meta;
    cudaSetDevice(this->gpu.meta.index); CUDACHECK();

    auto const & gridCount = m.info.count.row;

    auto & g = this->mem.graph;
    g.resize(gridCount);
    for (auto & r : g) {
        r.resize(gridCount);
    }

    auto loader = [](std::ifstream & f, GridCSR::FS::path const & p, decltype(g.front().front().row) & d){
        f.open(p);

        f.seekg(0, std::ios::end);
        auto const fileSize = f.tellg();
        f.seekg(0, std::ios::beg);

        auto const vertexCount = fileSize / sizeof(GridCSR::Vertex);
        std::vector<GridCSR::Vertex> _temp(vertexCount);

        f.read((char*)_temp.data(), fileSize);
        f.close();

        d.alloc(vertexCount); CUDACHECK();
        d.copy_h2d(_temp.data()); CUDACHECK();

        _temp.clear();
    };

    // if decltype is directely used on the site, the weird '__T1' error happened
    using mGridEachFront = decltype(m.grid.each.front());

    tbb::parallel_for_each(m.grid.each.begin(), m.grid.each.end(),
        [this, &folderPath, &m, &loader](mGridEachFront & in){
            auto const basicString = folderPath.string() + std::string(in.name) + ".";

            auto const pathRow = GridCSR::FS::path(basicString + m.extension.row);
            auto const pathPtr = GridCSR::FS::path(basicString + m.extension.ptr);
            auto const pathCol = GridCSR::FS::path(basicString + m.extension.col);

            if (!(GridCSR::FS::exists(pathRow) && GridCSR::FS::exists(pathPtr) && GridCSR::FS::exists(pathCol))) {
                printf("Not exists: %s\n", in.name.c_str());
                exit(EXIT_FAILURE);
            }

            size_t const rowIndex = in.index.row;
            size_t const colIndex = in.index.col;

            std::ifstream f;
            cudaSetDevice(this->gpu.meta.index); CUDACHECK();
            loader(f, pathRow, this->mem.graph[rowIndex][colIndex].row);
            loader(f, pathPtr, this->mem.graph[rowIndex][colIndex].ptr);
            loader(f, pathCol, this->mem.graph[rowIndex][colIndex].col);
        });
}

device_setting_t::~device_setting_t() {
    cudaSetDevice(this->gpu.meta.index); CUDACHECK();

    for (auto & s : this->gpu.setting.stream) {
        cudaStreamDestroy(s); CUDACHECK();
    }

    cudaDeviceReset(); CUDACHECK();
}