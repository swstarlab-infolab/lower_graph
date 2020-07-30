#include "main.cuh"

#include <fstream>
#include <string>
#include <sstream>
#include <GridCSR/GridCSR.h>

static size_t filesize(GridCSR::FS::path const & path) {
    std::ifstream f;
    f.open(path);
    f.seekg(0, std::ios::end);
    auto const fileSize = f.tellg();
    f.seekg(0, std::ios::beg);
    f.close();
    return fileSize;
}

static std::string SI(size_t byte) {
    size_t constexpr KiB = size_t(1) << 10;
    size_t constexpr MiB = size_t(1) << 20;
    size_t constexpr GiB = size_t(1) << 30;
    size_t constexpr TiB = size_t(1) << 40;
    size_t constexpr PiB = size_t(1) << 50;
    size_t constexpr EiB = size_t(1) << 60;
    char buf[11];
    if (byte < KiB) {
        snprintf(buf, sizeof(buf), "%ldB", byte);
    } else if (KiB <= byte && byte < MiB) {
        snprintf(buf, sizeof(buf), "%.3lfKiB", double(byte) / double(KiB));
    } else if (MiB <= byte && byte < GiB) {
        snprintf(buf, sizeof(buf), "%.3lfMiB", double(byte) / double(MiB));
    } else if (GiB <= byte && byte < TiB) {
        snprintf(buf, sizeof(buf), "%.3lfGiB", double(byte) / double(GiB));
    } else if (TiB <= byte && byte < PiB) {
        snprintf(buf, sizeof(buf), "%.3lfTiB", double(byte) / double(TiB));
    } else if (PiB <= byte && byte < EiB) {
        snprintf(buf, sizeof(buf), "%.3lfPiB", double(byte) / double(PiB));
    } else {
        snprintf(buf, sizeof(buf), "%.3lfEiB", double(byte) / double(EiB));
    }

    std::ostringstream oss;
    oss << buf;
    std::string result = oss.str();
    return result;
}

static cudaError_t loadFileToGPU(
    GridCSR::FS::path const & path,
    CudaMemory<GridCSR::Vertex> & mem)
{
    auto fileSize = filesize(path);

    std::vector<GridCSR::Vertex> temp(fileSize / sizeof(GridCSR::Vertex));

    std::ifstream f;
    f.open(path, std::ios::binary);
    f.read((char*)temp.data(), fileSize);
    printf("%s, %ld\n", path.c_str(), fileSize / sizeof(GridCSR::Vertex));
    f.close();

    mem.mallocByte(fileSize);
    auto e = cudaGetLastError(); if (e) { return e; }
    mem.copyH2D(temp.data()); CUDACHECK();
    temp.clear();

    return cudaError::cudaSuccess;
}

int selectGPU(int const devices) {
    size_t maxfree = 0;
    size_t maxgpu = 0;
    for (int i = 0; i < devices; i++) {
        size_t free, total;
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free, &total);
        if (maxfree < free) {
            maxfree = free;
            maxgpu = i;
        }
    }
    return maxgpu;
}

void managed_t::init(GridCSR::FS::path const & folderPath) {
    GridCSR::MetaData meta;
    meta.Load(GridCSR::FS::path(folderPath.string() + "meta.json"));

    std::ifstream f;
    this->graph.resize(meta.info.count.row);
    for (auto & g : this->graph) {
        g.resize(meta.info.count.col);
    }

    int devices = -1;
    cudaGetDeviceCount(&devices);

    for (auto i = 0; i < meta.grid.each.size(); i++) {
        auto const baseString = folderPath.string() + std::string(meta.grid.each[i].name) + ".";

        auto const pathRow = GridCSR::FS::path(baseString + meta.extension.row);
        auto const pathPtr = GridCSR::FS::path(baseString + meta.extension.ptr);
        auto const pathCol = GridCSR::FS::path(baseString + meta.extension.col);

        if (!(GridCSR::FS::exists(pathRow) && GridCSR::FS::exists(pathPtr) && GridCSR::FS::exists(pathCol))) {
            printf("Not exists: %s\n", meta.grid.each[i].name.c_str());
            exit(EXIT_FAILURE);
        }

        size_t const rowIndex = meta.grid.each[i].index.row;
        size_t const colIndex = meta.grid.each[i].index.col;

        int gpuID = selectGPU(devices);
        printf("GPU: %d\n", gpuID);

        cudaSetDevice(gpuID);

        while (true) {
            auto e = loadFileToGPU(pathRow, this->graph[rowIndex][colIndex].row);
            if (e == cudaErrorMemoryAllocation) {
                exit(-1);
            } else if (e == cudaSuccess) {
                break;
            } else {
                fprintf(stderr, "error, %s\n", cudaGetErrorString(e)); exit(-1);
            }
        }

        while (true) {
            auto e = loadFileToGPU(pathPtr, this->graph[rowIndex][colIndex].ptr);
            if (e == cudaErrorMemoryAllocation) {
                exit(-1);
            } else if (e == cudaSuccess) {
                break;
            } else {
                fprintf(stderr, "error, %s\n", cudaGetErrorString(e)); exit(-1);
            }
        }

        while (true) {
            auto e = loadFileToGPU(pathCol, this->graph[rowIndex][colIndex].col);
            if (e == cudaErrorMemoryAllocation) {
                exit(-1);
            } else if (e == cudaSuccess) {
                break;
            } else {
                fprintf(stderr, "error, %s\n", cudaGetErrorString(e)); exit(-1);
            }
        } 
    }

    for (int i = 0; i < devices; i++) {
        size_t free, total;
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &total);
        fprintf(stdout, "GPU %d : %s / %s\n", i, SI(free).c_str(), SI(total).c_str());
    }

    cudaDeviceSynchronize();
}