#include "function.h"

#include <fstream>

std::string SIUnit(size_t byte) {
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

static auto filesize(std::string const & path) {
    std::ifstream f;
    f.open(path);
    f.seekg(0, std::ios::end);
    auto const fileSize = f.tellg();
    f.seekg(0, std::ios::beg);
    f.close();
    return fileSize;
}

void loadGrid(GlobalContext) {

}

void loadFileToGPU(
    GridCSR::FS::path const & path,
    GPUMemory<GridCSR::Vertex> & mem)
{
    auto fileSize = filesize(path);

    std::vector<GridCSR::Vertex> temp(fileSize / sizeof(GridCSR::Vertex));

    std::ifstream f;
    f.open(path, std::ios::binary);
    f.read((char*)temp.data(), fileSize);
    printf("%s, %ld\n", path.c_str(), fileSize / sizeof(GridCSR::Vertex));
    f.close();

    mem.mallocByte(fileSize);
    mem.copyH2D(temp.data());
    temp.clear();
}
/*
#include "main.h"
#include "cudaError.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <fstream>

auto cudaFindMaxMemorySize() {
    size_t freeMem, totalMem, realFreeMem = 0;
    void * myPointer;

    cudaMemGetInfo(&freeMem, &totalMem);
    for (auto i = freeMem; i >= 0; i-= (1024L * 1024L * 16L)) {
        cudaMalloc(&myPointer, i);
        if (auto e = cudaGetLastError(); e == cudaError::cudaSuccess) {
            realFreeMem = i;
            cudaFree(myPointer);
            break;
        }
        cudaFree(myPointer);
    }

    return realFreeMem;
}

auto cudaMallocSmartPtr(size_t byteSize) {
    auto myCreator = [](size_t myByteSize) {
        void * myPointer;
        cudaThrow(cudaMalloc(&myPointer, myByteSize));
        return myPointer;
    };

    auto myDeleter = [](void * myPointer) {
        cudaThrow(cudaFree(myPointer));
    };

    return std::shared_ptr<void>(myCreator(byteSize), myDeleter);
}


void copyBlockDev2GPU(int const devID, Grid const & in, ) {
    cudaSetDevice(devID);
    cudaMemcpy()
}

void copyMem2GPU(Grid const & in{
    cudaSetDevice(devID);
    cudaMemcpy()
}

#include <map>

using GridTable = std::map<GridPosition, Grid>;

bool isGridOnGPU(GridTable const & gpuTable) {
    return true;
}
bool isGridOnCPU(GridTable const & cpuTable) {
    return true;
}
bool isGridOnSSD(GridTable const & ssdTable) {
    return true;
}
 
void loadGrid(
    GridTable const & cpuTable,
    GridTable const & gpuTable,
    Grid const & target)
{
    if (isGridOnGPU(gpuTable, target)) {

    } else {
        if (isGridOnCPU(cpuTable, target)) {
            allocGPU();
        } else {
            if (isGridOnSSD(ssdTable, target)) {
                allocCPU();
                allocGPU();
            } else {
                throw "PANIC";
            }
        }
    }
};

#define location_ssd00 0x000001
#define location_cpu00 0x000002
#define location_gpu00 0x000004
#define location_gpu01 0x000008
#define location_gpu02 0x000010
#define location_gpu03 0x000020
#define location_gpu04 0x000040
#define location_gpu05 0x000080
#define location_gpu06 0x000100
#define location_gpu07 0x000200
#define location_gpu08 0x000400
#define location_gpu09 0x000800
#define location_gpu10 0x001000
#define location_gpu11 0x002000
#define location_gpu12 0x004000
#define location_gpu13 0x008000
#define location_gpu14 0x010000
#define location_gpu15 0x020000
#define location_gpu16 0x040000
#define location_gpu17 0x080000


*/