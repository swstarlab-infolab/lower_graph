#include "GPUManagerContext.h"

void GPUManagerContext::init() {
    ThrowCuda(cudaSetDevice(this->gpuID));

    size_t gridWidth = 1L << 24;

    this->mem.lookup.G0.setDev(this->gpuID);
    this->mem.lookup.G2.setDev(this->gpuID);
    this->mem.lookup.Gt.setDev(this->gpuID);
    this->mem.count.setDev(this->gpuID);
    this->mem.cub.setDev(this->gpuID);

    fprintf(stdout, "[GPU%2d] Try allocate local memory\n", this->gpuID);
    this->mem.lookup.G0.malloc(gridWidth + 1);
    this->mem.lookup.G2.malloc(gridWidth + 1);
    this->mem.lookup.Gt.malloc(gridWidth + 1);
    this->mem.count.malloc(1);

    size_t byte = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        byte,
        this->mem.lookup.Gt.data(),
        this->mem.lookup.G0.data(),
        this->mem.lookup.G0.count());

    this->mem.cub.mallocByte(byte);

    fprintf(stdout, "[GPU%2d] Try zerofill local memory\n", this->gpuID);
    this->mem.lookup.G0.zerofill();
    this->mem.lookup.G2.zerofill();
    this->mem.lookup.Gt.zerofill();
    this->mem.count.zerofill();
    this->mem.cub.zerofill();
    
    //auto memSize = cudaFindMaxMemorySize();
    auto memSize = 1024L * 1024L * 1024L * 30L;
    fprintf(stdout, "[GPU%2d] Try allocate GPU buffer memory: %s bytes\n", this->gpuID, SIUnit(memSize).c_str());
    this->buffer.setDev(this->gpuID);
    this->buffer.mallocByte(memSize);
    this->buddy.init(memrgn_t{ this->buffer.data(), this->buffer.byte() }, buddyAlign, buddyMinCof);
    fprintf(stdout, "[GPU%2d] init success\n", this->gpuID);
}