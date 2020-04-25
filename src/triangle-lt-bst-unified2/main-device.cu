#include "main.cuh"

#include <cub/device/device_scan.cuh>

void device_t::init(int const _deviceID, int const _streams, int const _gridWidth) {
    this->deviceID = _deviceID;
    this->stream.resize(_streams);
    this->streamMemory.resize(_streams);

    cudaSetDevice(this->deviceID); CUDACHECK();
    cudaDeviceReset(); CUDACHECK();
    for (auto & s : this->stream) {
        cudaStreamCreate(&s); CUDACHECK();
    }

    for (auto & s : this->streamMemory) {
        s.lookup.G0.malloc(_gridWidth + 1); CUDACHECK();
        s.lookup.G2.malloc(_gridWidth + 1); CUDACHECK();
        s.lookup.temp.malloc(_gridWidth + 1); CUDACHECK();

        s.lookup.G0.zerofill(); CUDACHECK();
        s.lookup.G2.zerofill(); CUDACHECK();
        s.lookup.temp.zerofill(); CUDACHECK();

        s.count.malloc(1); CUDACHECK();
        s.count.zerofill(); CUDACHECK();

        size_t byte = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            byte,
            s.lookup.temp.data(),
            s.lookup.G0.data(),
            s.lookup.G0.count()); CUDACHECK();

        s.cub.mallocByte(byte);
    }
}

device_t::~device_t() {
    cudaSetDevice(this->deviceID); CUDACHECK();
    for (auto & s : this->stream) {
        cudaStreamDestroy(s);
    }
}