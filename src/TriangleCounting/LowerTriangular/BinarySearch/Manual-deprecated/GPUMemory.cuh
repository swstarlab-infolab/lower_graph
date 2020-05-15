#ifndef __GPUMemory_cuh__
#define __GPUMemory_cuh__

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "error.h"

template <typename T>
class GPUMemory {
protected:
    int devID;
    std::shared_ptr<T> sharedPointer;
    std::size_t byteSize;

    __host__ auto myMalloc(size_t byteSize) {
        auto myCreator = [this](size_t myByteSize) {
            T * myPointer;
            ThrowCuda(cudaSetDevice(this->devID));
            ThrowCuda(cudaMalloc(&myPointer, myByteSize));
            return myPointer;
        };

        auto myDeleter = [this](void * myPointer) {
            //fprintf(stdout, "[GPU%2d] Try free device memory %p\n", this->devID, myPointer);
            TryCatch(
                ThrowCuda(cudaSetDevice(this->devID));
                ThrowCuda(cudaFree(myPointer));
            );
            //fprintf(stdout, "[GPU%2d] Free device memory %p\n", this->devID, myPointer);
        };

        return std::shared_ptr<T>(myCreator(byteSize), myDeleter);
    }

public:
    GPUMemory() {
        this->byteSize = 0;
        this->sharedPointer = NULL;
    }

    GPUMemory(int _devID, size_t _byteSize) {
        this->devID = _devID;
        this->mallocByte(_byteSize);
    }

    ~GPUMemory() noexcept = default;

    GPUMemory(GPUMemory const & other) = delete;
    GPUMemory(GPUMemory&& other) noexcept = delete;

    __host__ __device__ GPUMemory & operator=(GPUMemory const & other) = delete;
    __host__ __device__ GPUMemory & operator=(GPUMemory&& other) noexcept = delete;

    //__host__ __device__ T & operator[](size_t const index) { return this->sharedPointer.get()[index]; }
    //__host__ __device__ T const & operator[](size_t const index) const { return this->sharedPointer.get()[index]; }

    __host__ void malloc(size_t const count) {
        this->mallocByte(sizeof(T) * count);
    }

    __host__ void mallocByte(size_t const byte) {
        this->byteSize = byte;
        this->sharedPointer = myMalloc(this->byteSize);
    }

    __host__ void setDev(int const _devID) {
        this->devID = _devID;
    }

    __host__ void zerofill() {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemset(this->sharedPointer.get(), 0, this->byteSize));
    }

    __host__ __device__ void zerofill(cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemsetAsync(this->sharedPointer.get(), 0, this->byteSize, stream));
    }

    __host__ void memset(int const value) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemset(this->sharedPointer.get(), value, this->byteSize));
    }

    __host__ __device__ void memset(int const value, cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemsetAsync(this->sharedPointer.get(), value, this->byteSize, stream));
    }

    __host__ void copyH2D(T const * const hostSource) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(this->sharedPointer.get(), hostSource, this->byteSize, cudaMemcpyHostToDevice))
    }

    __host__ __device__ void copyH2D(T const * const hostSource, cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpyAsync(this->sharedPointer.get(), hostSource, this->byteSize, cudaMemcpyHostToDevice, stream));
    }

    __host__ void copyD2H(T * const hostDest) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(hostDest, this->sharedPointer.get(), this->byteSize, cudaMemcpyDeviceToHost));
    }

    __host__ __device__ void copyD2H(T * const hostDest, cudaStream_t const stream) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpyAsync(hostDest, this->sharedPointer.get(), this->byteSize, cudaMemcpyDeviceToHost, stream));
    }

    __host__ void copyFrom(T const * const dest) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(this->sharedPointer.get(), dest, this->byteSize, cudaMemcpyDefault));
    }

    __host__ __device__ void copyFrom(T const * const dest, cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(this->sharedPointer.get(), dest, this->byteSize, cudaMemcpyDefault, stream));
    }

    __host__ void copyTo(T * const dest) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(dest, this->sharedPointer.get(), this->byteSize, cudaMemcpyDefault));
    }

    __host__ __device__ void copyTo(T * const dest, cudaStream_t const stream) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(dest, this->sharedPointer.get(), this->byteSize, cudaMemcpyDefault, stream));
    }

    __host__ T * data() {
        return this->sharedPointer.get();
    }

    __host__ __device__ T const * data() const {
        return this->sharedPointer.get();
    }

    __host__ __device__ size_t byte() const {
        return this->byteSize;
    }

    __host__ __device__ size_t count() const {
        return this->byteSize / sizeof(T);
    }
};

#endif