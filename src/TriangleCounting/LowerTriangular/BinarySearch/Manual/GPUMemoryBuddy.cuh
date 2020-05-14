#ifndef __GPUMemoryBuddy_cuh__
#define __GPUMemoryBuddy_cuh__

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <BuddySystem/BuddySystem.h>

#include "error.h"

template <typename T>
class GPUMemoryBuddy : public std::enable_shared_from_this<GPUMemoryBuddy> {
protected:
    T * pointer;
    hashed_buddy_system* buddy;
    std::size_t byteSize;
    int devID;

public:
    GPUMemoryBuddy() {
        this->byteSize = 0;
        this->sharedPointer = NULL;
    }

    GPUMemoryBuddy(int const _devID, size_t const _byteSize) {
        this->devID = _devID;
        if (!this->mallocByte(_byteSize)) {
            Throw("GPU Memory Buddy Constructor Malloc failed!");
        }
    }

    ~GPUMemoryBuddy() noexcept {
        this->free();
    }

    GPUMemoryBuddy(GPUMemoryBuddy const & other) = delete;
    GPUMemoryBuddy(GPUMemoryBuddy&& other) noexcept = delete;

    __host__ __device__ GPUMemoryBuddy & operator=(GPUMemoryBuddy const & other) = delete;
    __host__ __device__ GPUMemoryBuddy & operator=(GPUMemoryBuddy&& other) noexcept = delete;

    __host__ bool malloc(size_t const count) {
        return this->mallocByte(sizeof(T) * count);
    }

    __host__ bool mallocByte(size_t const byte) {
        this->byteSize = byte;
        this->pointer = this->buddy->allocate(this->byteSize);
        return (this->pointer != nullptr);
        //this->sharedPointer = myMalloc(this->byteSize);
    }

    __host__ void free(size_t const count) {
        this->buddy->deallocate(this->pointer);
        this->pointer = nullptr;
    }

    __host__ void setDev(int const _devID) {
        this->devID = _devID;
    }

    __host__ void setBuddy(hashed_buddy_system const & b) {
        this->buddy = &b;
    }

    __host__ void zerofill() {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemset(this->pointer, 0, this->byteSize));
    }

    __host__ __device__ void zerofill(cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemsetAsync(this->pointer, 0, this->byteSize, stream));
    }

    __host__ void memset(int const value) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemset(this->pointer, value, this->byteSize));
    }

    __host__ __device__ void memset(int const value, cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemsetAsync(this->pointer, value, this->byteSize, stream));
    }
    j
    __host__ void copyH2D(T const * const hostSource) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(this->pointer, hostSource, this->byteSize, cudaMemcpyHostToDevice))
    }

    __host__ __device__ void copyH2D(T const * const hostSource, cudaStream_t const stream) {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpyAsync(this->pointer, hostSource, this->byteSize, cudaMemcpyHostToDevice, stream));
    }

    __host__ void copyD2H(T * const hostDest) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpy(hostDest, this->pointer, this->byteSize, cudaMemcpyDeviceToHost));
    }

    __host__ __device__ void copyD2H(T * const hostDest, cudaStream_t const stream) const {
        ThrowCuda(cudaSetDevice(this->devID));
        ThrowCuda(cudaMemcpyAsync(hostDest, this->pointer, this->byteSize, cudaMemcpyDeviceToHost, stream));
    }

    __host__ T * data() {
        return this->data();
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

    __host__ std::shared<GPUMemoryBuddy> get_shared_ptr() {
        return shared_from_this();
    }
};

#endif