#ifndef __DEVICE_MEMORY_CUH__
#define __DEVICE_MEMORY_CUH__

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T>
class CudaMemory {
protected:
    T * pointer;
    size_t byteSize;
public:
    CudaMemory() {
        this->pointer = nullptr;
        this->byteSize = 0;
    }
    //CudaMemory() : this->pointer(nullptr), this->bytesize(0) {}

/*
    CudaMemory(T const & other) {
        this->mallocByte(other.byteSize);
        this->copyFrom(other.data());
    }

    __host__ __device__ T & operator=(T const & other) {
        this->mallocByte(other.byteSize);
        this->copyFrom(other.data());
    }
    */

    CudaMemory(T&& other) noexcept {
        this->pointer = other.pointer;
        other.pointer = nullptr;
        this->byteSize = other.byteSize;
        other.byteSize = 0;
    }

    __host__ __device__ T & operator=(T&& other) noexcept {
        if (&other != this) {
            this->pointer = other.pointer;
            other.pointer = nullptr;
            this->byteSize = other.byteSize;
            other.byteSize = 0;
        }
        return *this;
    }

    // Should solve this destruct problem (don't use naive RAII)
    ~CudaMemory() noexcept {
        //fprintf(stdout, "destructor! %p, %ld\n", pointer, byteSize);
        //this->free();
    }

    __host__ __device__ T & operator[](size_t const index) {
        return this->pointer[index];
    }

    __host__ __device__ T const & operator[](size_t const index) const {
        return this->pointer[index];
    }

    __host__ __device__ cudaError_t malloc(size_t const count) {
        return this->mallocByte(sizeof(T) * count);
    }

    __host__ __device__ cudaError_t mallocByte(size_t const byte) {
        this->byteSize = byte;
        auto e = cudaMalloc((void**)&this->pointer, this->byteSize);
        return e;
    }

    __host__ cudaError_t zerofill() {
        return cudaMemset(this->pointer, 0, this->byteSize);
    }

    __host__ __device__ cudaError_t zerofill(cudaStream_t const stream) {
        return cudaMemsetAsync(this->pointer, 0, this->byteSize, stream);
    }

    __host__ cudaError_t memset(int const value) {
        return cudaMemset(this->pointer, value, this->byteSize);
    }

    __host__ __device__ cudaError_t memset(int const value, cudaStream_t const stream) {
        return cudaMemsetAsync(this->pointer, value, this->byteSize, stream);
    }

    __host__ cudaError_t copyH2D(T const * const hostSource) {
        return cudaMemcpy(this->pointer, hostSource, this->byteSize, cudaMemcpyHostToDevice);
    }

    __host__ __device__ cudaError_t copyH2D(T const * const hostSource, cudaStream_t const stream) {
        return cudaMemcpyAsync(this->pointer, hostSource, this->byteSize, cudaMemcpyHostToDevice, stream);
    }

    __host__ cudaError_t copyD2H(T * const hostDest) const {
        return cudaMemcpy(hostDest, this->pointer, this->byteSize, cudaMemcpyDeviceToHost);
    }

    __host__ __device__ cudaError_t copyD2H(T * const hostDest, cudaStream_t const stream) const {
        return cudaMemcpyAsync(hostDest, this->pointer, this->byteSize, cudaMemcpyDeviceToHost, stream);
    }

    __host__ cudaError_t copyFrom(T const * const dest) {
        return cudaMemcpy(this->pointer, dest, this->byteSize, cudaMemcpyDefault);
    }

    __host__ __device__ cudaError_t copyFrom(T const * const dest, cudaStream_t const stream) {
        return cudaMemcpyAsync(this->pointer, dest, this->byteSize, cudaMemcpyDefault, stream);
    }

    __host__ cudaError_t copyTo(T * const dest) const {
        return cudaMemcpy(dest, this->pointer, this->byteSize, cudaMemcpyDefault);
    }

    __host__ __device__ cudaError_t copyTo(T * const dest, cudaStream_t const stream) const {
        return cudaMemcpyAsync(dest, this->pointer, this->byteSize, cudaMemcpyDefault, stream);
    }

    __host__ __device__ cudaError_t free() {
        auto result = cudaFree(this->pointer);
        if (result == cudaError_t::cudaSuccess) {
            this->pointer = nullptr;
        }
        return result;
    }

    __host__ __device__ T * data() {
        return this->pointer;
    }

    __host__ __device__ T const * data() const {
        return this->pointer;
    }

    __host__ __device__ size_t byte() const {
        return this->byteSize;
    }

    __host__ __device__ size_t count() const {
        return this->byteSize / sizeof(T);
    }
};

/*
template <typename T>
struct CudaManagedMemory : public CudaMemory<T> {
    __host__ cudaError_t copyH2D(T const * const hostSource) = delete;
    __host__ __device__ cudaError_t copyH2D(T const * const hostSource, cudaStream_t const stream) = delete;
    __host__ cudaError_t copyD2H(T * const hostDest) const = delete;
    __host__ __device__ cudaError_t copyD2H(T * const hostDest, cudaStream_t const stream) const = delete;
    __host__ cudaError_t copyFrom(T const * const dest) = delete;
    __host__ __device__ cudaError_t copyFrom(T const * const dest, cudaStream_t const stream) = delete;
    __host__ cudaError_t copyTo(T * const dest) const = delete;
    __host__ __device__ cudaError_t copyTo(T * const dest, cudaStream_t const stream) const = delete;

    __host__ cudaError_t malloc(size_t const count) {
        this->byteSize = sizeof(T) * count;
        return cudaMallocManaged((void**)&this->pointer, this->byteSize);
    }

    __host__ cudaError_t mallocByte(size_t const byte) {
        this->byteSize = byte;
        return cudaMallocManaged((void**)&this->pointer, this->byteSize);
    }

    __host__ cudaError_t prefetch(int const deviceNumber) const {
        return cudaMemPrefetchAsync(this->pointer, this->byteSize, deviceNumber);
    }

    __host__ cudaError_t prefetch(int const deviceNumber, cudaStream_t & stream) const {
        return cudaMemPrefetchAsync(this->pointer, this->byteSize, deviceNumber, stream);
    }

    __host__ cudaError_t advise(int const deviceNumber, cudaMemoryAdvise adv) const {
        return cudaMemAdvise(this->pointer, this->byteSize, adv, deviceNumber);
    }
};
*/

#endif