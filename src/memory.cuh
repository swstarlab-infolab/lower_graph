#ifndef __DEVICE_MEMORY_CUH__
#define __DEVICE_MEMORY_CUH__

#include <cuda_runtime.h>
#include <cstdint>

template <typename T>
class CudaMemory {
protected:
    T * pointer;
    size_t byteSize;
public:
    __host__ __device__ T & operator[](size_t const index) {
        return this->pointer[index];
    }

    __host__ __device__ T const & operator[](size_t const index) const {
        return this->pointer[index];
    }

    __host__ __device__ cudaError_t malloc(size_t const count) {
        this->byteSize = sizeof(T) * count;
        return cudaMalloc((cudaError_t**)&this->pointer, this->byteSize);
    }

    __host__ __device__ cudaError_t mallocByte(size_t const byte) {
        this->byteSize = byte;
        return cudaMalloc((cudaError_t**)&this->pointer, this->byteSize);
    }

    __host__ cudaError_t zerofill() {
        return cudaMemset(this->pointer, 0x00, this->byteSize);
    }

    __host__ __device__ cudaError_t zerofill(cudaStream_t const stream) {
        return cudaMemsetAsync(this->pointer, 0x00, this->byteSize, stream);
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

template <typename T>
struct CudaManagedMemory : public CudaMemory<T> {
    __host__ cudaError_t copyH2D(T const * const hostSource) = delete;
    __host__ __device__ cudaError_t copyH2D(T const * const hostSource, cudaStream_t const stream) = delete;
    __host__ cudaError_t copyD2H(T * const hostDest) const = delete;
    __host__ __device__ cudaError_t copyD2H(T * const hostDest, cudaStream_t const stream) const = delete;

    __host__ cudaError_t malloc(size_t const count) {
        this->byteSize = sizeof(T) * count;
        return cudaMallocManaged((void**)&this->pointer, this->byteSize);
    }

    __host__ cudaError_t mallocByte(size_t const byte) {
        this->byteSize = byte;
        return cudaMallocManaged((void**)&this->pointer, this->byteSize);
    }

    __host__ cudaError_t prefetchAsync(int deviceNumber, cudaStream_t & stream = 0) {
        return cudaMemPrefetchAsync(this->pointer, this->byteSize, deviceNumber, stream);
    }
};

#endif