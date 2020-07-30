#ifndef __DEVICE_MEMORY_CUH__
#define __DEVICE_MEMORY_CUH__

#include <cuda_runtime.h>
#include <cstdint>

template <typename T>
struct device_memory_t {
    T * ptr;
    uint64_t count;

    __host__ __device__ T & operator[](uint64_t const index) {
        return this->ptr[index];
    }

    __host__ __device__ T const & operator[](uint64_t const index) const {
        return this->ptr[index];
    }

    __host__ void alloc(uint64_t const _count) {
        this->count = _count;
        cudaMalloc((void**)&this->ptr, sizeof(T) * this->count);
    }

    __host__ void zerofill() {
        cudaMemset(this->ptr, 0x00, sizeof(T) * this->count);
    }

    __host__ void copy_h2d(T const * host_src) {
        cudaMemcpy(this->ptr, host_src, sizeof(T) * this->count, cudaMemcpyHostToDevice);
    }

    __host__ void copy_d2h(T * host_dst) {
        cudaMemcpy(host_dst, this->ptr, sizeof(T) * this->count, cudaMemcpyDeviceToHost);
    }

    __host__ void copy_h2d_async(T const * host_src, cudaStream_t & stream) {
        cudaMemcpyAsync(this->ptr, host_src, sizeof(T) * this->count, cudaMemcpyHostToDevice, stream);
    }

    __host__ void copy_d2h_async(T * host_dst, cudaStream_t & stream) {
        cudaMemcpyAsync(host_dst, this->ptr, sizeof(T) * this->count, cudaMemcpyDeviceToHost, stream);
    }

    __host__ void free() {
        if (this->ptr) {
            cudaFree(this->ptr);
        }
    }

    __host__ __device__ uint64_t byte() {
        return sizeof(T) * this->count;
    }
};

template <typename T>
struct unified_memory_t : public device_memory_t<T> {
    __host__ void alloc(uint64_t const _count) {
        this->count = _count;
        cudaMallocManaged((void**)&this->ptr, sizeof(T) * this->count);
    }
};

#endif