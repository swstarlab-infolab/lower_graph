#ifndef __CPUMemory_h__
#define __CPUMemory_h__

#include "error.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template <typename T>
class CPUMemory {
protected:
    std::shared_ptr<T> sharedPointer;
    std::size_t byteSize;

    auto myMalloc(size_t byteSize) {
        auto myCreator = [](size_t myByteSize) {
            T * myPointer;
            cudaHostAlloc(&myPointer, myByteSize, cudaHostAllocPortable); // should write cudaHostAllocPortable for Multi-GPU environment!
            //myPointer = (T *)std::malloc(myByteSize);
            return myPointer;
        };

        auto myDeleter = [](void * myPointer) {
            //fprintf(stdout, "[CPU  ] Try free host memory %p\n", myPointer);
            cudaFreeHost(myPointer);
            //free(myPointer);
            //fprintf(stdout, "[CPU  ] Free host memory %p\n", myPointer);
        };

        return std::shared_ptr<T>(myCreator(byteSize), myDeleter);
    }

public:
    CPUMemory() {
        this->byteSize = 0;
        this->sharedPointer = NULL;
    }

    CPUMemory(size_t _byteSize) {
        this->mallocByte(_byteSize);
    }

    ~CPUMemory() noexcept = default;

    CPUMemory(CPUMemory const & other) = delete;
    CPUMemory(CPUMemory&& other) noexcept = delete;
    CPUMemory & operator=(CPUMemory const & other) = delete;
    CPUMemory & operator=(CPUMemory&& other) noexcept = delete;

    //T & operator[](size_t const index) { return this->sharedPointer.get()[index]; }
    //T const & operator[](size_t const index) const { return this->sharedPointer.get()[index]; }

    void malloc(size_t const count) {
        this->mallocByte(sizeof(T) * count);
    }

    void mallocByte(size_t const byte) {
        this->byteSize = byte;
        this->sharedPointer = myMalloc(this->byteSize);
    }

    void zerofill() {
        memset(this->sharedPointer.get(), 0, this->byteSize);
    }

    void memset(int const value) {
        memset(this->sharedPointer.get(), value, this->byteSize);
    }

    T * data() {
        return this->sharedPointer.get();
    }

    T const * data() const {
        return this->sharedPointer.get();
    }

    size_t byte() const {
        return this->byteSize;
    }

    size_t count() const {
        return this->byteSize / sizeof(T);
    }
};

#endif