#ifndef __main_cuh__
#define __main_cuh__

#include "GPUMemory.cuh"
#include "GPUMemoryBuddy.cuh"
#include "CPUMemory.h"

#include <vector>
#include <mutex>
#include <map>

#include <GridCSR/GridCSR.h>
#include <BuddySystem/BuddySystem.h>

struct GridPosition {
    size_t row, col;
};

struct Grid {
    std::mutex mtx;

    int location = 0;

    struct Mem {
        GridCSR::Vertex * ptr;
        size_t byte;

        __host__ __device__ GridCSR::Vertex & operator[](size_t const index) {
            return this->ptr[index];
        }

        __host__ __device__ GridCSR::Vertex const & operator[](size_t const index) const {
            return this->ptr[index];
        }


    } row, ptr, col;
};

using MemTable = std::map<GridPosition, Grid>;
using MemTables = std::vector<MemTable>;


#endif