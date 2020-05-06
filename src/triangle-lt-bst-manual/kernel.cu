#include "main.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common.h"

__global__
void genLookupTemp(grid_t const g, CudaMemory<lookup_t> lookup_temp) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.row.count(); i += gridDim.x * blockDim.x) {
        lookup_temp[g.row[i]] = g.ptr[i+1] - g.ptr[i];
    }
}

__global__
void resetLookupTemp(grid_t const g, CudaMemory<lookup_t> lookup_temp) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.row.count(); i += gridDim.x * blockDim.x) {
        lookup_temp[g.row[i]] = 0;
    }
}

__device__
static uint32_t ulog2floor(uint32_t x) {
    uint32_t r, q;
    r = (x > 0xFFFF) << 4; x >>= r;
    q = (x > 0xFF  ) << 3; x >>= q; r |= q;
    q = (x > 0xF   ) << 2; x >>= q; r |= q;
    q = (x > 0x3   ) << 1; x >>= q; r |= q;
                                   
    return (r | (x >> 1));
}

__device__
static void intersection(
    vertex_t const * Arr,
    uint32_t const ArrLen,
    vertex_t const candidate,
    count_t * count)
{
    //auto const maxLevel = uint32_t(ceil(log2(ArrLen + 1))) - 1;
    // ceil(log2(a)) == floor(log2(a-1))+1
    auto const maxLevel = ulog2floor(ArrLen);

    int now = (ArrLen - 1) >> 1;

    for (uint32_t level = 0; level <= maxLevel; level++) {
        auto const movement = 1 << (maxLevel - level - 1);

        if (now < 0) {
            now += movement;
        } else if (ArrLen <= now) {
            now -= movement;
        } else {
            if (Arr[now] < candidate) {
                now += movement;
            } else if (candidate < Arr[now]) {
                now -= movement;
            } else {
                (*count)++;
                break;
            }
        }
    }
}

/*
__global__ static void kernel(
    lookup_t const * lookupG0, vertex_t const * G0Col,
    vertex_t const * G1Row, vertex_t const * G1Ptr, vertex_t const * G1Col,
    lookup_t const * lookupG2, vertex_t const * G2Col,
    count_t const G1RowSize,
    count_t const gridWidth,
    count_t * count)
    */
__global__
void kernel(
        grid_t const g0, grid_t const g1, grid_t const g2,
        CudaMemory<lookup_t> const lookup0,
        CudaMemory<lookup_t> const lookup2,
        CudaMemory<count_t> count)
{
    count_t mycount = 0;

    __shared__ int SHARED[1024];

    for (uint32_t g1row_iter = blockIdx.x; g1row_iter < g1.row.count(); g1row_iter += gridDim.x) {

        // This makes huge difference!!!
        // Without "Existing Row" information: loop all 2^24 and check it all
        // With "Existing Row" information: extremely faster than without-version
        auto const g1row = g1.row[g1row_iter];

        if (lookup2[g1row] == lookup2[g1row + 1]) { continue; }

        auto const g1col_idx_s = g1.ptr[g1row_iter], g1col_idx_e = g1.ptr[g1row_iter+1];

        // variable for binary tree intersection
        auto const g1col_length = g1col_idx_e - g1col_idx_s;

        auto const g2col_s = lookup2[g1row], g2col_e = lookup2[g1row+1];

        for (uint32_t g2col_idx = g2col_s; g2col_idx < g2col_e; g2col_idx += blockDim.x) {
            SHARED[threadIdx.x] = (g2col_idx + threadIdx.x < g2col_e) ? (int)g2.col[g2col_idx + threadIdx.x] : -1;

            __syncthreads();

            for (uint32_t s = 0; s < blockDim.x; s++) {
                int const g2col = SHARED[s];
                if (g2col == -1) { break; }
                if (lookup0[g2col] == lookup0[g2col + 1]) { continue; }

                auto const g0col_idx_s = lookup0[g2col], g0col_idx_e = lookup0[g2col+1];

                // variable for binary tree intersection
                auto const g0col_length = g0col_idx_e - g0col_idx_s;

                if (g1col_length >= g0col_length) {
                    for (uint32_t g0col_idx = g0col_idx_s + threadIdx.x; g0col_idx < g0col_idx_e; g0col_idx += blockDim.x) {
                        intersection(&g1.col[g1col_idx_s], g1col_length, g0.col[g0col_idx], &mycount);
                    }
                } else {
                    for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e; g1col_idx += blockDim.x) {
                        intersection(&g0.col[g0col_idx_s], g0col_length, g1.col[g1col_idx], &mycount);
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(count.data(), mycount); }
}