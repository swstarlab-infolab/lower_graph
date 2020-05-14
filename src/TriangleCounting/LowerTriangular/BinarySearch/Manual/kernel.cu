#include "main.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GridCSR/GridCSR.h>
#include <GridCSR/CUDA/Kernel.cuh>

__global__
void genLookupTemp(Grid const g, GPUMemory<Lookup> lookupTemp) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.row.count(); i += gridDim.x * blockDim.x) {
        lookupTemp[g.row[i]] = g.ptr[i+1] - g.ptr[i];
    }
}

__global__
void resetLookupTemp(Grid const g, GPUMemory<Lookup> lookupTemp) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.row.count(); i += gridDim.x * blockDim.x) {
        lookupTemp[g.row[i]] = 0;
    }
}

/*
__global__ static void kernel(
    lookup_t const * lookupG0, GridCSR::Vertex const * G0Col,
    GridCSR::Vertex const * G1Row, GridCSR::Vertex const * G1Ptr, GridCSR::Vertex const * G1Col,
    lookup_t const * lookupG2, GridCSR::Vertex const * G2Col,
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
                        GridCSR::CUDA::BinarySearchIntersection(&g1.col[g1col_idx_s], g1col_length, g0.col[g0col_idx], &mycount);
                    }
                } else {
                    for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e; g1col_idx += blockDim.x) {
                        GridCSR::CUDA::BinarySearchIntersection(&g0.col[g0col_idx_s], g0col_length, g1.col[g1col_idx], &mycount);
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