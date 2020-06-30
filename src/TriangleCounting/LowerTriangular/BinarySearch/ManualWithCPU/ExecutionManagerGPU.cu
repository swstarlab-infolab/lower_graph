#include "ExecutionManager.h"
#include "context.h"

#include <GridCSR/CUDA/Kernel.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDACHECK()                        \
	do {                                   \
		auto e = cudaGetLastError();       \
		if (e) {                           \
			printf("%s:%d, %s(%d), %s\n",  \
				   __FILE__,               \
				   __LINE__,               \
				   cudaGetErrorName(e),    \
				   e,                      \
				   cudaGetErrorString(e)); \
			cudaDeviceReset();             \
			exit(EXIT_FAILURE);            \
		}                                  \
	} while (false)

static __global__ void test(ExecutionManager::Grid g)
{
	printf("[(%p,%ld)(%p,%ld)(%p,%ld)]\n",
		   g[0].ptr,
		   g[0].byte,
		   g[1].ptr,
		   g[1].byte,
		   g[2].ptr,
		   g[2].byte);
}

static __global__ void genLookupTemp(ExecutionManager::Grid g, uint32_t * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = g[1][i + 1] - g[1][i];
	}
}

static __global__ void resetLookupTemp(ExecutionManager::Grid g, uint32_t * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = 0;
	}
}

static __global__ void
kernel(ExecutionManager::Grids g, uint32_t const * lookup0, uint32_t const * lookup2, Count * count)
{
	Count mycount = 0;

	__shared__ int SHARED[1024];

	for (uint32_t g1row_iter = blockIdx.x; g1row_iter < g[1][0].count(); g1row_iter += gridDim.x) {

		// This makes huge difference!!!
		// Without "Existing Row" information: loop all 2^24 and check it all
		// With "Existing Row" information: extremely faster than without-version
		auto const g1row = g[1][0][g1row_iter];

		if (lookup2[g1row] == lookup2[g1row + 1]) {
			continue;
		}

		auto const g1col_idx_s = g[1][1][g1row_iter];
		auto const g1col_idx_e = g[1][1][g1row_iter + 1];

		// variable for binary tree intersection
		auto const g1col_length = g1col_idx_e - g1col_idx_s;

		auto const g2col_s = lookup2[g1row], g2col_e = lookup2[g1row + 1];

		for (uint32_t g2col_idx = g2col_s; g2col_idx < g2col_e; g2col_idx += blockDim.x) {
			SHARED[threadIdx.x] =
				(g2col_idx + threadIdx.x < g2col_e) ? (int)g[2][2][g2col_idx + threadIdx.x] : -1;

			__syncthreads();

			for (uint32_t s = 0; s < blockDim.x; s++) {
				int const g2col = SHARED[s];
				if (g2col == -1) {
					break;
				}
				if (lookup0[g2col] == lookup0[g2col + 1]) {
					continue;
				}

				auto const g0col_idx_s = lookup0[g2col], g0col_idx_e = lookup0[g2col + 1];

				// variable for binary tree intersection
				auto const g0col_length = g0col_idx_e - g0col_idx_s;

				if (g1col_length >= g0col_length) {
					for (uint32_t g0col_idx = g0col_idx_s + threadIdx.x; g0col_idx < g0col_idx_e;
						 g0col_idx += blockDim.x) {
						GridCSR::CUDA::BinarySearchIntersection(
							&g[1][2][g1col_idx_s], g1col_length, g[0][2][g0col_idx], &mycount);
					}
				} else {
					for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e;
						 g1col_idx += blockDim.x) {
						GridCSR::CUDA::BinarySearchIntersection(
							&g[0][2][g0col_idx_s], g0col_length, g[1][2][g1col_idx], &mycount);
					}
				}
			}
			__syncthreads();
		}
	}

	for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) {
		atomicAdd(count, mycount);
	}
}

static __global__ void print(Count * cnt)
{
	printf("Hello! cnt = %ld\n", *cnt);
	*cnt = 13;
	printf("Hello! cnt = %ld done\n", *cnt);
}

Count ExecutionManager::execGPU(ExecutionManager::Grids G)
{
	printf("AAAA:\n"
		   "[(%p,%ld)(%p,%ld)(%p,%ld)]\n"
		   "[(%p,%ld)(%p,%ld)(%p,%ld)]\n"
		   "[(%p,%ld)(%p,%ld)(%p,%ld)]\n",
		   G[0][0].ptr,
		   G[0][0].byte,
		   G[0][1].ptr,
		   G[0][1].byte,
		   G[0][2].ptr,
		   G[0][2].byte,
		   G[1][0].ptr,
		   G[1][0].byte,
		   G[1][1].ptr,
		   G[1][1].byte,
		   G[1][2].ptr,
		   G[1][2].byte,
		   G[2][0].ptr,
		   G[2][0].byte,
		   G[2][1].ptr,
		   G[2][1].byte,
		   G[2][2].ptr,
		   G[2][2].byte);
	cudaSetDevice(this->ID);
	cudaMemset(this->mem.count.ptr, 0x00, this->mem.count.byte);
	CUDACHECK();

	cudaSetDevice(this->ID);
	// genLookupTemp<<<ctx.gpu.blocks, ctx.gpu.threads>>>(G[0], this->mem.lookup[1].ptr);
	test<<<1, 1>>>(G[0]);
	CUDACHECK();

	/*
		cudaSetDevice(this->ID);
		cub::DeviceScan::ExclusiveSum(this->mem.scan.ptr,
									  this->mem.scan.byte,
									  this->mem.lookup[1].ptr,
									  this->mem.lookup[0].ptr,
									  this->mem.lookup[0].count());
		CUDACHECK();

		cudaSetDevice(this->ID);
		resetLookupTemp<<<ctx.gpu.blocks, ctx.gpu.threads>>>(G[0], this->mem.lookup[1].ptr);
		CUDACHECK();

		cudaSetDevice(this->ID);
		genLookupTemp<<<ctx.gpu.blocks, ctx.gpu.threads>>>(G[2], this->mem.lookup[1].ptr);
		CUDACHECK();

		cudaSetDevice(this->ID);
		cub::DeviceScan::ExclusiveSum(this->mem.scan.ptr,
									  this->mem.scan.byte,
									  this->mem.lookup[1].ptr,
									  this->mem.lookup[2].ptr,
									  this->mem.lookup[2].count());
		CUDACHECK();

		cudaSetDevice(this->ID);
		resetLookupTemp<<<ctx.gpu.blocks, ctx.gpu.threads>>>(G[2], this->mem.lookup[1].ptr);
		CUDACHECK();

		cudaSetDevice(this->ID);
		kernel<<<ctx.gpu.blocks, ctx.gpu.threads>>>(
			G, this->mem.lookup[0].ptr, this->mem.lookup[2].ptr, this->mem.count.ptr);
		*/
	CUDACHECK();
	print<<<1, 1>>>(this->mem.count.ptr);
	CUDACHECK();

	Count cnt = 0;
	cudaSetDevice(this->ID);
	// cudaMemcpyAsync(&cnt, this->mem.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost, stream);
	cudaMemcpy(&cnt, this->mem.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost);
	CUDACHECK();

	// cudaStreamSynchronize(stream); CUDACHECK();
	// cudaDeviceSynchronize(); CUDACHECK();
	// cudaStreamDestroy(stream); CUDACHECK();

	return cnt;
}

Count ExecutionManager::execCPU(ExecutionManager::Grids G)
{
	Count cnt = 0;
	return cnt;
}