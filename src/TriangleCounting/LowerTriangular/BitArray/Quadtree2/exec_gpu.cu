#include "exec_man.h"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gdrapi.h>
#include <iostream>

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

namespace Exec
{

static __global__ void genLookupTemp(Grid g, uint32_t * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = g[1][i + 1] - g[1][i];
	}
}

static __global__ void resetLookupTemp(Grid g, uint32_t * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = 0;
	}
}

static __device__ void bitarrSet(uint32_t * bm0, uint32_t * bm1, const uint32_t vid)
{
	atomicOr(&bm0[vid >> EXP_BITMAP0], 1 << ((vid >> (EXP_BITMAP0 - EXP_BITMAP1)) & 31));
	atomicOr(&bm1[vid >> EXP_BITMAP1], 1 << (vid & 31));
}

static __device__ bool bitarrCheck(uint32_t * bm0, uint32_t * bm1, const uint32_t vid)
{
	if (bm0[vid >> EXP_BITMAP0] & (1 << ((vid >> (EXP_BITMAP0 - EXP_BITMAP1) & 31)))) {
		return bm1[vid >> EXP_BITMAP1] & (1 << (vid & 31));
	} else {
		return false;
	}
}

static __global__ void kernel(Grids			   g,
							  uint32_t const * lookup0,
							  uint32_t const * lookup2,
							  uint32_t *	   bitarr0,
							  uint32_t *	   bitarr1,
							  Count *		   count)
{
	uint32_t * mybm0   = &bitarr0[(GRIDWIDTH >> EXP_BITMAP0) * blockIdx.x];
	uint32_t * mybm1   = &bitarr1[(GRIDWIDTH >> EXP_BITMAP1) * blockIdx.x];
	Count	   mycount = 0;

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

		for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e;
			 g1col_idx += blockDim.x) {
			bitarrSet(mybm0, mybm1, g[1][2][g1col_idx]);
		}

		// variable for binary tree intersection
		// auto const g1col_length = g1col_idx_e - g1col_idx_s;

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
				// auto const g0col_length = g0col_idx_e - g0col_idx_s;

				for (uint32_t g0col_idx = g0col_idx_s + threadIdx.x; g0col_idx < g0col_idx_e;
					 g0col_idx += blockDim.x) {
					if (bitarrCheck(mybm0, mybm1, g[0][2][g0col_idx])) {
						mycount++;
					}
				}
			}
			__syncthreads();
		}

		for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e;
			 g1col_idx += blockDim.x) {

			auto const c = g[1][2][g1col_idx];

			mybm0[c >> EXP_BITMAP0] = 0;
			mybm1[c >> EXP_BITMAP1] = 0;
		}

		__syncthreads();
	}

	for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) {
		atomicAdd(count, mycount);
	}
}

Count Manager::launchKernelGPU(Grids & G)
{
	auto & stream  = this->myStream;
	auto & blocks  = this->gpuSetting.block;
	auto & threads = this->gpuSetting.thread;

	if (!(G[0][0].byte && G[1][0].byte && G[2][0].byte)) {
		return 0;
	}

	/*
	printf("this->mem.count == %p, %ld\n", this->mem.count.ptr, this->mem.count.byte);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("G[%d][%d] == %p, %ld\n", i, j, G[i][j].ptr, G[i][j].count());
		}
	}

	for (int i = 0; i < 3; i++) {
		printf("this->mem.lookup[%d] == %p, %ld\n",
			   i,
			   this->mem.lookup[i].ptr,
			   this->mem.lookup[i].byte);
	}

	for (int i = 0; i < 2; i++) {
		printf("this->mem.bitarr[%d] == %p, %ld\n",
			   i,
			   this->mem.bitarr[i].ptr,
			   this->mem.bitarr[i].byte);
	}

	printf("this->mem.cub == %p, %ld\n", this->mem.cub.ptr, this->mem.cub.byte);
	*/

	cudaSetDevice(this->deviceID);
	cudaMemsetAsync(this->mem.count.ptr, 0x00, this->mem.count.byte, stream);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	genLookupTemp<<<blocks, threads, 0, stream>>>(G[0], this->mem.lookup[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	cub::DeviceScan::ExclusiveSum(this->mem.cub.ptr,
								  this->mem.cub.byte,
								  this->mem.lookup[1].ptr,
								  this->mem.lookup[0].ptr,
								  this->mem.lookup[0].count(),
								  stream);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	resetLookupTemp<<<blocks, threads, 0, stream>>>(G[0], this->mem.lookup[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	genLookupTemp<<<blocks, threads, 0, stream>>>(G[2], this->mem.lookup[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	cub::DeviceScan::ExclusiveSum(this->mem.cub.ptr,
								  this->mem.cub.byte,
								  this->mem.lookup[1].ptr,
								  this->mem.lookup[2].ptr,
								  this->mem.lookup[2].count(),
								  stream);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	resetLookupTemp<<<blocks, threads, 0, stream>>>(G[2], this->mem.lookup[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	kernel<<<blocks, threads, 0, stream>>>(G,
										   this->mem.lookup[0].ptr,
										   this->mem.lookup[2].ptr,
										   this->mem.bitarr[0].ptr,
										   this->mem.bitarr[1].ptr,
										   this->mem.count.ptr);
	CUDACHECK();

	Count cnt = 0;
	cudaSetDevice(this->deviceID);
	cudaMemcpyAsync(&cnt, this->mem.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost, stream);
	CUDACHECK();

	cudaStreamSynchronize(stream);
	CUDACHECK();

	return cnt;
}
} // namespace Exec
