#include "exec.h"
#include "type.h"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

static __device__ void bitmapSet(uint32_t * bm0, uint32_t * bm1, const uint32_t vid)
{
	atomicOr(&bm0[vid >> EXP_BITARR[0]], 1 << ((vid >> (EXP_BITARR[0] - EXP_BITARR[1])) & 31));
	atomicOr(&bm1[vid >> EXP_BITARR[1]], 1 << (vid & 31));
}

static __device__ bool bitmapCheck(uint32_t * bm0, uint32_t * bm1, const uint32_t vid)
{
	if (bm0[vid >> EXP_BITARR[0]] & (1 << ((vid >> (EXP_BITARR[0] - EXP_BITARR[1]) & 31)))) {
		return bm1[vid >> EXP_BITARR[1]] & (1 << (vid & 31));
	} else {
		return false;
	}
}

static __global__ void kernel(Grid3		 g,
							  uint32_t * lookup0,
							  uint32_t * lookup2,
							  uint32_t * bitmap0,
							  uint32_t * bitmap1,
							  Count *	 count)
{
	uint32_t * mybm0   = &bitmap0[(GRID_WIDTH >> EXP_BITARR[0]) * blockIdx.x];
	uint32_t * mybm1   = &bitmap1[(GRID_WIDTH >> EXP_BITARR[1]) * blockIdx.x];
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
			bitmapSet(mybm0, mybm1, g[1][2][g1col_idx]);
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
					if (bitmapCheck(mybm0, mybm1, g[0][2][g0col_idx])) {
						mycount++;
					}
				}
			}
			__syncthreads();
		}

		for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e;
			 g1col_idx += blockDim.x) {

			auto const c = g[1][2][g1col_idx];

			mybm0[c >> EXP_BITARR[0]] = 0;
			mybm1[c >> EXP_BITARR[1]] = 0;
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

Count Manager::runKernel(sp<Context> ctx, Grid3 G)
{
	printf("DEV:%d\n"
		   "ROW:(%p,%ld), PTR:(%p,%ld), COL:(%p,%ld)\n"
		   "ROW:(%p,%ld), PTR:(%p,%ld), COL:(%p,%ld)\n"
		   "ROW:(%p,%ld), PTR:(%p,%ld), COL:(%p,%ld)\n",
		   this->deviceID,
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

	printf("DEV:%d\n"
		   "LOOKUP0:(%p,%ld), LOOKUP1:(%p,%ld), LOOKUP2:(%p,%ld)\n"
		   "BITARR0:(%p,%ld), BITARR1:(%p,%ld)\n"
		   "CUBTMP :(%p,%ld), COUNT  :(%p,%ld)\n",
		   this->deviceID,
		   this->mem.lookUp[0].ptr,
		   this->mem.lookUp[0].byte,
		   this->mem.lookUp[1].ptr,
		   this->mem.lookUp[1].byte,
		   this->mem.lookUp[2].ptr,
		   this->mem.lookUp[2].byte,
		   this->mem.bitArray[0].ptr,
		   this->mem.bitArray[0].byte,
		   this->mem.bitArray[1].ptr,
		   this->mem.bitArray[1].byte,
		   this->mem.cubTemp.ptr,
		   this->mem.cubTemp.byte,
		   this->mem.count.ptr,
		   this->mem.count.byte);

	cudaSetDevice(this->deviceID);
	cudaMemset((void *)this->mem.count.ptr, 0, this->mem.count.byte);
	// cudaMemsetAsync((void *)this->mem.count.ptr, 0, this->mem.count.byte, this->myStream);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	// genLookupTemp<<<ctx->blocks, ctx->threads, 0, this->myStream>>>(G[0],
	// this->mem.lookUp[1].ptr);
	genLookupTemp<<<ctx->blocks, ctx->threads>>>(G[0], this->mem.lookUp[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	/*
	cub::DeviceScan::ExclusiveSum(this->mem.cubTemp.ptr,
								  this->mem.cubTemp.byte,
								  this->mem.lookUp[1].ptr,
								  this->mem.lookUp[0].ptr,
								  this->mem.lookUp[0].count(),
								  this->myStream);
								  */
	cub::DeviceScan::ExclusiveSum(this->mem.cubTemp.ptr,
								  this->mem.cubTemp.byte,
								  this->mem.lookUp[1].ptr,
								  this->mem.lookUp[0].ptr,
								  this->mem.lookUp[0].count());
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	// resetLookupTemp<<<ctx->blocks, ctx->threads, 0, this->myStream>>>(G[0],
	// this->mem.lookUp[1].ptr);
	resetLookupTemp<<<ctx->blocks, ctx->threads>>>(G[0], this->mem.lookUp[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	// genLookupTemp<<<ctx->blocks, ctx->threads, 0, this->myStream>>>(G[2],
	// this->mem.lookUp[1].ptr);
	genLookupTemp<<<ctx->blocks, ctx->threads>>>(G[2], this->mem.lookUp[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	/*
	cub::DeviceScan::ExclusiveSum(this->mem.cubTemp.ptr,
								  this->mem.cubTemp.byte,
								  this->mem.lookUp[1].ptr,
								  this->mem.lookUp[2].ptr,
								  this->mem.lookUp[2].count(),
								  this->myStream);
								  */
	cub::DeviceScan::ExclusiveSum(this->mem.cubTemp.ptr,
								  this->mem.cubTemp.byte,
								  this->mem.lookUp[1].ptr,
								  this->mem.lookUp[2].ptr,
								  this->mem.lookUp[2].count());
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	// resetLookupTemp<<<ctx->blocks, ctx->threads, 0, this->myStream>>>(G[2],
	// this->mem.lookUp[1].ptr);
	resetLookupTemp<<<ctx->blocks, ctx->threads>>>(G[2], this->mem.lookUp[1].ptr);
	CUDACHECK();

	cudaSetDevice(this->deviceID);
	/*
	kernel<<<ctx->blocks, ctx->threads, 0, this->myStream>>>(G,
															 this->mem.lookUp[0].ptr,
															 this->mem.lookUp[2].ptr,
															 this->mem.bitArray[0].ptr,
															 this->mem.bitArray[1].ptr,
															 this->mem.count.ptr);
															 */

	kernel<<<ctx->blocks, ctx->threads>>>(G,
										  this->mem.lookUp[0].ptr,
										  this->mem.lookUp[2].ptr,
										  this->mem.bitArray[0].ptr,
										  this->mem.bitArray[1].ptr,
										  this->mem.count.ptr);
	CUDACHECK();

	Count out = 0;

	cudaSetDevice(this->deviceID);
	// cudaMemcpyAsync( &out, this->mem.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost,
	// this->myStream);
	// cudaMemcpy(&out, this->mem.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost);
	CUDACHECK();

	// cudaStreamSynchronize(this->myStream);
	CUDACHECK();

	return out;
}

} // namespace Exec