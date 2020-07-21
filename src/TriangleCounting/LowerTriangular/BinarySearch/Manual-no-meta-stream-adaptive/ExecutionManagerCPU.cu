/*
#include "ExecutionManager.cuh"

#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>
#include <thread>

static void genLookupTemp(Grid const g, Lookup * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = g[1][i + 1] - g[1][i];
	}
}

static void resetLookupTemp(Grid const g, Lookup * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = 0;
	}
}

static void kernel(Grids const g, Lookup const * lookup0, Lookup const * lookup2, Count * count)
{
	Count mycount = 0;

	int SHARED[1024];

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

Count launchKernelCPU(Context & ctx, DeviceID myID, Grids & G)
{
	auto & myCtx   = ctx.executionManagerCtx[myID];
	auto & blocks  = ctx.setting[1];
	auto & threads = ctx.setting[2];

	// if (!(G[0][0].byte && G[1][0].byte && G[2][0].byte)) { return 0; }
	Count cnt = 0;

	memset(myCtx.count.ptr, 0, myCtx.count.byte);
	genLookupTemp(G[0], myCtx.lookup.temp.ptr);

	tbb::parallel_scan(tbb::blocked_range<>(0, n), );
	tbb:: ::ExclusiveSum(myCtx.cub.ptr,
						 myCtx.cub.byte,
						 myCtx.lookup.temp.ptr,
						 myCtx.lookup.G0.ptr,
						 myCtx.lookup.G0.count());

	resetLookupTemp(G[0], myCtx.lookup.temp.ptr);
	genLookupTemp(G[2], myCtx.lookup.temp.ptr);
	cub::DeviceScan::ExclusiveSum(myCtx.cub.ptr,
								  myCtx.cub.byte,
								  myCtx.lookup.temp.ptr,
								  myCtx.lookup.G2.ptr,
								  myCtx.lookup.G2.count());

	resetLookupTemp(G[2], myCtx.lookup.temp.ptr);
	kernel(G, myCtx.lookup.G0.ptr, myCtx.lookup.G2.ptr, myCtx.count.ptr);

	// cudaStreamSynchronize(stream); CUDACHECK();
	// cudaDeviceSynchronize(); CUDACHECK();
	// cudaStreamDestroy(stream); CUDACHECK();

	return cnt;
}
*/