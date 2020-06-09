#include "ExecutionManager.cuh"
#include "make.cuh"
#include "type.cuh"

#include <BuddySystem/BuddySystem.h>
#include <GridCSR/CUDA/Kernel.cuh>
#include <array>
#include <chrono>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <memory>
#include <thread>

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

static __global__ void genLookupTemp(Grid const g, Lookup * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = g[1][i + 1] - g[1][i];
	}
}

static __global__ void resetLookupTemp(Grid const g, Lookup * luTemp)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g[0].count();
		 i += gridDim.x * blockDim.x) {
		luTemp[g[0][i]] = 0;
	}
}

static __global__ void
kernel(Grids const g, Lookup const * lookup0, Lookup const * lookup2, Count * count)
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

static Count launchKernel(Context & ctx, DeviceID myID, Grids & G)
{
	auto & myCtx   = ctx.executionManagerCtx[myID];
	auto & blocks  = ctx.setting[1];
	auto & threads = ctx.setting[2];

	// if (!(G[0][0].byte && G[1][0].byte && G[2][0].byte)) { return 0; }

	// cudaStream_t stream;

	cudaSetDevice(myID);
	cudaMemset(myCtx.lookup.temp.ptr, 0, myCtx.lookup.temp.byte);
	cudaMemset(myCtx.lookup.G0.ptr, 0, myCtx.lookup.G0.byte);
	cudaMemset(myCtx.lookup.G2.ptr, 0, myCtx.lookup.G2.byte);
	cudaMemset(myCtx.count.ptr, 0, myCtx.count.byte);

	// cudaStreamCreate(&stream);
	// CUDACHECK();
	cudaSetDevice(myID);
	genLookupTemp<<<blocks, threads>>>(G[0], myCtx.lookup.temp.ptr);
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	cub::DeviceScan::ExclusiveSum(myCtx.cub.ptr,
								  myCtx.cub.byte,
								  myCtx.lookup.temp.ptr,
								  myCtx.lookup.G0.ptr,
								  myCtx.lookup.G0.count());
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	resetLookupTemp<<<blocks, threads>>>(G[0], myCtx.lookup.temp.ptr);
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	genLookupTemp<<<blocks, threads>>>(G[2], myCtx.lookup.temp.ptr);
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	cub::DeviceScan::ExclusiveSum(myCtx.cub.ptr,
								  myCtx.cub.byte,
								  myCtx.lookup.temp.ptr,
								  myCtx.lookup.G2.ptr,
								  myCtx.lookup.G2.count());
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	resetLookupTemp<<<blocks, threads>>>(G[2], myCtx.lookup.temp.ptr);
	cudaDeviceSynchronize();

	cudaSetDevice(myID);
	kernel<<<blocks, threads>>>(G, myCtx.lookup.G0.ptr, myCtx.lookup.G2.ptr, myCtx.count.ptr);

	// cudaSetDevice(myID);
	// edgeCount<<<blocks, threads>>>(G, myCtx.count.ptr);

	Count cnt = 0;
	cudaSetDevice(myID);
	// cudaMemcpyAsync(&cnt, myCtx.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost, stream);
	cudaMemcpy(&cnt, myCtx.count.ptr, sizeof(Count), cudaMemcpyDeviceToHost);
	CUDACHECK();

	// cudaStreamSynchronize(stream); CUDACHECK();
	// cudaDeviceSynchronize(); CUDACHECK();
	// cudaStreamDestroy(stream); CUDACHECK();

	return cnt;
}
// Under construction...

static void ExecutionGPU(Context &							   ctx,
						 DeviceID							   myID,
						 std::shared_ptr<bchan<Command>>	   in,
						 std::shared_ptr<bchan<CommandResult>> out)
{
	using DataTxCallback = bchan<MemInfo<Vertex>>;

	size_t hitCount = 0, missCount = 0;

	for (auto & req : *in) {
		// PREPARE
		auto start = std::chrono::system_clock::now();

		Grids								memInfo;
		std::array<std::array<fiber, 3>, 3> waitGroup;
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i][type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					Tx tx;
					tx.method = Method::Ready;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.dataManagerCtx[myID].chan->push(tx);

					for (auto & cbres : *callback) {
						memInfo[i][type] = cbres;
					}
				});
			}
		}

		// Must wait all memory info
		for (auto & row : waitGroup) {
			for (auto & w : row) {
				if (w.joinable()) {
					w.join();
				}
			}
		}

		for (auto & row : memInfo) {
			for (auto & i : row) {
				if (i.hit) {
					hitCount++;
				} else {
					missCount++;
				}
			}
		}

		// LAUNCH
		auto tri = launchKernel(ctx, myID, memInfo);

		/*
				printf("Kernel End:\n"
					   "(%d,%d):[%s,%s,%s]\n"
					   "(%d,%d):[%s,%s,%s]\n"
					   "(%d,%d):[%s,%s,%s]\n",
					   req.gidx[0][0],
					   req.gidx[0][1],
					   memInfo[0][0].print().c_str(),
					   memInfo[0][1].print().c_str(),
					   memInfo[0][2].print().c_str(),
					   req.gidx[1][0],
					   req.gidx[1][1],
					   memInfo[1][0].print().c_str(),
					   memInfo[1][1].print().c_str(),
					   memInfo[1][2].print().c_str(),
					   req.gidx[2][0],
					   req.gidx[2][1],
					   memInfo[2][0].print().c_str(),
					   memInfo[2][1].print().c_str(),
					   memInfo[2][2].print().c_str());
					   */

		auto end = std::chrono::system_clock::now();

		// RELEASE MEMORY
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i][type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					Tx tx;
					tx.method = Method::Done;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.dataManagerCtx[myID].chan->push(tx);

					for (auto & cbres : *callback) {
						memInfo[i][type] = cbres;
					}
				});
			}
		}

		for (auto & row : waitGroup) {
			for (auto & w : row) {
				if (w.joinable()) {
					w.join();
				}
			}
		}

		// CALLBACK RESPONSE
		CommandResult res;
		res.gidx		= req.gidx;
		res.deviceID	= myID;
		res.triangle	= tri;
		res.elapsedTime = std::chrono::duration<double>(end - start).count();

		out->push(res);
	}

	ctx.dataManagerCtx[myID].chan->close();
	out->close();

	printf("HIT: %ld, MISS: %ld, HIT/TOTAL: %lf\n",
		   hitCount,
		   missCount,
		   double(hitCount) / double(hitCount + missCount));
}

std::shared_ptr<bchan<CommandResult>>
ExecutionManager(Context & ctx, int myID, std::shared_ptr<bchan<Command>> in)
{
	// auto out = make<bchan<CommandResult>>(1 << 4);
	auto out = std::make_shared<bchan<CommandResult>>(1 << 4);
	// prepare channels
	if (myID < -1) {
		// No operation
	} else if (myID == -1) {
		// std::thread([&, myID, in, out] { ExecutionCPU(ctx, myID, in, out); }).detach();
	} else {
		std::thread([&, myID, in, out] { ExecutionGPU(ctx, myID, in, out); }).detach();
	}

	return out;
}