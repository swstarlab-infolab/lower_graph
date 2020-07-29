#include "data.h"
#include "exec.h"
#include "util.h"

#include <cub/device/device_scan.cuh>

namespace Exec
{

sp<bchan<JobResult>> Manager::run(sp<Context> ctx, sp<bchan<Job>> in)
{
	auto out = makeSp<bchan<JobResult>>(16);

	std::thread([=] {
		Grid3 memInfo;

		for (auto & job : *in) {
			// Ready
			parallelFiber(3 * 3, [&](size_t const i) {
				auto temp =
					Data::managerSpace[this->deviceID]->ready(job[i / 3] + ctx->extension[i % 3]);
				memInfo[i / 3][i % 3].ptr  = temp.ptr;
				memInfo[i / 3][i % 3].byte = temp.byte;
			});

			// Push Result
			JobResult res;
			res.job		 = job;
			res.triangle = this->runKernel(ctx, memInfo);

			// Done
			parallelFiber(3 * 3, [&](size_t const i) {
				Data::managerSpace[this->deviceID]->done(job[i / 3] + ctx->extension[i % 3]);
			});

			out->push(res);
		}

		out->close();
	}).detach();

	return out;
}

void Manager::init(sp<Context> ctx, int const deviceID)
{
	this->deviceID = deviceID;

	if (this->deviceID >= 0) {
		cudaSetDevice(this->deviceID);
		CUDACHECK();
		cudaStreamCreate(&this->myStream);
		CUDACHECK();

		for (auto & e : this->mem.lookUp) {
			e.byte = sizeof(uint32_t) * GRID_WIDTH;
			cudaSetDevice(this->deviceID);
			CUDACHECK();
			e.ptr = (uint32_t *)(Data::managerSpace[this->deviceID]->malloc(e.byte));
			CUDACHECK();
			cudaMemset(e.ptr, 0x00, e.byte);
			CUDACHECK();
		}

		{
			cudaSetDevice(this->deviceID);
			CUDACHECK();
			auto & e = this->mem.cubTemp;
			cub::DeviceScan::ExclusiveSum(nullptr,
										  e.byte,
										  this->mem.lookUp[1].ptr,
										  this->mem.lookUp[0].ptr,
										  this->mem.lookUp[0].byte / sizeof(uint32_t));
			CUDACHECK();
			e.ptr = (uint32_t *)Data::managerSpace[this->deviceID]->malloc(e.byte);
			CUDACHECK();
			// cudaMalloc(&e.ptr, e.byte);
			cudaMemset(e.ptr, 0x00, e.byte);
			CUDACHECK();
		}

		{
			auto & e = this->mem.count;
			e.byte	 = sizeof(Count);
			cudaSetDevice(this->deviceID);
			CUDACHECK();
			e.ptr = (Count *)Data::managerSpace[this->deviceID]->malloc(e.byte);
			CUDACHECK();
			// cudaMalloc(&e.ptr, e.byte);
			printf("DEV:%d, %p, %ld\n", this->deviceID, e.ptr, e.byte);
			cudaMemset(e.ptr, 0x00, e.byte);
			CUDACHECK();
		}

		for (size_t i = 0; i < this->mem.bitArray.size(); i++) {
			auto & e = this->mem.bitArray[i];
			e.byte	 = sizeof(uint32_t) * ctx->blocks * ceil(GRID_WIDTH, 1L << EXP_BITARR[i]);
			cudaSetDevice(this->deviceID);
			CUDACHECK();
			e.ptr = (uint32_t *)Data::managerSpace[this->deviceID]->malloc(e.byte);
			CUDACHECK();
			// cudaMalloc(&e.ptr, e.byte);
			cudaMemset(e.ptr, 0x00, e.byte);
			CUDACHECK();
		}
	}
}

} // namespace Exec