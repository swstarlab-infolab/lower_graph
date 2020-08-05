#include "exec_man.h"
#include "util.h"

#include <cub/device/device_scan.cuh>

namespace Exec
{

Manager::Manager(int const						 deviceID,
				 int const						 streamID,
				 GPUSetting const				 gpuSetting,
				 std::shared_ptr<Sched::Manager> sched,
				 std::shared_ptr<Data::Manager>	 data)
	: deviceID(deviceID), streamID(streamID), gpuSetting(gpuSetting), sched(sched), data(data)
{

	cudaSetDevice(this->deviceID);
	cudaStreamCreate(&this->myStream);
	auto e = cudaStreamQuery(this->myStream);
	if (e != cudaSuccess) {
		printf("Constructor: Error, Exec::Manager=(%d, %d), Stream: %p, %s(%d), %s\n",
			   this->deviceID,
			   this->streamID,
			   this->myStream,
			   cudaGetErrorName(e),
			   e,
			   cudaGetErrorString(e));
		assert(e == cudaSuccess);
	}

	{
		auto & target = this->mem.bitarr[0];
		target.byte =
			sizeof(uint32_t) * this->gpuSetting.block * ceil(GRIDWIDTH, 1L << EXP_BITMAP0);
		// printf("bitmap0: %ld\n", target.byte);
		target.ptr = (uint32_t *)this->data->alloc(target.byte);
	}

	{
		auto & target = this->mem.bitarr[1];
		target.byte =
			sizeof(uint32_t) * this->gpuSetting.block * ceil(GRIDWIDTH, 1L << EXP_BITMAP1);
		// printf("bitmap1: %ld\n", target.byte);
		target.ptr = (uint32_t *)this->data->alloc(target.byte);
	}

	for (int i = 0; i < 3; i++) {
		auto & target = this->mem.lookup[i];
		target.byte	  = sizeof(uint32_t) * GRIDWIDTH;
		target.ptr	  = (uint32_t *)this->data->alloc(target.byte);
	}

	{
		auto & target = this->mem.cub;
		cub::DeviceScan::ExclusiveSum(nullptr,
									  target.byte,
									  this->mem.lookup[1].ptr,
									  this->mem.lookup[0].ptr,
									  this->mem.lookup[0].count());
		target.ptr = this->data->alloc(target.byte);
	}

	{
		auto & target = this->mem.count;
		target.byte	  = sizeof(Count);
		target.ptr	  = (Count *)this->data->alloc(target.byte);
	}

	printf("Constructor: Exec::Manager, deviceID=%d, streamID=%d, Init Complete\n",
		   this->deviceID,
		   this->streamID);
}
} // namespace Exec