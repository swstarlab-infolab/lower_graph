#include "exec_man.h"
#include "util.h"

#include <cub/device/device_scan.cuh>

namespace Exec
{

Manager::Manager(int const						 deviceID,
				 GPUSetting const				 gpuSetting,
				 std::shared_ptr<Sched::Manager> sched,
				 std::shared_ptr<Data::Manager>	 data)
	: deviceID(deviceID), gpuSetting(gpuSetting), sched(sched), data(data)
{
	printf("Constructor: Exec::Manager, deviceId=%d\n", this->deviceID);

	cudaSetDevice(this->deviceID);
	cudaStreamCreate(&this->myStream);
	auto e = cudaStreamQuery(this->myStream);
	printf("Exec::Manager: %d, Stream: %p, %s(%d), %s\n",
		   this->deviceID,
		   this->myStream,
		   cudaGetErrorName(e),
		   e,
		   cudaGetErrorString(e));

	{
		auto & target = this->mem.bitarr[0];
		target.byte =
			sizeof(uint32_t) * this->gpuSetting.block * ceil(GRIDWIDTH, 1L << EXP_BITMAP0);
		printf("bitmap0: %ld\n", target.byte);
		target.ptr = (uint32_t *)this->data->alloc(target.byte);
	}

	{
		auto & target = this->mem.bitarr[1];
		target.byte =
			sizeof(uint32_t) * this->gpuSetting.block * ceil(GRIDWIDTH, 1L << EXP_BITMAP1);
		printf("bitmap1: %ld\n", target.byte);
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
}
} // namespace Exec