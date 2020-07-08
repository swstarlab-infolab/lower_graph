#include "ExecutionManager.h"
#include "context.h"
#include "type.h"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <thread>

void ExecutionManager::init(int const id, sp<DataManager> dm)
{
	this->ID = id;
	this->DM = dm;

	if (this->ID > -1) {
		this->initGPU();
	} else if (this->ID == -1) {
		this->initCPU();
	} else {
		throw std::runtime_error("Wrong Index");
	}
}

void ExecutionManager::initCPU()
{
	// std::cout << "EM: CPU init" << std::endl;

	for (auto & lu : this->mem.lookup) {
		lu.byte = sizeof(Vertex) * ctx.grid.width;
		lu.ptr	= (Vertex *)this->DM->manualAlloc(lu.byte);
		memset(lu.ptr, 0x00, lu.byte);
	}

	this->mem.count.byte = sizeof(Count);
	this->mem.count.ptr	 = (Count *)this->DM->manualAlloc(this->mem.count.byte);
	memset(this->mem.count.ptr, 0x00, this->mem.count.byte);
}

void ExecutionManager::initGPU()
{
	// std::cout << "EM: GPU " << this->ID << " init" << std::endl;

	for (auto & lu : this->mem.lookup) {
		cudaSetDevice(this->ID);
		lu.byte = sizeof(Vertex) * ctx.grid.width;
		lu.ptr	= (Vertex *)this->DM->manualAlloc(lu.byte);
		cudaMemset(lu.ptr, 0x00, lu.byte);
	}

	cudaSetDevice(this->ID);
	this->mem.count.byte = sizeof(Count);
	this->mem.count.ptr	 = (Count *)this->DM->manualAlloc(this->mem.count.byte);
	cudaMemset(this->mem.count.ptr, 0x00, this->mem.count.byte);

	cudaSetDevice(this->ID);
	cub::DeviceScan::ExclusiveSum(nullptr,
								  this->mem.scan.byte,
								  this->mem.lookup[1].ptr,
								  this->mem.lookup[0].ptr,
								  this->mem.lookup[0].count());
	this->mem.scan.ptr = (Count *)this->DM->manualAlloc(this->mem.scan.byte);
	cudaMemset(this->mem.scan.ptr, 0x00, this->mem.scan.byte);
}

void ExecutionManager::run()
{
	std::thread([&] {
		// printf("EM: start %d\n", this->ID);

		auto myInChan  = (this->ID > -1) ? ctx.chan.orderGPU : ctx.chan.orderCPU;
		auto myOutChan = ctx.chan.report[this->ID];

		for (auto & order : *myInChan) {
			std::array<std::array<fiber, 3>, 3> fibers;
			ExecutionManager::Grids				mInfo;
			// printf("EM: ORDER RECEIVED\n");

			for (uint8_t i = 0; i < fibers.size(); i++) {
				for (uint8_t j = 0; j < fibers[i].size(); j++) {
					fibers[i][j] = fiber([&, i, j] {
						DataManager::Tx tx;

						auto result = this->DM->reqReady(order[i], (DataManager::Type)j);
						mInfo[i][j] = result.info;
					});
				}
			}

			for (uint8_t i = 0; i < fibers.size(); i++) {
				for (uint8_t j = 0; j < fibers[i].size(); j++) {
					if (fibers[i][j].joinable()) {
						fibers[i][j].join();
					}
				}
			}

			printf("Kernel End:\n"
				   "(%d,%d):[(%p,%ld)(%p,%ld)(%p,%ld)]\n"
				   "(%d,%d):[(%p,%ld)(%p,%ld)(%p,%ld)]\n"
				   "(%d,%d):[(%p,%ld)(%p,%ld)(%p,%ld)]\n",
				   order[0][0],
				   order[0][1],
				   mInfo[0][0].ptr,
				   mInfo[0][0].byte,
				   mInfo[0][1].ptr,
				   mInfo[0][1].byte,
				   mInfo[0][2].ptr,
				   mInfo[0][2].byte,
				   order[1][0],
				   order[1][1],
				   mInfo[1][0].ptr,
				   mInfo[1][0].byte,
				   mInfo[1][1].ptr,
				   mInfo[1][1].byte,
				   mInfo[1][2].ptr,
				   mInfo[1][2].byte,
				   order[2][0],
				   order[2][1],
				   mInfo[2][0].ptr,
				   mInfo[2][0].byte,
				   mInfo[2][1].ptr,
				   mInfo[2][1].byte,
				   mInfo[2][2].ptr,
				   mInfo[2][2].byte);

			Report report;
			report.g3		= order;
			report.deviceID = this->ID;
			report.triangle = (this->ID > -1) ? this->execGPU(mInfo) : this->execCPU(mInfo);

			for (uint8_t i = 0; i < fibers.size(); i++) {
				for (uint8_t j = 0; j < fibers[i].size(); j++) {
					fibers[i][j] = fiber([&, i, j] {
						DataManager::Tx tx;
						// printf("EM:    try to reqDone...\n");
						this->DM->reqDone(order[i], (DataManager::Type)j);
					});
				}
			}

			for (uint8_t i = 0; i < fibers.size(); i++) {
				for (uint8_t j = 0; j < fibers[i].size(); j++) {
					if (fibers[i][j].joinable()) {
						fibers[i][j].join();
					}
				}
			}

			// printf("EM: ORDER COMPLETE\n");
			myOutChan->push(report);
		}

		// ctx finalize closes all data manager channels

		myOutChan->close();
	}).detach();
}