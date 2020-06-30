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
	std::cout << "EM: CPU init" << std::endl;

	for (auto & lu : this->mem.lookup) {
		lu.byte = sizeof(Vertex32) * ctx.grid.width;
		lu.ptr	= (Vertex32 *)this->DM->manualAlloc(lu.byte);
		memset(lu.ptr, 0x00, lu.byte);
	}

	this->mem.count.byte = sizeof(Count);
	this->mem.count.ptr	 = (Count *)this->DM->manualAlloc(this->mem.count.byte);
	memset(this->mem.count.ptr, 0x00, this->mem.count.byte);
}

void ExecutionManager::initGPU()
{
	std::cout << "EM: GPU " << this->ID << " init" << std::endl;

	for (auto & lu : this->mem.lookup) {
		cudaSetDevice(this->ID);
		lu.byte = sizeof(Vertex32) * ctx.grid.width;
		lu.ptr	= (Vertex32 *)this->DM->manualAlloc(lu.byte);
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
		printf("EM: start %d\n", this->ID);

		auto myInChan  = (this->ID > -1) ? ctx.chan.orderGPU : ctx.chan.orderCPU;
		auto myOutChan = ctx.chan.report[this->ID];

		for (auto & order : *myInChan) {
			std::array<std::array<sp<bchan<DataManager::TxCb>>, 3>, 3> callbacks;

			for (auto & cbrow : callbacks) {
				for (auto & cb : cbrow) {
					cb = makeSp<bchan<DataManager::TxCb>>(2);
				}
			}

			std::array<std::array<fiber, 3>, 3> fibers;

			for (uint8_t i = 0; i < callbacks.size(); i++) {
				for (uint8_t j = 0; j < callbacks[i].size(); j++) {
					fibers[i][j] = fiber([&, i, j] {
						DataManager::Tx tx;
						this->DM->reqReady(order[i], (DataManager::Type)j);
					});
				}
			}

			for (uint8_t i = 0; i < callbacks.size(); i++) {
				for (uint8_t j = 0; j < callbacks[i].size(); j++) {
					if (fibers[i][j].joinable()) {
						fibers[i][j].join();
					}
				}
			}

			Report report;
			report.g3		= order;
			report.deviceID = this->ID;
			report.triangle = 0; // Something Calc

			myOutChan->push(report);

			for (uint8_t i = 0; i < callbacks.size(); i++) {
				for (uint8_t j = 0; j < callbacks[i].size(); j++) {
					fibers[i][j] = fiber([&, i, j] {
						DataManager::Tx tx;
						this->DM->reqDone(order[i], (DataManager::Type)j);
					});
				}
			}

			for (uint8_t i = 0; i < callbacks.size(); i++) {
				for (uint8_t j = 0; j < callbacks[i].size(); j++) {
					if (fibers[i][j].joinable()) {
						fibers[i][j].join();
					}
				}
			}
		}
		myOutChan->close();
		this->DM->closeAllChan();
	}).detach();
}