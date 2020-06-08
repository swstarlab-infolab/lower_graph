#include "DataManager.cuh"
#include "ExecutionManager.h"
#include "ScheduleManager.h"
#include "make.h"
#include "type.h"

#include <GridCSR/GridCSR.h>
#include <boost/fiber/all.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

auto DataManagerInit(Context & ctx, int myID)
{

	using DataChanType = bchan<Tx>;

	auto & myMem = ctx.dataManagerCtx[myID];

	printf("Start to initialize Device: %d\n", myID);
	if (myID >= 0) {
		// GPU Memory
		cudaSetDevice(myID);
		size_t freeMem;
		cudaMemGetInfo(&freeMem, nullptr);
		freeMem -= (1L << 29);
		myMem.buf	= allocCUDAByte(freeMem);
		myMem.buddy = std::make_shared<portable_buddy_system>();
		myMem.buddy.get()->init(memrgn_t{myMem.buf.get(), freeMem}, 256, 1);
		myMem.conn				   = std::make_shared<DataManagerContext::Connections>();
		myMem.conn.get()->upstream = -1;
		for (int32_t i = 0; i < ctx.deviceCount; i++) {
			if (myID != i) {
				myMem.conn.get()->neighbor.push_back(i);
			}
		}

		myMem.chan	= std::make_shared<bchan<Tx>>(16);
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 10); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else if (myID == -1) {
		// CPU Memory
		// size_t freeMem = (1L << 35);
		size_t freeMem = (1L << 34);
		myMem.buf	   = allocHostByte(freeMem);
		myMem.buddy	   = std::make_shared<portable_buddy_system>();
		myMem.buddy->init(memrgn_t{myMem.buf.get(), freeMem}, 8, 1);
		myMem.conn				   = std::make_shared<DataManagerContext::Connections>();
		myMem.conn.get()->upstream = -2;
		myMem.chan				   = std::make_shared<bchan<Tx>>(16);
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 10); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else {
		// Storage
		myMem.conn				   = std::make_shared<DataManagerContext::Connections>();
		myMem.conn.get()->upstream = -2;
		myMem.chan				   = std::make_shared<bchan<Tx>>(16);
	}
	printf("Start to initialize Device: %d, Done\n", myID);
}

void init(Context & ctx, int argc, char * argv[])
{
	// Argument
	if (argc != 5) {
		fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	ctx.folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	ctx.meta.Load(ctx.folderPath / "meta.json");
	for (int i = 0; i < 3; i++) {
		ctx.setting[i] = strtol(argv[i + 2], nullptr, 10);
	}

	// get total GPUs
	cudaGetDeviceCount(&ctx.deviceCount);

	// -1     : CPU
	//  0 ~  N: GPU
	// -2 ~ -N: Storage
	for (int32_t i = 0; i < ctx.deviceCount; i++) {
		DataManagerInit(ctx, i); // GPU
	}
	DataManagerInit(ctx, -1); // CPU
	DataManagerInit(ctx, -2); // Storage
}

int main(int argc, char * argv[])
{
	using DataTxChanPtr = std::shared_ptr<bchan<Tx>>;
	using ResultChanPtr = std::shared_ptr<bchan<CommandResult>>;

	Context ctx;
	init(ctx, argc, argv);

	auto exeReq = ScheduleManager(ctx);

	std::vector<ResultChanPtr> resultChan(ctx.deviceCount);

	for (int i = 0; i < ctx.deviceCount; i++) {
		DataManager(ctx, i);
		resultChan[i] = Computation(ctx, i, exeReq);
	}
	DataManager(ctx, -1);
	DataManager(ctx, -2);
	auto c = merge(resultChan);
	ScheduleWaiter(c);

	return 0;
}