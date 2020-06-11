#include "DataManager.cuh"
#include "ExecutionManager.cuh"
#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"

#include <GridCSR/GridCSR.h>
#include <boost/fiber/all.hpp>
#include <cub/device/device_scan.cuh>
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
	if (myID > -1) {
		// GPU Memory
		size_t freeMem;
		cudaSetDevice(myID);
		cudaMemGetInfo(&freeMem, nullptr);
		freeMem -= (1L << 29);
		cudaSetDevice(myID);
		myMem.buf	= allocCUDA<void>(freeMem);
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
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else if (myID == -1) {
		// CPU Memory
		size_t freeMem = (1L << 35);
		myMem.buf	   = allocHost<void>(freeMem);
		myMem.buddy	   = std::make_shared<portable_buddy_system>();
		myMem.buddy->init(memrgn_t{myMem.buf.get(), freeMem}, 8, 1);
		myMem.conn				   = std::make_shared<DataManagerContext::Connections>();
		myMem.conn.get()->upstream = -2;
		myMem.chan				   = std::make_shared<bchan<Tx>>(16);
		myMem.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myMem.cacheMtx = std::make_shared<std::mutex>();
	} else {
		// Storage
		myMem.conn				   = std::make_shared<DataManagerContext::Connections>();
		myMem.conn.get()->upstream = -2;
		myMem.chan				   = std::make_shared<bchan<Tx>>(16);
	}
}

void ExecutionManagerInit(Context & ctx, int myID)
{
	if (myID > -1) {
		// GPU
		auto const GridWidth = ctx.meta.info.width.row;

		auto & myMem = ctx.dataManagerCtx[myID];

		ExecutionManagerContext myCtx;

		cudaSetDevice(myID);
		myCtx.lookup.G0.byte   = sizeof(Lookup) * GridWidth;
		myCtx.lookup.G0.ptr	   = (Lookup *)myMem.buddy->allocate(myCtx.lookup.G0.byte);
		myCtx.lookup.G2.byte   = sizeof(Lookup) * GridWidth;
		myCtx.lookup.G2.ptr	   = (Lookup *)myMem.buddy->allocate(myCtx.lookup.G2.byte);
		myCtx.lookup.temp.byte = sizeof(Lookup) * GridWidth;
		myCtx.lookup.temp.ptr  = (Lookup *)myMem.buddy->allocate(myCtx.lookup.temp.byte);

		cudaSetDevice(myID);
		cudaMemset(myCtx.lookup.temp.ptr, 0, myCtx.lookup.temp.byte);
		cudaMemset(myCtx.lookup.G0.ptr, 0, myCtx.lookup.G0.byte);
		cudaMemset(myCtx.lookup.G2.ptr, 0, myCtx.lookup.G2.byte);

		cub::DeviceScan::ExclusiveSum(nullptr,
									  myCtx.cub.byte,
									  myCtx.lookup.temp.ptr,
									  myCtx.lookup.G0.ptr,
									  myCtx.lookup.G0.count());
		myCtx.cub.ptr = myMem.buddy->allocate(myCtx.cub.byte);

		myCtx.count.byte = sizeof(Count);
		myCtx.count.ptr	 = (Count *)myMem.buddy->allocate(myCtx.count.byte);

		ctx.executionManagerCtx.insert({myID, myCtx});
	} else if (myID == -1) {
		// CPU
	} else {
		// noop
	}
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
		ExecutionManagerInit(ctx, i);
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

	auto start = std::chrono::system_clock::now();

	auto exeReq = ScheduleManager(ctx);

	std::vector<ResultChanPtr> resultChan(ctx.deviceCount);

	for (int i = 0; i < ctx.deviceCount; i++) {
		DataManager(ctx, i);
		resultChan[i] = ExecutionManager(ctx, i, exeReq);
	}
	DataManager(ctx, -1);
	DataManager(ctx, -2);
	auto c = merge(resultChan);
	ScheduleWaiter(c);

	auto end = std::chrono::system_clock::now();
	std::cout << "REALTIME: " << std::chrono::duration<double>(end - start).count() << std::endl;

	return 0;
}