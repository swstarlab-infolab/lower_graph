#include "DataManager.cuh"
#include "ExecutionManager.cuh"
#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"

#include <GridCSR/GridCSR.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

static uint32_t findMaxGridIndex(fs::path const & folder, std::string const & ext)
{
	uint32_t max = 0;
	bool	 ok	 = false;
	for (fs::recursive_directory_iterator iter(folder), end; iter != end; iter++) {
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
			if (ext != "" && iter->path().extension() != ext) {
				continue;
			}

			auto in = iter->path().stem().string();

			uint32_t temp[2]  = {0, 0};
			auto	 delimPos = in.find("-");
			temp[0]			  = atoi(in.substr(0, delimPos).c_str());
			temp[1]			  = atoi(in.substr(delimPos + 1, in.size()).c_str());

			max = (temp[0] > max) ? temp[0] : max;
			max = (temp[1] > max) ? temp[1] : max;

			ok = true;
		}
	}

	if (ok) {
		return max;
	} else {
		throw std::runtime_error("No grid file");
	}
}

static void DataManagerInit(Context & ctx, int myID)
{

	using DataChanType = bchan<Tx>;

	auto & myCtx = ctx.dataManagerCtx[myID];

	printf("Start to initialize Device: %d\n", myID);
	if (myID > -1) {
		// GPU Memory
		size_t freeMem;
		cudaSetDevice(myID);
		cudaMemGetInfo(&freeMem, nullptr);
		freeMem -= (1L << 29);
		cudaSetDevice(myID);
		myCtx.buf	= allocCUDA<void>(freeMem);
		myCtx.buddy = std::make_shared<portable_buddy_system>();
		myCtx.buddy.get()->init(memrgn_t{myCtx.buf.get(), freeMem}, 256, 1);
		myCtx.conn				   = std::make_shared<DataManagerContext::Connections>();
		myCtx.conn.get()->upstream = -1;

		myCtx.chan	= std::make_shared<bchan<Tx>>(16);
		myCtx.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myCtx.cacheMtx = std::make_shared<std::mutex>();

		cudaSetDevice(myID);
		cudaStreamCreate(&myCtx.stream);
	} else if (myID == -1) {
		// CPU Memory
		size_t freeMem = (1L << 37) - (1L << 35); // 128GB
		// size_t freeMem = (1L << 35); // 32GB
		myCtx.buf	= allocHost<void>(freeMem);
		myCtx.buddy = std::make_shared<portable_buddy_system>();
		myCtx.buddy->init(memrgn_t{myCtx.buf.get(), freeMem}, 8, 1);
		myCtx.conn				   = std::make_shared<DataManagerContext::Connections>();
		myCtx.conn.get()->upstream = -2;
		myCtx.chan				   = std::make_shared<bchan<Tx>>(16);
		myCtx.cache = std::make_shared<DataManagerContext::Cache>(1L << 24); //, KeyHash, KeyEqual);
		myCtx.cacheMtx = std::make_shared<std::mutex>();
	} else {
		// Storage
		myCtx.conn				   = std::make_shared<DataManagerContext::Connections>();
		myCtx.conn.get()->upstream = -2;
		myCtx.chan				   = std::make_shared<bchan<Tx>>(16);
	}
}

static void ExecutionManagerInit(Context & ctx, int myID)
{
	if (myID > -1) {
		// GPU

		auto const GridWidth = ctx.grid.width;

		auto & myMem = ctx.dataManagerCtx[myID];

		ExecutionManagerContext myCtx;

		myCtx.my.resize(ctx.setting[0]);
		for (auto & c : myCtx.my) {
			cudaSetDevice(myID);
			cudaStreamCreate(&c.stream);
			auto e = cudaStreamQuery(c.stream);
			printf("DEVICE: %d, STREAM: %p, %s(%d), %s\n",
				   myID,
				   c.stream,
				   cudaGetErrorName(e),
				   e,
				   cudaGetErrorString(e));

			cudaSetDevice(myID);
			c.lookup.G0.byte   = sizeof(Lookup) * GridWidth;
			c.lookup.G0.ptr	   = (Lookup *)myMem.buddy->allocate(c.lookup.G0.byte);
			c.lookup.G2.byte   = sizeof(Lookup) * GridWidth;
			c.lookup.G2.ptr	   = (Lookup *)myMem.buddy->allocate(c.lookup.G2.byte);
			c.lookup.temp.byte = sizeof(Lookup) * GridWidth;
			c.lookup.temp.ptr  = (Lookup *)myMem.buddy->allocate(c.lookup.temp.byte);

			cudaSetDevice(myID);
			cudaMemset(c.lookup.temp.ptr, 0, c.lookup.temp.byte);
			cudaMemset(c.lookup.G0.ptr, 0, c.lookup.G0.byte);
			cudaMemset(c.lookup.G2.ptr, 0, c.lookup.G2.byte);

			cudaSetDevice(myID);
			cub::DeviceScan::ExclusiveSum(
				nullptr, c.cub.byte, c.lookup.temp.ptr, c.lookup.G0.ptr, c.lookup.G0.count());
			c.cub.ptr = myMem.buddy->allocate(c.cub.byte);

			c.count.byte = sizeof(Count);
			c.count.ptr	 = (Count *)myMem.buddy->allocate(c.count.byte);
		}

		ctx.executionManagerCtx.insert({myID, myCtx});
	} else if (myID == -1) {
		/*
		// CPU
		auto const GridWidth = ctx.grid.width;

		auto & myMem = ctx.dataManagerCtx[myID];

		ExecutionManagerContext myCtx;
		myCtx.lookup.G0.byte   = sizeof(Lookup) * GridWidth;
		myCtx.lookup.G0.ptr	   = (Lookup *)myMem.buddy->allocate(myCtx.lookup.G0.byte);
		myCtx.lookup.G2.byte   = sizeof(Lookup) * GridWidth;
		myCtx.lookup.G2.ptr	   = (Lookup *)myMem.buddy->allocate(myCtx.lookup.G2.byte);
		myCtx.lookup.temp.byte = sizeof(Lookup) * GridWidth;
		myCtx.lookup.temp.ptr  = (Lookup *)myMem.buddy->allocate(myCtx.lookup.temp.byte);

		memset(myCtx.lookup.temp.ptr, 0, myCtx.lookup.temp.byte);
		memset(myCtx.lookup.G0.ptr, 0, myCtx.lookup.G0.byte);
		memset(myCtx.lookup.G2.ptr, 0, myCtx.lookup.G2.byte);

		myCtx.count.byte = sizeof(Count);
		myCtx.count.ptr	 = (Count *)myMem.buddy->allocate(myCtx.count.byte);

		ctx.executionManagerCtx.insert({myID, myCtx});
		*/
	}
}

static void init(Context & ctx, int argc, char * argv[])
{
	// Argument
	if (argc != 5) {
		fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	ctx.folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	ctx.grid.count = findMaxGridIndex(ctx.folderPath, ".row") + 1;
	ctx.grid.width = (1 << 24);

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
