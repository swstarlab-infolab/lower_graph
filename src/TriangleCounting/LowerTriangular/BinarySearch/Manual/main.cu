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
	fprintf(stdout, "\n");

	// get total GPUs
	cudaGetDeviceCount(&ctx.deviceCount);

	//  0   : CPU
	//  1~ N: GPU
	// -1~-N: Storage
	for (int32_t i = 1; i <= ctx.deviceCount; i++) {
		typename Context::Connections c;
		c.upstream = 0;
		for (int32_t j = 1; j <= ctx.deviceCount; j++) {
			if (i != j) {
				c.neighbor.push_back(j);
			}
		}
		ctx.conn[i] = c;
	}

	// CPU
	ctx.conn[0] = {-1, {}};

	// SSD
	ctx.conn[-1] = {-1, {}};

	using DataChanType = bchan<Tx<DataMethod, MemInfo>>;
	using FileChanType = bchan<Tx<DataMethod, FileInfo>>;

	for (int32_t i = 1; i <= ctx.deviceCount; i++) {
		ctx.memChan[i] = std::make_shared<DataChanType>(16);
	}
	ctx.memChan[0]	 = std::make_shared<DataChanType>(16);
	ctx.fileChan[-1] = std::make_shared<FileChanType>(16);
}

int main(int argc, char * argv[])
{
	using DataTxChan	= bchan<Tx<DataMethod, MemInfo>>;
	using DataTxChanPtr = std::shared_ptr<DataTxChan>;
	using ResultChanPtr = std::shared_ptr<bchan<CommandResult>>;

	Context ctx;
	init(ctx, argc, argv);

	auto					   exeReq = ScheduleManager(ctx);
	std::vector<DataTxChanPtr> dataChan(ctx.deviceCount);
	std::vector<ResultChanPtr> resultChan(ctx.deviceCount);

	for (int i = 0; i < ctx.deviceCount; i++) {
		dataChan[i] = std::make_shared<DataTxChan>(16);
		DataManager(ctx, i + 1);
		resultChan[i] = Computation(ctx, i + 1, exeReq);
	}
	auto c = merge(resultChan);
	ScheduleWaiter(c);

	return 0;
}