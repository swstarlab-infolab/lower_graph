#include "context.h"

#include "util.h"

#include <cuda_runtime.h>

Context ctx; // Global variable

uint32_t Context::findMaxGridIndex(std::string const & ext)
{
	uint32_t max = 0;
	bool	 ok	 = false;
	for (fs::recursive_directory_iterator iter(this->folder), end; iter != end; iter++) {
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
			if (ext != "" && iter->path().extension() != ext) {
				continue;
			}

			auto temp = filenameDecode(iter->path().stem().string());

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

void Context::init(int argc, char * argv[])
{
	if (argc != 5) {
		fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	this->folder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

	// set threshold
	this->threshold = (1 << 25); // 32MiB
	this->chanSz	= (1 << 4);	 // 16

	// ctx.grid
	this->grid.width = (1 << 24);
	this->grid.count = this->findMaxGridIndex(".row") + 1;
	std::cout << "this->grid.count: " << this->grid.count << std::endl;

	// ctx.gpu
	cudaGetDeviceCount(&this->gpu.devices);
	this->gpu.streams = strtol(argv[2], nullptr, 10);
	this->gpu.blocks  = strtol(argv[3], nullptr, 10);
	this->gpu.threads = strtol(argv[4], nullptr, 10);

	// ctx.chan
	this->chan.orderCPU = makeSp<bchan<Order>>(this->chanSz);
	this->chan.orderGPU = makeSp<bchan<Order>>(this->chanSz);
	for (int i = -1; i < this->gpu.devices; i++) {
		this->chan.report[i] = makeSp<bchan<Report>>(this->chanSz);
	}

	// Alloc Data Manager
	this->DM[-2] = makeSp<DataManager>();
	this->DM[-2]->init(-2, nullptr);

	this->DM[-1] = makeSp<DataManager>();
	this->DM[-1]->init(-1, this->DM[-2]);

	for (int i = 0; i < this->gpu.devices; i++) {
		this->DM[i] = makeSp<DataManager>();
		this->DM[i]->init(i, this->DM[-1]);
	}

	// Alloc Execution Manager
	for (int i = -1; i < this->gpu.devices; i++) {
		this->EM[i] = makeSp<ExecutionManager>();
		this->EM[i]->init(i, this->DM[i]);
	}

	// Alloc Schedule Manager
	this->SM = makeSp<ScheduleManager>();
}

void Context::finalize()
{
	for (int i = -2; i < this->gpu.devices; i++) {
		this->DM[i]->closeAllChan();
	}
}