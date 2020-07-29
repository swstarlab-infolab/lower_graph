#include "context.h"

#include "util.h"

#include <argparse/argparse.hpp>
#include <cuda_runtime.h>
#include <iostream>

void Context::printCUDAInfo()
{
	for (size_t i = 0; i < this->cuda.size(); i++) {
		auto & prop = this->cuda[i];
		printf("----------------Device %ld----------------\n", i);
		printf("\tName                     : %s\n", prop.name);
		printf("\tOn-Board                 : %c\n", OX(!prop.integrated));
		char pciID[13];
		cudaDeviceGetPCIBusId(pciID, 13, i);
		printf("\tPCI ID                   : %s\n", pciID);
		printf("\tCompute capability       : %d.%d\n", prop.major, prop.minor);
		printf("Kernel Support Info\n");
		printf("\tKernel execution timeout : %c\n", OX(prop.kernelExecTimeoutEnabled));
		printf("\tConcurrent Kernel exec   : %c\n", OX(prop.concurrentKernels));
		printf("Memcpy Support Info\n");
		printf("\tDevice copy overlap      : %c\n", OX(prop.deviceOverlap));
		printf("\tCopy Engines             : %d\n", prop.asyncEngineCount);
		printf("Memory Support Info\n");
		printf("\tPageable mem access      : %c\n", OX(prop.pageableMemoryAccess));
		printf("\tManaged mem support      : %c\n", OX(prop.managedMemory));
		printf("\tConcurrent Managed mem   : %c\n", OX(prop.concurrentManagedAccess));
		printf("\tH->D Direct Managed mem access : %c\n", OX(prop.directManagedMemAccessFromHost));

		printf("Processor Spec\n");
		printf("\tClock rate               : %d MHz\n", prop.clockRate / 1000);
		printf("\tMPs                      : %d\n", prop.multiProcessorCount);
		printf("\tThreads in a Warp        : %d\n", prop.warpSize);
		printf("\tMax threads per Block    : %d\n", prop.maxThreadsPerBlock);
		printf("\tMax thread dimensions    : (%d, %d, %d)\n",
			   prop.maxThreadsDim[0],
			   prop.maxThreadsDim[1],
			   prop.maxThreadsDim[2]);
		printf("\tMax grid dimensions      : (%d, %d, %d)\n",
			   prop.maxGridSize[0],
			   prop.maxGridSize[1],
			   prop.maxGridSize[2]);

		printf("Memory Spec\n");
		printf("\tMemory Bus Width         : %d bit\n", prop.memoryBusWidth);
		printf("\tMemory Clock Rate        : %d MHz\n", prop.memoryClockRate / 1000);
		printf("\tECC support              : %c\n", OX(prop.ECCEnabled));
		printf("\tTotal global mem         : %s\n", SIUnit(prop.totalGlobalMem).c_str());
		printf("\tTotal constant mem       : %s\n", SIUnit(prop.totalConstMem).c_str());
		printf("\tShared mem per a Block   : %s\n", SIUnit(prop.sharedMemPerBlock).c_str());
		printf("\tRegisters per a Block    : %s\n", SIUnit(prop.regsPerBlock).c_str());
		printf("\tMax mem pitch            : %s\n", SIUnit(prop.memPitch).c_str());
		printf("\tTexture Alignment        : %s\n", SIUnit(prop.textureAlignment).c_str());
		printf("\n");
	}

	if (this->cuda.size() >= 2) {
		printf("Peer Access Matrix; row = source device, column = destination device\n");

		printf("   ");
		for (size_t dst = 0; dst < this->cuda.size(); dst++) {
			printf("%02ld ", dst);
		}
		printf("\n");

		for (size_t src = 0; src < this->cuda.size(); src++) {
			printf("%02ld ", src);
			for (size_t dst = 0; dst < this->cuda.size(); dst++) {
				int result;
				cudaDeviceCanAccessPeer(&result, src, dst);
				printf(" %c ", OX(result));
			}
			printf("\n");
		}
	}
}

#define CUDACHECK()                        \
	do {                                   \
		auto e = cudaGetLastError();       \
		if (e) {                           \
			printf("%s:%d, %s(%d), %s\n",  \
				   __FILE__,               \
				   __LINE__,               \
				   cudaGetErrorName(e),    \
				   e,                      \
				   cudaGetErrorString(e)); \
			cudaDeviceReset();             \
			exit(EXIT_FAILURE);            \
		}                                  \
	} while (false)

void Context::parse(int argc, char * argv[])
{
	argparse::ArgumentParser args(argv[0]);
	args.add_argument("-vv", "--verbose")
		.help("set verbose output")
		.default_value(false)
		.implicit_value(true);

	args.add_argument("-i", "--input")
		.required()
		.help("Set a path of input data folder ")
		.action([&](const std::string & value) { this->inFolder = value; });

	args.add_argument("-s", "--streams")
		.required()
		.help("# of CUDA streams per each GPU ")
		.action([&](const std::string & value) {
			this->streams = std::strtol(value.c_str(), nullptr, 10);
		});

	args.add_argument("-b", "--blocks")
		.required()
		.help("# of CUDA blocks per each stream ")
		.action([&](const std::string & value) {
			this->blocks = std::strtol(value.c_str(), nullptr, 10);
		});

	args.add_argument("-t", "--threads")
		.required()
		.help("# of CUDA threads per each block ")
		.action([&](const std::string & value) {
			this->threads = std::strtol(value.c_str(), nullptr, 10);
		});

	try {
		args.parse_args(argc, argv);
	} catch (const std::runtime_error & err) {
		std::cerr << err.what() << std::endl;
		std::cerr << args;
		exit(EXIT_FAILURE);
	}

	this->verbose = (args["--verbose"] == true);

	int devices = 0;
	cudaGetDeviceCount(&devices);
	CUDACHECK();
	this->cuda.resize(devices);
	for (size_t i = 0; i < this->cuda.size(); i++) {
		cudaSetDevice(i);
		CUDACHECK();
		cudaDeviceReset();
		CUDACHECK();
		cudaGetDeviceProperties(&this->cuda[i], i);
		CUDACHECK();
	}

	for (size_t src = 0; src < this->cuda.size(); src++) {
		for (size_t dst = 0; dst < this->cuda.size(); dst++) {
			int result;
			cudaDeviceCanAccessPeer(&result, src, dst);
			CUDACHECK();
			if (result) {
				cudaSetDevice(src);
				CUDACHECK();
				cudaDeviceEnablePeerAccess(dst, 0);
				CUDACHECK();
			}
		}
	}
}