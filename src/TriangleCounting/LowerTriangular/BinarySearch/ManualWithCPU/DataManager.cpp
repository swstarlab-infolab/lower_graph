#include "DataManager.h"

#include <cuda_runtime.h>

void DataManager::init(int const ID, sp<DataManager> upstream)
{
	this->ID	   = ID;
	this->upstream = upstream;

	if (this->ID > -1) {
		this->initGPU();
	} else if (this->ID == -1) {
		this->initCPU();
	} else {
		this->initStorage();
	}
}

void DataManager::run() {}

void DataManager::initCPU()
{
	size_t freeMem = (1L << 37); // 128GB

	this->mem.buf.byte = freeMem;
	cudaMallocHost((void **)&this->mem.buf.ptr, this->mem.buf.byte);
	this->buddy.init(memrgn_t{this->mem.buf.ptr, this->mem.buf.byte}, 8, 1);
}

void DataManager::initGPU()
{
	size_t freeMem;
	cudaSetDevice(this->ID);
	cudaMemGetInfo(&freeMem, nullptr);
	freeMem -= (1L << 29);

	cudaSetDevice(this->ID);
	this->mem.buf.byte = freeMem;
	cudaMalloc((void **)&this->mem.buf.ptr, this->mem.buf.byte);
	this->buddy.init(memrgn_t{this->mem.buf.ptr, this->mem.buf.byte}, 256, 1);
}

void DataManager::initStorage()
{
	// dd
}

void * DataManager::manualAlloc(size_t const byte) { return this->buddy.allocate(byte); }