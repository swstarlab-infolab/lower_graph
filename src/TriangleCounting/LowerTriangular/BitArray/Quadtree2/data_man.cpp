#include "data_man.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <thread>

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

#define USE_GDRCOPY

namespace Data
{

Manager::Manager(int const deviceID, fs::path const & folderPath)
	: deviceID(deviceID), folderPath(folderPath)
{
	printf("Constructor: Data::Manager, deviceId=%d\n", this->deviceID);

	// Allocate Memory
	cudaSetDevice(this->deviceID);
	CUDACHECK();
	size_t totalMem = 0;
	cudaMemGetInfo(&this->mem.byte, &totalMem);
	CUDACHECK();
	this->mem.byte -= (1L << 30);

#ifdef USE_GDRCOPY
	cudaMalloc((void **)&this->mem.devPtr, this->mem.byte);
#else
	cudaMallocManaged((void **)&this->mem.devPtr, this->mem.byte);
#endif
	CUDACHECK();
	assert(this->mem.devPtr != 0);
	printf("Data::Manager: %d, Memory: %p, Size: %ld/%ld\n",
		   this->deviceID,
		   (void *)this->mem.devPtr,
		   this->mem.byte,
		   totalMem);

	// Register to GPU Direct RDMA Driver
#ifdef USE_GDRCOPY
	this->mem.gdr.g = gdr_open();
	gdr_pin_buffer(this->mem.gdr.g, this->mem.devPtr, this->mem.byte, 0, 0, &this->mem.gdr.mh);

	// Map PCIe BAR address to User space address
	gdr_map(this->mem.gdr.g, this->mem.gdr.mh, &this->mem.mapPtr, this->mem.byte);
#endif
	this->mem.buddy = std::make_shared<portable_buddy_system>(
		memrgn_t{(void *)this->mem.devPtr, this->mem.byte}, 256, 1);

	this->mem.cache.clear();

	this->reqQ		   = std::make_shared<boost::fibers::buffered_channel<Req>>(1 << 4);
	this->doneQ		   = std::make_shared<boost::fibers::buffered_channel<Req>>(1 << 4);
	this->reqMustAlloc = std::make_shared<boost::fibers::buffered_channel<Req>>(1 << 4);
}

Manager::~Manager()
{
	printf("Destructor: Data::Manager\n");

	if (!this->reqQ->is_closed()) {
		this->reqQ->close();
	}

	if (!this->doneQ->is_closed()) {
		this->doneQ->close();
	}

	if (!this->reqMustAlloc->is_closed()) {
		this->reqMustAlloc->close();
	}

	// Unmap memory
	cudaSetDevice(this->deviceID);
	CUDACHECK();
#ifdef USE_GDRCOPY
	gdr_unmap(this->mem.gdr.g, this->mem.gdr.mh, this->mem.mapPtr, this->mem.byte);

	// Unpin memory
	gdr_unpin_buffer(this->mem.gdr.g, this->mem.gdr.mh);
	gdr_close(this->mem.gdr.g);
#endif

	// Free device memory
	cudaFree((void *)this->mem.devPtr);
	CUDACHECK();

	// Reset device
	cudaDeviceReset();
	CUDACHECK();
}

void * Manager::alloc(size_t const size) { return this->mem.buddy->allocate(size); }

MemInfo Manager::load(Key const & key)
{
	Req req;
	req.key = key;
	req.cb	= std::make_shared<boost::fibers::buffered_channel<MemInfo>>(2);

	this->reqQ->push(req);

	for (auto & val : *req.cb) {
		return val;
	}

	throw "Error";
}

void Manager::done(Key const & key)
{
	Req req;
	req.key = key;

	this->doneQ->push(req);
}

void Manager::run()
{
	std::thread([=] {
		for (auto & tx : *this->doneQ) {
			std::lock_guard<std::mutex> lg(this->mem.cacheMtx);
			this->mem.cache.at(tx.key).refCnt -= 1;
			// printf("Done Data::Manager=%d, targetPath: %s\n", this->deviceID, tx.key.c_str());
		}
	}).detach();

	std::thread([=] {
		for (auto & tx : *this->reqMustAlloc) {
			auto targetPath = fs::path(this->folderPath / fs::path(tx.key));

			std::unique_lock<std::mutex> ul(this->mem.cacheMtx);
			bool						 iHaveLock = true;

			MemInfo myInfo;
			if (this->mem.cache.find(tx.key) != this->mem.cache.end()) {
				myInfo = this->mem.cache.at(tx.key).info;
				this->mem.cache.at(tx.key).refCnt += 1;
				// printf( "CacheHit2 Data::Manager=%d, targetPath: %s\n", this->deviceID,
				// tx.key.c_str());
				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				tx.cb->push(myInfo);
				tx.cb->close();

				continue;
			}

			myInfo.byte = fs::file_size(targetPath);

			while (true) {
				myInfo.ptr = this->mem.buddy->allocate(myInfo.byte);

				if (myInfo.ptr != nullptr) {
					this->mem.cache.insert({tx.key, {myInfo, 1}});
					break;
				} else {
					// allocation failure
					if (this->mem.cache.size() > 0) {
						bool evictSuccess = false;
						while (!evictSuccess) {
							if (!iHaveLock) {
								ul.lock();
								iHaveLock = true;
							}

							for (auto it = this->mem.cache.begin(); it != this->mem.cache.end();) {
								if (it->second.refCnt == 0) {
									this->mem.buddy->deallocate(it->second.info.ptr);
									it			 = this->mem.cache.erase(it);
									evictSuccess = true;
									// problem
									// printf("Data::Manager=%d, Evict %s\n", this->deviceID,
									// it->first.c_str());
									break;
								} else {
									++it;
								}
							}

							if (iHaveLock) {
								ul.unlock();
								iHaveLock = false;
							}
						}
					} else {
						throw "Strange Error";
					}
				}
			}

			auto fp = open64(targetPath.c_str(), O_RDONLY);

			constexpr uint64_t cDef		 = (1L << 30); // chunk Default
			uint64_t		   chunkByte = (myInfo.byte < cDef) ? myInfo.byte : cDef;
			uint64_t		   bytePos	 = 0;
#ifdef USE_GDRCOPY
			uint8_t * hostVisiblePtr =
				(uint8_t *)((int64_t)myInfo.ptr -
							((int64_t)this->mem.devPtr - (int64_t)this->mem.mapPtr));
#endif

			// printf("hostVisiblePtr=%p, devVisiblePtr=%p\n", hostVisiblePtr, myInfo.ptr);

			while (bytePos < myInfo.byte) {
				chunkByte = (myInfo.byte - bytePos > chunkByte) ? chunkByte : myInfo.byte - bytePos;
#ifdef USE_GDRCOPY
				auto loaded = read(fp, &(((uint8_t *)hostVisiblePtr)[bytePos]), chunkByte);
#else
				auto loaded = read(fp, &(((uint8_t *)myInfo.ptr)[bytePos]), chunkByte);
#endif
				// printf("key=%s, myInfo.ptr=%p, loaded=%d\n", tx.key.c_str(), myInfo.ptr, loaded);
				assert(loaded > -1);
				bytePos += loaded;
			}

			close(fp);

			if (iHaveLock) {
				ul.unlock();
				iHaveLock = false;
			}

			tx.cb->push(myInfo);
			tx.cb->close();

			// printf("Data::Manager=%d, targetPath: %s LoadSuccess\n", this->deviceID,
			// targetPath.c_str());
		}
	}).detach();

	std::thread([=] {
		for (auto & tx : *this->reqQ) {
			MemInfo myInfo = {
				0,
			};

			std::unique_lock<std::mutex> ul(this->mem.cacheMtx);

			if (this->mem.cache.find(tx.key) != this->mem.cache.end()) {
				myInfo = this->mem.cache.at(tx.key).info;
				this->mem.cache.at(tx.key).refCnt += 1;
				// printf( "CacheHit Data::Manager=%d, targetPath: %s\n", this->deviceID,
				// tx.key.c_str());
				ul.unlock();
				tx.cb->push(myInfo);
				tx.cb->close();
			} else {
				ul.unlock();
				this->reqMustAlloc->push(tx);
			}
		}
	}).detach();
}

} // namespace Data