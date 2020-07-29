#include "data.h"

#include "util.h"

#include <limits>
#include <sys/sysinfo.h>

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

namespace Data
{

std::unordered_map<int, sp<Manager>> managerSpace;

void init(sp<Context> ctx)
{
	for (size_t i = 0; i < ctx->cuda.size(); i++) {
		managerSpace[i] = makeSp<Manager>();
		managerSpace[i]->init(ctx, i, -1);
	}
	managerSpace[-1] = makeSp<Manager>();
	managerSpace[-1]->init(ctx, -1, std::numeric_limits<int>::min());
}

void run(sp<Context> ctx)
{
	for (size_t i = 0; i < ctx->cuda.size(); i++) {
		managerSpace[i]->run(ctx);
	}
	managerSpace[-1]->run(ctx);
}

static auto allocCUDA(size_t const byte)
{
	return sp<void>(
		[&] {
			void * p;
			cudaMalloc((void **)&p, byte);
			return p;
		}(),
		[](void * p) {
			printf("allocCUDA Free\n");
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}

static auto allocCUDAHost(size_t const byte)
{
	return sp<void>(
		[&] {
			void * p;
			cudaHostAlloc((void **)&p, byte, cudaHostAllocPortable);
			return p;
		}(),
		[](void * p) {
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}

void Manager::init(sp<Context> ctx, int const deviceID, int const upstreamDeviceID)
{
	// printf("DataManager constructor, device=%d, upstream=%d\n", deviceID, upstreamDeviceID);

	this->deviceID		   = deviceID;
	this->upstreamDeviceID = upstreamDeviceID;

	this->allocator = makeSp<portable_buddy_system>();

	if (this->deviceID >= 0) {
		size_t availByte, totalByte;
		cudaMemGetInfo(&availByte, &totalByte);
		CUDACHECK();
		// auto allocByte = availByte - (1L << 31);
		auto allocByte = (1L << 33) + (1L << 31);

		// cudaMalloc(&this->myBuffer, allocByte);
		cudaSetDevice(this->deviceID);
		this->myBuffer = allocCUDA(allocByte);
		CUDACHECK();

		printf("device=%d, alloc/left/total=%s/%s/%s\n",
			   this->deviceID,
			   SIUnit(allocByte).c_str(),
			   SIUnit(availByte).c_str(),
			   SIUnit(totalByte).c_str());
		this->allocator->init(memrgn_t{this->myBuffer.get(), allocByte}, 256, 1);

		cudaSetDevice(this->deviceID);
		CUDACHECK();
		cudaStreamCreate(&this->myStream);
		CUDACHECK();
	} else {
		struct sysinfo memInfo;
		sysinfo(&memInfo);
		// sysinfo 가 cached memory를 못알아냄
		// auto totalByte = memInfo.totalram * memInfo.mem_unit;
		// auto availByte = (memInfo.freeram + memInfo.bufferram) * memInfo.mem_unit;
		auto allocByte = 1UL << 33;

		this->myBuffer = allocCUDAHost(allocByte);
		CUDACHECK();
		/*
		printf("device=%d, alloc/left/total=%s/%s/%s\n",
			   this->deviceID,
			   SIUnit(allocByte).c_str(),
			   SIUnit(availByte).c_str(),
			   SIUnit(totalByte).c_str());
			   */
		this->allocator->init(memrgn_t{this->myBuffer.get(), allocByte}, 8, 1);
		CUDACHECK();
	}

	// printf("DataManager constructor, device=%d, upstream=%d, done\n", deviceID,
	// upstreamDeviceID);
}

void Manager::run(sp<Context> ctx)
{
	this->readyChan = this->readyInternal(ctx);
	this->doneChan	= this->doneInternal();
}

Manager::~Manager() noexcept
{
	printf("DataManager destructor: device=%d\n", deviceID);

	if (this->deviceID >= 0) {
		cudaStreamDestroy(this->myStream);
	}

	this->readyChan->close();
	this->doneChan->close();

	printf("DataManager destructor: device=%d, done\n", deviceID);
}

void * Manager::malloc(size_t const byte) { return this->allocator->allocate(byte); }
void   Manager::free(void * const ptr) { this->allocator->deallocate(ptr); }

sp<bchan<Tx>> Manager::doneInternal()
{
	auto in = makeSp<bchan<Tx>>(16);

	std::thread([=] {
		for (auto & tx : *in) {
			{
				std::lock_guard<std::mutex> lg(this->cacheMtx);
				this->cache[tx.key].refCnt -= 1;
			}

			MemInfo myInfo;
			myInfo.ptr	= nullptr;
			myInfo.path = "";
			myInfo.byte = 0;
			myInfo.hit	= true;
			myInfo.ok	= true;

			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();

	return in;
}

sp<bchan<Tx>> Manager::readyInternal(sp<Context> ctx)
{
	auto in = makeSp<bchan<Tx>>(16);

	std::thread([=] {
		for (auto & tx : *in) {
			// printf("DM%d: %s\n", this->deviceID, tx.key.c_str());
			MemInfo myInfo;
			myInfo.ptr	= nullptr;
			myInfo.path = "";
			myInfo.byte = 0;
			myInfo.hit	= false;
			myInfo.ok	= false;

			std::unique_lock<std::mutex> ul(this->cacheMtx);

			bool iHaveLock = true;

			if (this->cache.find(tx.key) != this->cache.end()) {
				// printf("DM%d: %s HIT\n", this->deviceID, tx.key.c_str());
				myInfo = this->cache[tx.key].info;
				this->cache[tx.key].refCnt += 1;

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				myInfo.hit = true;
				myInfo.ok  = true;
			} else {
				// printf("DM%d: %s MISS\n", this->deviceID, tx.key.c_str());
				// path generate
				fs::path path = ctx->inFolder / tx.key;
				myInfo.byte	  = fs::file_size(path);

				// allocate
				while (true) {
					myInfo.ptr = (uint32_t *)this->allocator->allocate(myInfo.byte);

					if (myInfo.ptr != nullptr) {
						this->cache[tx.key] = Value{myInfo, 1};
						// printf("DM%d: %s ALLOC SUCCESS\n", this->deviceID, tx.key.c_str());
						break;
					} else {
						// printf( "DM%d: %s ALLOC FAIL, TRY TO ALLOC\n", this->deviceID,
						// tx.key.c_str());
						// allocation failure
						if (this->cache.size() > 0) {
							bool evictSuccess = false;
							while (!evictSuccess) {
								// lock
								if (!iHaveLock) {
									ul.lock();
									iHaveLock = true;
								}

								for (auto it = this->cache.begin(); it != this->cache.end();) {
									if (it->second.refCnt == 0) {
										this->allocator->deallocate(it->second.info.ptr);
										it			 = this->cache.erase(it);
										evictSuccess = true;
										// printf("DM%d: %s EVICTED. YES! BUT I DONT HAVE
										// LOCK\n", this->deviceID, tx.key.c_str()); break;
									} else {
										++it;
									}
								}

								// unlock
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

				// request & memcpy
				if (this->deviceID >= 0) {
					// printf("DM%d: %s MEMCPY (%d->%d ) START\n", this->deviceID,
					// tx.key.c_str(), this->upstreamDeviceID, this->deviceID);
					auto otherInfo = managerSpace[this->upstreamDeviceID]->ready(tx.key);

					cudaSetDevice(this->deviceID);
					cudaMemcpyAsync(myInfo.ptr,
									otherInfo.ptr,
									otherInfo.byte,
									cudaMemcpyHostToDevice,
									this->myStream);
					cudaStreamSynchronize(this->myStream);

					if (iHaveLock) {
						ul.unlock();
						iHaveLock = false;
					}

					managerSpace[this->upstreamDeviceID]->done(tx.key);
					// printf("DM%d: %s MEMCPY (%d->%d ) DONE\n", this->deviceID,
					// tx.key.c_str(), this->upstreamDeviceID, this->deviceID);
				} else {
					// printf("DM%d: %s OPENFILE (Storage->%d) START\n", this->deviceID,
					// tx.key.c_str(), this->deviceID);
					auto fp = open64(path.c_str(), O_RDONLY);

					constexpr uint64_t cDef		 = (1L << 30); // chunk Default
					uint64_t		   chunkByte = (myInfo.byte < cDef) ? myInfo.byte : cDef;
					uint64_t		   bytePos	 = 0;
					while (bytePos < myInfo.byte) {
						chunkByte =
							(myInfo.byte - bytePos > chunkByte) ? chunkByte : myInfo.byte - bytePos;
						auto loaded = read(fp, &(((uint8_t *)myInfo.ptr)[bytePos]), chunkByte);
						bytePos += loaded;
					}

					close(fp);
					// printf("DM%d: %s FILELOAD (Storage->%d) DONE\n", this->deviceID,
					// tx.key.c_str(), this->deviceID);
				}

				// Done
				myInfo.hit = false;
				myInfo.ok  = true;
			}

			// printf("DM%d: %s DONE\n", this->deviceID, tx.key.c_str());
			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();

	return in;
}

MemInfo Manager::ready(Key const & key)
{
	Tx tx;
	tx.key = key;
	tx.cb  = makeSp<bchan<MemInfo>>(2);
	this->readyChan->push(tx);
	for (auto & result : *tx.cb) {
		return result;
	}
	throw "Ready Error";
}

MemInfo Manager::done(Key const & key)
{
	Tx tx;
	tx.key = key;
	tx.cb  = makeSp<bchan<MemInfo>>(2);
	this->doneChan->push(tx);
	for (auto & result : *tx.cb) {
		return result;
	}
	throw "Done Error";
}

} // namespace Data