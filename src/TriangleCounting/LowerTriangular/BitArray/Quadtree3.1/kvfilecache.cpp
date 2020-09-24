#include "kvfilecache.h"

#include <fcntl.h>
#include <unistd.h>

bool KeyValueFileCache::tryAlloc(int const device_id, void ** addr, size_t byte)
{
	if (device_id < 0) {
		*addr = malloc(byte);
		if (addr == nullptr) {
			return false;
		}
	} else {
		try {
			*addr = this->device_pool[device_id]->allocate(byte);
		} catch (rmm::bad_alloc e) {
			return false;
		}
	}
	return true;
}

void KeyValueFileCache::mustDealloc(int const device_id, void * addr, size_t byte)
{
	if (device_id < 0) {
		free(addr);
	} else {
		this->device_pool[device_id]->deallocate(addr, byte);
	}
}

void KeyValueFileCache::loadSSDtoCPU(fs::path const & path, void * to, size_t byte)
{
	// cpu
	auto const __CDEF = 1UL << 26;

	auto fp = open64(path.c_str(), O_RDONLY);

	uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
	uint64_t offset	   = 0;

	while (offset < byte) {
		chunkSize = (byte - offset > chunkSize) ? chunkSize : byte - offset;
		auto b	  = read(fp, &(((uint8_t *)to)[offset]), chunkSize);
		offset += b;
	}

	close(fp);
};

// utilize OS's page cache
void KeyValueFileCache::loadSSDtoGPU(int const		  device_id,
									 fs::path const & path,
									 void *			  to,
									 size_t			  byte)
{
	// cpu
	auto const __CDEF = 1UL << 26;

	auto	  fp   = open64(path.c_str(), O_RDONLY);
	uint8_t * temp = (uint8_t *)malloc(byte);

	uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
	uint64_t offset	   = 0;

	cudaSetDevice(device_id);

	// overlapped loading
	while (offset < byte) {
		chunkSize = (byte - offset > chunkSize) ? chunkSize : byte - offset;

		void * cpuPtr = &(((uint8_t *)temp)[offset]);
		void * gpuPtr = &(((uint8_t *)to)[offset]);

		// read chunk from SSD to CPU
		auto b = read(fp, cpuPtr, chunkSize);

		// read chunk from CPU to GPU
		// asynchronous
		cudaMemcpyAsync(
			gpuPtr, cpuPtr, chunkSize, cudaMemcpyHostToDevice, this->cudaLoadingStream[device_id]);

		offset += b;
	}

	close(fp);

	cudaStreamSynchronize(this->cudaLoadingStream[device_id]);

	free(temp);
};

void KeyValueFileCache::init(GridInfo & gridInfo)
{
	int devices = 0;
	cudaGetDeviceCount(&devices);
	this->fileInfo.resize(devices + 1);
	for (auto & f : this->fileInfo) {
		f = std::make_shared<HashMapType>(1UL << 12);
	}

	for (size_t i = 0; i < this->fileInfo.size(); i++) {
		// printf("gridInfo.hashmap.size()=%ld\n", gridInfo.hashmap.size());
		for (auto const & kv : gridInfo.hashmap) {
			// printf("%d ", j);
			for (uint32_t fileType = 0; fileType < 3; fileType++) {
				FileInfoValue value;
				value.path	= kv.second->path[fileType];
				value.byte	= kv.second->byte[fileType];
				value.state = FileState::notexist;

				(*this->fileInfo[i])[DataManagerKey{kv.first, fileType}] = value;
				printf("%s, %ld\n", value.path.c_str(), value.byte);
			}
			// j++;
		}
	}

	this->cudaLoadingStream.resize(devices);
	for (int i = 0; i < devices; i++) {
		cudaStreamCreate(&this->cudaLoadingStream[i]);
	}
}

KeyValueFileCache::~KeyValueFileCache() noexcept
{
	for (auto & s : this->cudaLoadingStream) {
		cudaStreamDestroy(s);
	}
}

DataInfo KeyValueFileCache::mustPrepare(int device_id, DataManagerKey const & key)
{

	auto & target = this->fileInfo[device_id + 1]->at(key);

	std::lock_guard<std::mutex> lg(target.lock);

	DataInfo result;

	if (target.state == FileState::exist) {
		target.refCount++;
		result.addr = target.addr;
		result.byte = target.byte;
		return result;
	}

	while (!this->tryAlloc(device_id, &target.addr, target.byte)) {
		for (auto & kv : *this->fileInfo[device_id + 1]) {
			auto & evictTarget = kv.second;

			std::lock_guard<std::mutex> lg(evictTarget.lock);

			if (evictTarget.refCount == 0) {
				this->mustDealloc(device_id, evictTarget.addr, evictTarget.byte);

				evictTarget.state = FileState::notexist;
				evictTarget.addr  = nullptr;
				break;
			}
		}
	}

	if (device_id < 0) {
		this->loadSSDtoCPU(target.path, target.addr, target.byte);
	} else {
		this->loadSSDtoGPU(device_id, target.path, target.addr, target.byte);
	}

	target.state	= FileState::exist;
	target.refCount = 1;

	result.addr = target.addr;
	result.byte = target.byte;

	return result;
}

void KeyValueFileCache::done(int device_id, DataManagerKey const & key)
{
	auto & target = this->fileInfo[device_id + 1]->at(key);

	std::lock_guard<std::mutex> lg(target.lock);

	if (target.state == FileState::exist) {
		target.refCount--;
	}
}
