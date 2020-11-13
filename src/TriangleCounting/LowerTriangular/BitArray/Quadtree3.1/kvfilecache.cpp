#include "kvfilecache.h"

#include "util/logging.h"

#include <fcntl.h>
#include <unistd.h>

bool KeyValueFileCache::tryAlloc(int const myDeviceID, void ** addr, size_t byte)
{
	if (myDeviceID < 0) {
		*addr = malloc(byte);
		if (addr == nullptr) {
			return false;
		}
	} else {
		try {
			cudaSetDevice(myDeviceID);
			*addr = this->device_pool[myDeviceID]->allocate(byte);
		} catch (rmm::bad_alloc e) {
			return false;
		}
	}
	return true;
}

void KeyValueFileCache::mustDealloc(int const myDeviceID, void * addr, size_t byte)
{
	if (myDeviceID < 0) {
		free(addr);
	} else {
		cudaSetDevice(myDeviceID);
		this->device_pool[myDeviceID]->deallocate(addr, byte);
	}
}

void KeyValueFileCache::loadToMe(int const				myDeviceID,
								 int const				otherDeviceID,
								 FileInfoValue const &	myInfo,
								 DataInfo<void> const & otherInfo)
{
	if (myDeviceID < 0) {
		// SSD->CPU
		auto const __CDEF = 1UL << 26;

		auto fp = open64(myInfo.path.c_str(), O_RDONLY);

		uint64_t chunkSize = (myInfo.byte < __CDEF) ? myInfo.byte : __CDEF;
		uint64_t offset	   = 0;

		while (offset < myInfo.byte) {
			chunkSize = (myInfo.byte - offset > chunkSize) ? chunkSize : myInfo.byte - offset;
			auto b	  = read(fp, &(((uint8_t *)myInfo.addr)[offset]), chunkSize);
			offset += b;
		}

		close(fp);
	} else {
		cudaMemcpyKind kind;

		if (otherDeviceID < 0) {
			// CPU -> GPU
			kind = cudaMemcpyHostToDevice;
		} else {
			// GPU->GPU
			kind = cudaMemcpyDeviceToDevice;
		}

		cudaSetDevice(myDeviceID);
		cudaMemcpyAsync(
			myInfo.addr, otherInfo.addr, myInfo.byte, kind, this->cudaLoadingStream[myDeviceID]);
		cudaStreamSynchronize(this->cudaLoadingStream[myDeviceID]);
	}
}

bool KeyValueFileCache::tryPrepareNVLink(int const				otherDeviceID,
										 DataManagerKey const & key,
										 DataInfo<void> &		info)
{
	auto & target = this->fileInfo[otherDeviceID + 1]->at(key);

	std::lock_guard<std::mutex> lg(target.lock);

	if (target.state == FileState::exist) {
		target.refCount++;
		info.addr = target.addr;
		info.byte = target.byte;

		return true;
	}

	return false;
}

void KeyValueFileCache::init(GridInfo const & gridInfo)
{
	cudaGetDeviceCount(&this->devices);
	this->fileInfo.resize(this->devices + 1);
	for (auto & f : this->fileInfo) {
		f = std::make_shared<HashMapType>(1UL << 12);
	}

	for (size_t i = 0; i < this->fileInfo.size(); i++) {
		for (auto const & kv : gridInfo.hashmap) {
			for (uint32_t fileType = 0; fileType < 3; fileType++) {
				FileInfoValue value;
				value.path	= kv.second->path[fileType];
				value.byte	= kv.second->byte[fileType];
				value.state = FileState::notexist;

				(*this->fileInfo[i])[DataManagerKey{kv.first, fileType}] = value;
			}
		}
	}

	for (int i = 0; i < this->devices; i++) {
		for (int j = 0; j < this->devices; j++) {
			if (i == j) {
				continue;
			}
			int canAccess;
			cudaDeviceCanAccessPeer(&canAccess, i, j);
			if (canAccess) {
				cudaSetDevice(i);
				cudaDeviceEnablePeerAccess(j, 0);
			}
		}
	}

	this->device_pool.resize(devices);
	this->cudaLoadingStream.resize(devices);

	for (int i = 0; i < devices; i++) {
		cudaSetDevice(i);
		cudaDeviceReset();

		cudaSetDevice(i);
		size_t free;
		cudaMemGetInfo(&free, nullptr);
		free -= (1UL << 29);

		this->device_pool[i] = std::make_shared<DevicePoolType>(
			rmm::mr::get_per_device_resource(rmm::cuda_device_id(i)), free, free);
		rmm::mr::set_per_device_resource(rmm::cuda_device_id(i), &(*(this->device_pool[i])));

		cudaStreamCreate(&this->cudaLoadingStream[i]);
	}
}

KeyValueFileCache::~KeyValueFileCache() noexcept
{
	int device = 0;
	for (auto & s : this->cudaLoadingStream) {
		cudaSetDevice(device);
		cudaStreamDestroy(s);
		device++;
	}
}

bool KeyValueFileCache::refCountUpForExist(FileInfoValue & target, DataInfo<void> & result)
{
	std::lock_guard<std::mutex> lg(target.lock);

	if (target.state == FileState::exist) {
		target.refCount++;

		result.addr = target.addr;
		result.byte = target.byte;

		return true;
	}

	return false;
}

bool KeyValueFileCache::changeState(FileInfoValue & target,
									FileState const from,
									FileState const to)
{
	std::lock_guard<std::mutex> lg(target.lock);

	if (target.state == from) {
		target.state = to;
		return true;
	}

	return false;
}

DataInfo<void> KeyValueFileCache::mustPrepare(int myDeviceID, DataManagerKey const & key)
{
	auto & target = this->fileInfo[myDeviceID + 1]->at(key);

	DataInfo<void> myInfo;

	// FileState = notexist, loading, exist, evicting
	if (this->refCountUpForExist(target, myInfo)) {
		return myInfo;
	}

	// FileState = notexist, loading, evicting
	if (!this->changeState(target, FileState::notexist, FileState::loading)) {
		while (true) {
			if (this->refCountUpForExist(target, myInfo)) {
				return myInfo;
			}
		}
	}

	// FileState = loading, evicting
	while (![&] {
		std::lock_guard<std::mutex> lg(target.lock);
		return this->tryAlloc(myDeviceID, &target.addr, target.byte);
	}()) {
		for (auto & kv : *this->fileInfo[myDeviceID + 1]) {
			auto & evictTarget = kv.second;

			bool success = false;
			{
				std::lock_guard<std::mutex> lg(evictTarget.lock);
				if (evictTarget.state == FileState::exist && evictTarget.refCount == 0) {
					evictTarget.state = FileState::evicting;
					success			  = true;
				}
			}

			if (success) {
				std::lock_guard<std::mutex> lg(evictTarget.lock);

				this->mustDealloc(myDeviceID, evictTarget.addr, evictTarget.byte);

				evictTarget.state = FileState::notexist;
				evictTarget.addr  = nullptr;

				break;
			}
		}
	}

	DataInfo<void> otherInfo;
	int			   otherDeviceID = -2;
	if (myDeviceID < 0) {
		// SSD->CPU
		this->loadToMe(myDeviceID, otherDeviceID, target, otherInfo);
	} else {
		// search other GPU

		otherDeviceID = (myDeviceID + 1) % this->devices;

		int	 accum	 = 0;
		bool success = false;
		while (accum < this->devices) {
			if (otherDeviceID == myDeviceID) {
				continue;
			}

			if (otherDeviceID >= this->devices) {
				otherDeviceID = otherDeviceID % this->devices;
			}

			if (this->tryPrepareNVLink(otherDeviceID, key, otherInfo)) {
				success = true;
				break;
			}

			accum++;
		}

		// if failed
		if (!success) {
			// CPU->GPU
			otherDeviceID = -1;
			otherInfo	  = this->mustPrepare(otherDeviceID, key);
		}

		this->loadToMe(myDeviceID, otherDeviceID, target, otherInfo);
		/*
		if (success) {
			LOGF("[%d -> %d] <%d, %d>, %ld",
				 otherDeviceID,
				 myDeviceID,
				 key.gridID,
				 key.fileType,
				 target.byte);
		}
		*/
		this->done(otherDeviceID, key);
	}

	{
		std::lock_guard<std::mutex> lg(target.lock);
		target.state	= FileState::exist;
		target.refCount = 1;

		myInfo.addr = target.addr;
		myInfo.byte = target.byte;
	}

	return myInfo;
}

void KeyValueFileCache::done(int const myDeviceID, DataManagerKey const & key)
{
	auto & target = this->fileInfo[myDeviceID + 1]->at(key);

	std::lock_guard<std::mutex> lg(target.lock);

	if (target.state == FileState::exist) {
		target.refCount--;
	}
}
