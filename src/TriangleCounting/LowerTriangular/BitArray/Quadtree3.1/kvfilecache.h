#ifndef D386761A_51A2_4858_ABA2_F75F105E1654
#define D386761A_51A2_4858_ABA2_F75F105E1654

#include "base/shard.h"
#include "base/type.h"
#include "gridinfo.h"

#include <array>
#include <cuda_runtime.h>
#include <jemalloc/jemalloc.h>
#include <memory>
#include <mutex>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <unordered_map>

// device_id < 0: CPU
// device_id >= 0: GPU
// grid_id, file_type
struct DataManagerKey {
	uint32_t gridID, fileType;

	bool operator==(DataManagerKey const & k) const
	{
		return this->gridID == k.gridID && this->fileType == k.fileType;
	}
};

// read-only cache, no write supported
class KeyValueFileCache
{
private:
	enum FileState { notexist, loading, exist, evicting };

	struct FileInfoValue {
		std::mutex lock;

		FileState state;
		size_t	  refCount;
		void *	  addr;
		size_t	  byte;
		fs::path  path;

		FileInfoValue & operator=(FileInfoValue const & copy)
		{
			this->state	   = copy.state;
			this->refCount = copy.refCount;
			this->byte	   = copy.byte;
			this->path	   = copy.path;

			return *this;
		}
	};

	// HashMap
	struct MyHash {
		size_t operator()(DataManagerKey const & k) const
		{
			return std::hash<uint64_t>()(((uint64_t)k.gridID << 32) + (uint64_t)k.fileType);
		}
	};

	using HashMapType = std::unordered_map<DataManagerKey, FileInfoValue, MyHash>;
	std::vector<std::shared_ptr<HashMapType>> fileInfo;

	// DevicePool
	using DevicePoolType = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
	std::vector<std::shared_ptr<DevicePoolType>> device_pool;

	// DeviceStreams
	std::vector<cudaStream_t> cudaLoadingStream;

	// hashmap functions
	bool refCountUpForExist(FileInfoValue & target, DataInfo<void> & result);
	bool changeState(FileInfoValue & target, FileState const from, FileState const to);

	// hashmap function: for GPU-GPU
	bool
	tryPrepareNVLink(int const otherDeviceID, DataManagerKey const & key, DataInfo<void> & info);

	// hashmap function: for my device
	bool tryAlloc(int const myDeviceID, void ** addr, size_t byte);
	void mustDealloc(int const myDeviceID, void * addr, size_t byte);

	// loading function
	void loadToMe(int const				 myDeviceID,
				  int const				 otherDeviceID,
				  FileInfoValue const &	 myInfo,
				  DataInfo<void> const & otherInfo);

public:
	int devices;

	void init(GridInfo const & gridInfo);
	~KeyValueFileCache() noexcept;

	DataInfo<void> mustPrepare(int const myDeviceID, DataManagerKey const & key);
	void		   done(int const myDeviceID, DataManagerKey const & key);
};

#endif /* D386761A_51A2_4858_ABA2_F75F105E1654 */
