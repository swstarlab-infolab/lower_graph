#ifndef FD70FEEA_6D13_4034_BECE_35EC616FA6BC
#define FD70FEEA_6D13_4034_BECE_35EC616FA6BC

#include "type.h"

#include <array>
#include <boost/pool/pool.hpp>
#include <jemalloc/jemalloc.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <vector>

struct DataInfo {
	void *	 addr;
	size_t	 byte;
	fs::path path;
};

// using GridDataInfo = std::array<DataInfo, 3>;

class Cache
{
private:
	using DevicePoolType = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
	std::vector<std::shared_ptr<DevicePoolType>> device_pool;

public:
	fs::path folderPath;
	int		 gpus;

	void init();

	// file_type: 0=row, 1=ptr, 2=col
	GridDataInfo load(size_t const device_id, size_t const grid_id, uint8_t const file_type);

	// file_type: 0=row, 1=ptr, 2=col
	void done(size_t const device_id, size_t const grid_id, uint8_t const file_type);
};
#endif /* FD70FEEA_6D13_4034_BECE_35EC616FA6BC */
