#ifndef FD70FEEA_6D13_4034_BECE_35EC616FA6BC
#define FD70FEEA_6D13_4034_BECE_35EC616FA6BC

#include "my_mysql.h"
#include "type.h"

#include <array>
#include <boost/pool/pool.hpp>
#include <jemalloc/jemalloc.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <vector>

struct MemReqInfo {
	int		device_id;
	size_t	grid_id;
	uint8_t file_type;
};

class Cache
{
public:
private:
	using DevicePoolType = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
	std::vector<std::shared_ptr<DevicePoolType>> device_pool;
	std::vector<cudaStream_t>					 cudaStream;

	DataInfo genDataInfo(mysqlpp::Connection & conn, MemReqInfo const & reqInfo, int const state);
	bool	 changeState(mysqlpp::Connection & conn,
						 MemReqInfo const &	   reqInfo,
						 int const			   state_from,
						 int const			   state_to);
	bool	 refCountUpForExist(mysqlpp::Connection & conn, MemReqInfo const & reqInfo);

public:
	fs::path folderPath;
	int		 gpus;

	void init();
	~Cache();

	// file_type: 0=row, 1=ptr, 2=col
	DataInfo load(MemReqInfo const & reqInfo);

	// file_type: 0=row, 1=ptr, 2=col
	void done(MemReqInfo const & reqInfo);
};
#endif /* FD70FEEA_6D13_4034_BECE_35EC616FA6BC */
