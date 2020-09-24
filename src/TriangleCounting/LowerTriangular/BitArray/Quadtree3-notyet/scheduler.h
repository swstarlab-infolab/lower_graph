#ifndef B03917E3_E4B8_49DF_B110_A0D13A6202EC
#define B03917E3_E4B8_49DF_B110_A0D13A6202EC

#include "shard.h"
#include "type.h"

#include <array>

class Scheduler
{
	size_t criteria;

public:
	void init(fs::path const & folderPath);
	bool fetchJob(int const device_id, std::array<uint32_t> & job);
	void finishJob(std::array<uint32_t> const job,
				   size_t const				  triangles,
				   double const				  load_time,
				   double const				  kernel_time);
};
#endif /* B03917E3_E4B8_49DF_B110_A0D13A6202EC */
