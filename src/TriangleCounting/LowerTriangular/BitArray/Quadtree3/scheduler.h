#ifndef B03917E3_E4B8_49DF_B110_A0D13A6202EC
#define B03917E3_E4B8_49DF_B110_A0D13A6202EC

#include "mysql_connection.h"
#include "shard.h"
#include "type.h"

#include <array>

class Scheduler
{
public:
	using Grid3	  = std::array<size_t, 3>;
	Grid3 jobHalt = {-1, -1, -1};

	size_t gpus;
	size_t criteria;
	void   init(fs::path const & folderPath);
	Grid3  fetchJob(size_t const device_id);
	void   finishJob(Grid3 const  grid3,
					 size_t const triangles,
					 double const load_time,
					 double const kernel_time);
};
#endif /* B03917E3_E4B8_49DF_B110_A0D13A6202EC */
