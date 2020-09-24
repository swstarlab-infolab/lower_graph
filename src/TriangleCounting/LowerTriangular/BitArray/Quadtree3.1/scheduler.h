#ifndef B03917E3_E4B8_49DF_B110_A0D13A6202EC
#define B03917E3_E4B8_49DF_B110_A0D13A6202EC

#include "base/shard.h"
#include "base/type.h"
#include "gridinfo.h"

#include <array>
#include <map>
#include <tbb/concurrent_queue.h>

using Job = std::array<uint32_t, 3>;

class Scheduler
{
private:
	tbb::concurrent_queue<Job> jobsCPU, jobsGPU;

	size_t criteria;

public:
	void init(GridInfo const & gridInfo);
	bool fetchJob(int const device_id, Job & job);
	void recordJobResult(Job const &  grid3,
						 size_t const triangles,
						 double const load_time,
						 double const kernel_time);
};
#endif /* B03917E3_E4B8_49DF_B110_A0D13A6202EC */
