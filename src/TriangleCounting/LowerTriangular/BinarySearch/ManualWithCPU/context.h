#ifndef EA7351CA_4A66_487C_83BF_81FA47871E41
#define EA7351CA_4A66_487C_83BF_81FA47871E41

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "DataManager.h"
#include "ExecutionManager.h"
#include "ScheduleManager.h"
#include "type.h"

#include <string>

struct Context {
private:
	uint32_t findMaxGridIndex(std::string const & ext);

public:
	fs::path folder;
	size_t	 threshold;
	size_t	 chanSz;
	struct {
		uint32_t width, count;
	} grid;

	struct {
		int devices, streams, blocks, threads;
	} gpu;

	struct {
		sp<bchan<Order>> orderCPU, orderGPU;

		std::unordered_map<int, sp<bchan<Report>>> report;
	} chan;

	std::unordered_map<int, sp<ExecutionManager>> EM;
	std::unordered_map<int, sp<DataManager>>	  DM;
	sp<ScheduleManager>							  SM;

	void init(int argc, char * argv[]);
};

extern Context ctx; // main.cpp

#endif /* EA7351CA_4A66_487C_83BF_81FA47871E41 */
