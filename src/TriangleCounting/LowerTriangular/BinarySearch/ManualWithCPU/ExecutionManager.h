#ifndef BB067612_C0EA_47B3_9ACE_43EDFF450D05
#define BB067612_C0EA_47B3_9ACE_43EDFF450D05

#include "DataManager.h"
#include "ExecutionManager.h"
#include "type.h"

#include <memory>

class ExecutionManager
{
private:
	int ID;

	sp<DataManager> DM;

	struct {
		std::array<MemInfo<uint32_t>, 3> lookup;
		MemInfo<void>					 scan;
		MemInfo<Count>					 count;
	} mem;

	void initCPU();
	void initGPU();

public:
	void init(int const ID, sp<DataManager> dm);
	void run();
};

#endif /* BB067612_C0EA_47B3_9ACE_43EDFF450D05 */
