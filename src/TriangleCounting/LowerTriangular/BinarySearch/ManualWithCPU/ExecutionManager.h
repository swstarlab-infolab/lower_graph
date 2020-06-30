#ifndef BB067612_C0EA_47B3_9ACE_43EDFF450D05
#define BB067612_C0EA_47B3_9ACE_43EDFF450D05

#include "DataManager.h"
#include "type.h"

#include <array>
#include <memory>

class ExecutionManager
{
public:
	using Grid	= MemInfo<Vertex>[3];
	using Grids = Grid[3];

private:
	int ID;

	sp<DataManager> DM;

	struct {
		std::array<MemInfo<uint32_t>, 3> lookup;
		MemInfo<void>					 scan;
		MemInfo<Count>					 count;
	} mem;

	void  initCPU();
	void  initGPU();
	Count execGPU(Grids g);
	Count execCPU(Grids g);

public:
	void init(int const ID, sp<DataManager> dm);
	void run();
};

#endif /* BB067612_C0EA_47B3_9ACE_43EDFF450D05 */
