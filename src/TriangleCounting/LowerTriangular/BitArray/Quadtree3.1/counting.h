#ifndef DD290292_80F2_4286_9FBC_3BD2FE246214
#define DD290292_80F2_4286_9FBC_3BD2FE246214

/*
#include "base/type.h"

#include <cuda_runtime.h>

class Counting
{
	int			 deviceID;
	cudaStream_t stream;

	struct {
		std::array<DataInfo, 3> lookup;
		std::array<DataInfo, 2> bitarr;
		DataInfo				cub;
		DataInfo				count;
	} mem;

public:
	void init(int const deviceID, int const streamID);
	~Counting();
	size_t count(DataInfo const & info);
};
*/

#include "base/type.h"

using Grid	= std::array<DataInfo<uint32_t>, 3>;
using Grids = std::array<Grid, 3>;

using Lookup  = std::vector<uint32_t>;
using Lookups = std::array<Lookup, 3>;

Count countingCPU(Grids const & Gs, Lookups & Ls);

#endif /* DD290292_80F2_4286_9FBC_3BD2FE246214 */
