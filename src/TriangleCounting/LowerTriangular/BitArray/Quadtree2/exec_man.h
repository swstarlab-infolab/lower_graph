#ifndef C692EFAE_8862_4877_8701_A6E131756639
#define C692EFAE_8862_4877_8701_A6E131756639

#include "data_man.h"
#include "sched_man.h"
#include "type.h"

#include <array>
#include <cuda_runtime.h>

namespace Exec
{
struct Result {
	int			   deviceID, streamID;
	Sched::JobType job;
	Count		   triangle;
	struct {
		double kernel, load;
	} time;
};

struct GPUSetting {
	int stream, block, thread;
};

template <typename T>
struct MemInfo {
	T *						   ptr;
	size_t					   byte;
	__host__ __device__ size_t count() const { return byte / sizeof(T); }
	__host__ __device__ T & operator[](size_t const i) { return this->ptr[i]; }
	__host__ __device__ T const & operator[](size_t const i) const { return this->ptr[i]; }
	__host__ MemInfo<T> & operator=(Data::MemInfo const & in)
	{
		this->ptr  = (T *)in.ptr;
		this->byte = in.byte;
		return *this;
	}
};

template <>
struct MemInfo<void> {
	void * ptr;
	size_t byte;
};

using OutType = std::shared_ptr<boost::fibers::buffered_channel<Result>>;
using Grid	  = MemInfo<uint32_t>[3];
using Grids	  = Grid[3];

class Manager
{
public:
	OutType run();
	Count	launchKernelGPU(Grids & G);

	Manager(int const						deviceID,
			int const						streamID,
			GPUSetting const				gpuSetting,
			std::shared_ptr<Sched::Manager> sched,
			std::shared_ptr<Data::Manager>	data);
	~Manager();

private:
	struct {
		std::array<MemInfo<uint32_t>, 3> lookup;
		std::array<MemInfo<uint32_t>, 2> bitarr;
		MemInfo<void>					 cub;
		MemInfo<Count>					 count;
	} mem;

	cudaStream_t					myStream;
	int								deviceID;
	int								streamID;
	std::shared_ptr<Sched::Manager> sched;
	std::shared_ptr<Data::Manager>	data;
	GPUSetting						gpuSetting;
};
} // namespace Exec
#endif /* C692EFAE_8862_4877_8701_A6E131756639 */
