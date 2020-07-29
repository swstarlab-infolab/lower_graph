#ifndef A9E6DD5D_CCA5_43BE_8F4C_1B6066553763
#define A9E6DD5D_CCA5_43BE_8F4C_1B6066553763

#include "context.h"
#include "data.h"

#include <array>
#include <boost/fiber/all.hpp>
#include <cuda_runtime.h>

#define CUDACHECK()                        \
	do {                                   \
		auto e = cudaGetLastError();       \
		if (e) {                           \
			printf("%s:%d, %s(%d), %s\n",  \
				   __FILE__,               \
				   __LINE__,               \
				   cudaGetErrorName(e),    \
				   e,                      \
				   cudaGetErrorString(e)); \
			cudaDeviceReset();             \
			exit(EXIT_FAILURE);            \
		}                                  \
	} while (false)

namespace Exec
{

template <typename T>
struct MemInfo {
	T *						   ptr;
	size_t					   byte;
	__host__ __device__ size_t count() { return byte / sizeof(T); }
	__host__ __device__ T & operator[](size_t const i) { return this->ptr[i]; }
	__host__ __device__ T const & operator[](size_t const i) const { return this->ptr[i]; }
};
using Grid	= MemInfo<uint32_t>[3];
using Grid3 = Grid[3];

template <>
struct MemInfo<void> {
	void * ptr;
	size_t byte;
};

class Manager
{
private:
	cudaStream_t myStream;
	int			 deviceID;

	struct {
		std::array<MemInfo<uint32_t>, 3> lookUp;
		std::array<MemInfo<uint32_t>, 2> bitArray;
		MemInfo<void>					 cubTemp;
		MemInfo<Count>					 count;
	} mem;
	Count runKernel(sp<Context> ctx, Grid3 G);

public:
	void				 init(sp<Context> ctx, int const deviceID);
	sp<bchan<JobResult>> run(sp<Context> ctx, sp<bchan<Job>> in);
};

} // namespace Exec

#endif /* A9E6DD5D_CCA5_43BE_8F4C_1B6066553763 */
