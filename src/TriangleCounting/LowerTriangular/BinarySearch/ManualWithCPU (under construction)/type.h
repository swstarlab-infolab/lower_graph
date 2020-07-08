#ifndef D898F24A_D90E_476E_838B_F302356EB0A9
#define D898F24A_D90E_476E_838B_F302356EB0A9

#include <array>
#include <boost/fiber/all.hpp>
#include <memory>
#include <string>

// shorten long name
#define sp	   std::shared_ptr
#define makeSp std::make_shared
#define bchan  boost::fibers::buffered_channel
#define uchan  boost::fibers::unbuffered_channel
#define fiber  boost::fibers::fiber

using Vertex	  = uint32_t;
using GridIndex32 = std::array<uint32_t, 2>;
using Grid3		  = std::array<GridIndex32, 3>;
using Count		  = unsigned long long;

using Order = Grid3;

struct Report {
	Grid3 g3;
	Count triangle;
	int	  deviceID;
};

template <typename T>
struct MemInfo {
	T *	   ptr;
	size_t byte;

#ifdef __CUDACC__
	__host__ __device__ size_t count() { return byte / sizeof(T); }
	__host__ __device__ T & operator[](size_t const pos) { return ptr[pos]; }
	__host__ __device__ T const & operator[](size_t const pos) const { return ptr[pos]; }
#else
	size_t	  count() { return byte / sizeof(T); }
	T &		  operator[](size_t const pos) { return ptr[pos]; }
	T const & operator[](size_t const pos) const { return ptr[pos]; }
#endif
};

template <>
struct MemInfo<void> {
	void * ptr;
	size_t byte;
};

#endif /* D898F24A_D90E_476E_838B_F302356EB0A9 */
