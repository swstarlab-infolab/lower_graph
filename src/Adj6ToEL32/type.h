#ifndef CA0B2FF9_2C71_4DD4_927B_AB0FDD3FD13F
#define CA0B2FF9_2C71_4DD4_927B_AB0FDD3FD13F

#include <array>
#include <atomic>
#include <boost/fiber/all.hpp>
#include <memory>
#include <stdint.h>

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#define bchan  boost::fibers::buffered_channel
#define uchan  boost::fibers::unbuffered_channel
#define fiber  boost::fibers::fiber
#define sp	   std::shared_ptr
#define makeSp std::make_shared

using V32  = uint32_t;			 // Vertex
using E32  = std::array<V32, 2>; // Edge
using GE32 = std::array<E32, 2>;

struct RowPos {
	size_t src, cnt, dstStart;
};

#endif /* CA0B2FF9_2C71_4DD4_927B_AB0FDD3FD13F */
