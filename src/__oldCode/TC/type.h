#ifndef F22DF52F_BF2C_422C_BCD2_18B68DE1E4F3
#define F22DF52F_BF2C_422C_BCD2_18B68DE1E4F3

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <array>
#include <boost/fiber/all.hpp>
#include <memory>
#include <string>

#define fiber  boost::fibers::fiber
#define bchan  boost::fibers::buffered_channel
#define uchan  boost::fibers::unbuffered_channel
#define sp	   std::shared_ptr
#define makeSp std::make_shared

constexpr std::array<size_t, 2> EXP_BITARR = {12L, 5L};
constexpr size_t				GRID_WIDTH = (1L << 24);

using Count = unsigned long long;
using Job	= std::array<std::string, 3>;
struct JobResult {
	Job	  job;
	Count triangle;
};

#endif /* F22DF52F_BF2C_422C_BCD2_18B68DE1E4F3 */
