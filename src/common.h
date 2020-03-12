#ifndef __COMMON_H__
#define __COMMON_H__

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <cstdint>

using vertex_t = uint32_t;
struct edge_t { vertex_t src, dst; };

#endif