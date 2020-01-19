#ifndef __GRID_H__
#define __GRID_H__

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <vector>

// Vertex ID
using vertex_t = uint32_t;

// Edge Type
struct edge_t { vertex_t src, dst; };

// Grid Information
struct gridInfo_t {
    struct { uint32_t row, col; } pos;
    struct { fs::path row, ptr, col; } path;
};

// Grid Data
struct gridData_t {
    std::vector<vertex_t> ptr, col;
};

#endif