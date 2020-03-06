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

#define FORMAT_GRID_POWER 24

#ifndef FORMAT_GRID_POWER
#define FORMAT_GRID_POWER 24
#endif

#define FORMAT_GRID_WIDTH uint32_t(1<<FORMAT_GRID_POWER)

// Grid Information
struct gridInfo_t {
    struct { uint32_t row, col; } pos;
    struct { fs::path row, ptr, col; } path;
};

// Grid Data
struct gridData_t {
    std::vector<vertex_t> row, ptr, col;
};

void readADJ6(fs::path const & folder, std::vector<edge_t> & edgelist);
void readTSV(fs::path const & folder, std::vector<edge_t> & edgelist);
void writeGCSR(std::vector<edge_t> const & edgelist, fs::path const & folder, std::string const & dbname);
void readGCSR(fs::path const & meta_path, std::vector<gridInfo_t> & info, std::vector<gridData_t> & data);


#endif