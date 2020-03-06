#ifndef __CONVERTER_H__
#define __CONVERTER_H__

#include "common.h"

#include <vector>
#include <tbb/concurrent_vector.h>

class GridCSRConverter {
    uint32_t gridCount;
    vertex_t gridWidth;
    vertex_t maxVID;

    std::vector<std::vector<std::vector<edge_t>>> bin;
    std::vector<std::vector<tbb::concurrent_vector<uint32_t>>> position;
    struct {
        std::vector<std::vector<std::vector<vertex_t>>> row, ptr, col;
    } out;

    vertex_t _temp_src, _temp_dst;

public:
    GridCSRConverter(vertex_t const max_vid, vertex_t const grid_width);
    void insert(edge_t const & e);
    void run();
    void genGCSR();
    void write(fs::path const & folderPath, std::string const & dataName);
};

#endif