#ifndef __CONVERTER_H__
#define __CONVERTER_H__

#include "../common.h"

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

    void insert(edge_t const & e);
    void divideEdges();
    void genGCSR();

public:
    GridCSRConverter(vertex_t const grid_width) : gridCount(0), gridWidth(grid_width), maxVID(0) {}
    void loadAdj6(fs::path const & folderPath);
    void loadTSV(fs::path const & folderPath);
    void run() { divideEdges(); genGCSR(); }
    void storeGCSR(fs::path const & folderPath, std::string const & dataName);
};

#endif