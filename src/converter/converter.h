#ifndef __CONVERTER_H__
#define __CONVERTER_H__

#include "../common.h"

#include <vector>
#include <tbb/concurrent_vector.h>


class GridCSRConverter {
public:

    /*
    GridCSRConverterMode explain:

    is_directed,    sort_type       result

    true            not_sort        directed graph, as-is; preserving original graph
    true            lower_triangle  directed graph, (src,dst), src > dst
    true            degree          directed graph, (src,dst), degree(src) < degree(dst) 

    false           not_sort        undirected graph; store both (src,dst) and (dst,src)
    false           lower_triangle  undirected graph; store both (src,dst) and (dst,src)
    false           degree          undirected graph; store both (src,dst) and (dst,src)
    */

    struct GridCSRConverterMode {
        bool is_directed;
        enum class SortingType {
            not_sort,
            lower_triangle,
            degree
        };
        SortingType sort_type;
    };

private:
    uint32_t gridCount;
    vertex_t gridWidth;
    vertex_t maxVID;

    std::vector<std::vector<std::vector<edge_t>>> bin;
    std::vector<std::vector<tbb::concurrent_vector<uint32_t>>> position;
    struct {
        std::vector<std::vector<std::vector<vertex_t>>> row, ptr, col;
    } out;

    vertex_t _temp_src, _temp_dst;

    std::vector<vertex_t> degree;

    GridCSRConverterMode mode;

    // These insert functions are not thread-safe!
    void insertLowerTriangle(edge_t const & e);
    void insertUndirectedDegree(edge_t const & e);
    void insertUndirected(edge_t const & e);
    void insert(edge_t const & e);

    void removeDuplicated();
    void removeDegreeBased();
    void genGCSR();

public:
    GridCSRConverter(vertex_t const grid_width) : gridCount(0), gridWidth(grid_width), maxVID(0) {}
    void setMode(bool const is_directed, GridCSRConverterMode::SortingType sort_type) {
        this->mode.is_directed = is_directed;
        if (is_directed) {
            this->mode.sort_type = sort_type;
        } else {
            using type = GridCSRConverterMode::SortingType;
            this->mode.sort_type = type::not_sort;
        }
    }
    void loadAdj6(fs::path const & folderPath);
    void loadTSV(fs::path const & folderPath);
    void run() {
        removeDuplicated();
        using type = GridCSRConverterMode::SortingType;
        if (this->mode.is_directed && this->mode.sort_type == type::degree) {
            removeDegreeBased();
        }
        genGCSR();
    }
    void storeGCSR(fs::path const & folderPath, std::string const & dataName);
};

#endif