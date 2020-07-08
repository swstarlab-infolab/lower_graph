#ifndef __CONVERTER_H__
#define __CONVERTER_H__

#include <GridCSR/GridCSR.h>

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
    GridCSR::Vertex gridWidth;
    GridCSR::Vertex maxVID;

    std::vector<std::vector<std::vector<GridCSR::Edge>>> bin;
    std::vector<std::vector<tbb::concurrent_vector<uint32_t>>> position;
    struct {
        std::vector<std::vector<std::vector<GridCSR::Vertex>>> row, ptr, col;
    } out;

    GridCSR::Vertex _temp_src, _temp_dst;

    std::vector<GridCSR::Vertex> degree;

    GridCSRConverterMode mode;

    // These insert functions are not thread-safe!
    void insertLowerTriangle(GridCSR::Edge const & e);
    void insertUndirectedDegree(GridCSR::Edge const & e);
    void insertUndirected(GridCSR::Edge const & e);
    void insert(GridCSR::Edge const & e);

    void removeDuplicated();
    void removeDegreeBased();
    void genGCSR();

public:
    GridCSRConverter(GridCSR::Vertex const grid_width) : gridCount(0), gridWidth(grid_width), maxVID(0) {}
    void setMode(bool const is_directed, GridCSRConverterMode::SortingType sort_type) {
        this->mode.is_directed = is_directed;
        if (is_directed) {
            this->mode.sort_type = sort_type;
        } else {
            using type = GridCSRConverterMode::SortingType;
            this->mode.sort_type = type::not_sort;
        }
    }
    void loadAdj6(GridCSR::FS::path const & folderPath);
    void loadTSV(GridCSR::FS::path const & folderPath);
    void run() {
        removeDuplicated();
        using type = GridCSRConverterMode::SortingType;
        if (this->mode.is_directed && this->mode.sort_type == type::degree) {
            removeDegreeBased();
        }
        genGCSR();
    }
    void storeGCSR(GridCSR::FS::path const & folderPath, std::string const & dataName);
};

#endif