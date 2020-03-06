#include "converter.h"

#include <cmath>

GridCSRConverter::GridCSRConverter(vertex_t const max_vid, vertex_t const grid_width) {
    this->maxVID = max_vid;
    this->gridWidth = grid_width;
    this->gridCount = ceil(this->maxVID / float(this->gridWidth));

    this->bin.resize(this->gridCount);
    for (auto & row : this->bin) {
        row.resize(this->gridCount);
    }

    this->position.resize(this->gridCount);
    for (auto & row : this->position) {
        row.resize(this->gridCount);
    }

    this->out.row.resize(this->gridCount);
    for (auto & row : this->out.row) {
        row.resize(this->gridCount);
    }

    this->out.ptr.resize(this->gridCount);
    for (auto & row : this->out.ptr) {
        row.resize(this->gridCount);
    }

    this->out.col.resize(this->gridCount);
    for (auto & row : this->out.col) {
        row.resize(this->gridCount);
    }
}

void GridCSRConverter::insert(edge_t const & e) {
    if (e.src > e.dst) {
        // Make lower triangular matrix
        this->_temp_src = e.src;
        this->_temp_dst = e.dst;
    } else if (e.src < e.dst) {
        // Make lower triangular matrix (flip)
        this->_temp_src = e.dst;
        this->_temp_dst = e.src;
    } else {
        // Skip self-loop
        return;
    }

    this->bin[this->_temp_src / this->gridWidth][this->_temp_dst / this->gridWidth]
        .push_back(edge_t{this->_temp_src % this->gridWidth, this->_temp_dst % this->gridWidth});
}