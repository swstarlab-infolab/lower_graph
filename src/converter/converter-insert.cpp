#include "converter.h"

#include <cmath>

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

    printf("(%d,%d)\n", this->_temp_src, this->_temp_dst);

    // Always _temp_src > _temp_dst

    if (this->_temp_src > this->maxVID) {
        this->maxVID = this->_temp_src;

        // update this gridCount
        size_t new_gridCount = (this->maxVID / float(this->gridWidth)) + 1;
        if (new_gridCount > this->gridCount) {
            this->gridCount = new_gridCount;

            //alloc
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
    }

    this->bin[this->_temp_src / this->gridWidth][this->_temp_dst / this->gridWidth]
        .push_back(edge_t{this->_temp_src % this->gridWidth, this->_temp_dst % this->gridWidth});
}