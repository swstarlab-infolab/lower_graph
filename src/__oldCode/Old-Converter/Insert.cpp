#include "Converter.h"

#include <cmath>

void GridCSRConverter::insert(GridCSR::Edge const & e) {
    auto const larger = (e.src > e.dst) ? e.src : e.dst;

    if (larger > this->maxVID) {
        this->maxVID = larger;

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

    this->bin[e.src / this->gridWidth][e.dst / this->gridWidth]
        .push_back(GridCSR::Edge{e.src % this->gridWidth, e.dst % this->gridWidth});
}