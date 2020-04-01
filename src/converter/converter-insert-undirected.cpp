#include "converter.h"

#include <cmath>

void GridCSRConverter::insertUndirected(edge_t const & e) {
    if (e.src == e.dst) {
        // Skip self-loop
        return;
    }

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

    // To make undirected graph, push both
    this->bin[e.src / this->gridWidth][e.dst / this->gridWidth]
        .push_back(edge_t{e.src % this->gridWidth, e.dst % this->gridWidth});

    this->bin[e.dst / this->gridWidth][e.src / this->gridWidth]
        .push_back(edge_t{e.dst % this->gridWidth, e.src % this->gridWidth});
}