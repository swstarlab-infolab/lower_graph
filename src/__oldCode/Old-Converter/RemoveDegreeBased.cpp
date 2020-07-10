#include "Converter.h"

#include <limits>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>

void GridCSRConverter::removeDegreeBased() {
    // rank
    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    tbb::parallel_for_each(b.begin(), b.end(), [this, &row, &col, &b](GridCSR::Edge & pos){
                        auto g_row = row * this->gridWidth + pos.src;
                        auto g_col = col * this->gridWidth + pos.dst;
                        if (this->degree[g_row] > this->degree[g_col]) {
                            pos = GridCSR::Edge{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
                        } else if (this->degree[g_row] == this->degree[g_col]) {
                            if (pos.src > pos.dst) {
                                pos = GridCSR::Edge{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
                            }
                        }
                    });
                }
            }
        }, tbb::auto_partitioner());

    // sort
    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    tbb::parallel_sort(this->bin[row][col].begin(), this->bin[row][col].end(),
                        [](GridCSR::Edge const left, GridCSR::Edge const right) {
                            return (left.src != right.src) ? (left.src < right.src) : (left.dst < right.dst);
                        }
                    );
                }
            }
        }, tbb::auto_partitioner());

    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    GridCSR::Vertex count = 0;
                    for (int i = b.size(); i > 0; --i) {
                        if (b[i-1].src == std::numeric_limits<uint32_t>::max() && b[i-1].dst == std::numeric_limits<uint32_t>::max()) {
                            count++;
                        } else {
                            break; 
                        }
                    }
                    b.resize(b.size() - count);
                }
            }
        }, tbb::auto_partitioner());
}
