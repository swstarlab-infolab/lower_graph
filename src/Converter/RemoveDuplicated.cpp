#include "Converter.h"

#include <limits>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>

void GridCSRConverter::removeDuplicated() {
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
                    auto & p = this->position[row][col];
                    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, b.size()),
                        [&b, &p](tbb::blocked_range<uint32_t> const & s){
                            for (uint32_t start = s.begin(); start != s.end(); start+=s.grainsize()) {
                                for (uint32_t i = 0; i < s.grainsize(); i++) {
                                    auto const left = start + i;
                                    auto const right = start + i + 1;
                                    if (right < b.size()) {
                                        if ((b[left].src == b[right].src && b[left].dst == b[right].dst)) {
                                            p.push_back(right);
                                        }
                                    }
                                }
                            }
                        }, tbb::auto_partitioner());
                }
            }
        }, tbb::auto_partitioner());

    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    //std::sort(p.begin(), p.end());
                    tbb::parallel_sort(this->position[row][col].begin(), this->position[row][col].end());
                }
            }
        }, tbb::auto_partitioner());

    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    auto & p = this->position[row][col];
                    tbb::parallel_for_each(p.begin(), p.end(), [&b](uint32_t const & pos){
                        b[pos] = GridCSR::Edge{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
                    });
                }
            }
        }, tbb::auto_partitioner());

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
                    this->bin[row][col].resize(this->bin[row][col].size() - this->position[row][col].size());
                }
            }
        }, tbb::auto_partitioner());

    /*
    for (uint32_t row = 0; row < this->gridCount; row++) {
        for (uint32_t col = 0; col < this->gridCount; col++) {
            auto & b = this->bin[row][col];
            auto & p = this->position[row][col];

            if (p.size()) {
                printf("b[%d][%d] ", row, col);

                for (auto & e : b) {
                    printf("(%d,%d) ", e.src + row * this->gridWidth, e.dst + col * this->gridWidth);
                }
                printf("\n");
            }
        }
    }
    */
}
