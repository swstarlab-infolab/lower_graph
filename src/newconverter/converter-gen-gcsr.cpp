#include "converter.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

void GridCSRConverter::genGCSR() {
    // Write out_row
    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    auto & o = this->out.row[row][col];

                    if (b.size()) {
                        size_t unique_count = 0;
                        int64_t max = -1;

                        for (auto & e : b) {
                            if ((int64_t)e.src > max) {
                                unique_count++;
                                max = e.src;
                            }
                        }

                        o.resize(unique_count);

                        unique_count = 0;
                        max = -1;

                        for (auto & e : b) {
                            if ((int64_t)e.src > max) {
                                o[unique_count] = e.src;
                                unique_count++;
                                max = e.src;
                            }
                        }
                    }
                }
            }
        }, tbb::auto_partitioner());
    
    // Write data
    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    auto & o = this->out.ptr[row][col];

                    if (b.size()) {
                        o.resize(this->out.row[row][col].size() + 1);

                        size_t unique_count = 0;
                        int64_t max = -1;

                        for (auto & e : b) {
                            if ((int64_t)e.src > max) {
                                unique_count++;
                                max = e.src;
                                o[unique_count] = o[unique_count - 1];
                            }
                            o[unique_count]++;
                        }
                    }
                }
            }
        }, tbb::auto_partitioner());

    // Write data
    tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
        [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
            for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                    auto & b = this->bin[row][col];
                    auto & o = this->out.col[row][col];

                    if (b.size()) {
                        o.resize(b.size());

                        for (size_t i = 0; i < b.size(); ++i) {
                            o[i] = b[i].dst;
                        }
                    }
                }
            }
        }, tbb::auto_partitioner());

    /* 
    for (uint32_t row = 0; row < this->gridCount; row++) {
        for (uint32_t col = 0; col < this->gridCount; col++) {
            {
                auto & o = this->out.row[row][col];
                if (o.size()) {
                    printf("ROW: ");
                    for (auto & e : o) { printf("%d ", e); }
                    printf("\n");
                }
            }
            {
                auto & o = this->out.ptr[row][col];
                if (o.size()) {
                    printf("PTR: ");
                    for (auto & e : o) { printf("%d ", e); }
                    printf("\n");
                }
            }
            {
                auto & o = this->out.col[row][col];
                if (o.size()) {
                    printf("COL: ");
                    for (auto & e : o) { printf("%d ", e); }
                    printf("\n");
                }
            }
        }
    }
    */
}