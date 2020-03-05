#include <cstdint>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>
#include <tbb/concurrent_vector.h>
#include <dataformat.h>

using vertex_t = uint32_t;
struct edge_t { vertex_t src, dst; };

class GridCSRConverter {
    uint32_t gridCount;
    vertex_t gridWidth;

    std::vector<std::vector<std::vector<edge_t>>> bin;
    std::vector<std::vector<tbb::concurrent_vector<uint32_t>>> position;

    vertex_t _temp_src, _temp_dst;

public:
    GridCSRConverter(vertex_t const max_vid, vertex_t const grid_width) {
        this->gridWidth = grid_width;
        this->gridCount = ceil(max_vid / float(this->gridWidth));

        this->bin.resize(this->gridCount);
        for (auto & row : this->bin) {
            row.resize(this->gridCount);
        }

        this->position.resize(this->gridCount);
        for (auto & row : this->position) {
            row.resize(this->gridCount);
        }
    }

    void insert(edge_t const e) {
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

    void run() {
        tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
            [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
                for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                    for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                        tbb::parallel_sort(this->bin[row][col].begin(), this->bin[row][col].end(),
                            [](edge_t const left, edge_t const right) {
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
                            b[pos] = edge_t{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
                        });
                    }
                }
            }, tbb::auto_partitioner());

        tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
            [this](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
                for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                    for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                        tbb::parallel_sort(this->bin[row][col].begin(), this->bin[row][col].end(),
                            [](edge_t const left, edge_t const right) {
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

    void write() {
        // Write JSON


        // Write data
        tbb::parallel_for(tbb::blocked_range2d<uint32_t, uint32_t>(0, this->gridCount, 0, this->gridCount),
            [this, &func](tbb::blocked_range2d<uint32_t, uint32_t> const & r){
                for (uint32_t row = r.rows().begin(); row < r.rows().end(); row++) {
                    for (uint32_t col = r.cols().begin(); col < r.cols().end(); col++) {
                        auto & b = this->bin[row][col];
                        auto & p = this->position[row][col];

                        if (b.size()) {

                        }
                    }
                }
            }, tbb::auto_partitioner());
    }
};

int main() {
    std::vector<edge_t> input = {
        { 1, 0 },
        { 1, 0 },
        { 1, 0 },
        { 1, 0 },
        { 1, 0 },
        { 1, 4 },
        { 1, 7 },
        { 2, 4 },
        { 2, 4 },
        { 2, 4 },
        { 3, 7 },
        { 2, 1 },
        { 3, 1 },
        { 3, 1 },
        { 3, 1 },
        { 3, 1 },
        { 3, 1 },
        { 3, 4 },
        { 6, 2 },
        { 6, 3 },
        { 6, 2 },
        { 6, 2 },
        { 7, 3 },
        { 7, 3 },
        { 7, 3 },
        { 3, 3 },
        { 7, 3 },
    };

    vertex_t max_vid =
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, input.size()),
            0,
            [&input](tbb::blocked_range<size_t> & r, vertex_t max) -> vertex_t {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    auto & in = input[i];
                    max = (in.src > max) ? in.src : max;
                    max = (in.dst > max) ? in.dst : max;
                }
                return max;
            },
            [](vertex_t const & a, vertex_t const & b) -> vertex_t {
                return (a > b) ? a : b;
            }, tbb::auto_partitioner());

    GridCSRConverter converter(max_vid, 1 << 2);

    for (auto & e : input) { converter.insert(e); }
    input.clear();

    converter.run();
    converter.write();
    return 0;
}