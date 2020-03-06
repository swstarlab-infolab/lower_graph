#include "common.h"
#include "converter.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

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
    converter.genGCSR();
    converter.write(".", "test");
    return 0;
}