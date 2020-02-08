#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/concurrent_vector.h>

#include <dataformat.h>

#include <vector>
#include <fstream>
#include <iostream>

#include <iomanip>

#define EDGEMAX edge_t{0xFFFFFFFF, 0xFFFFFFFF}
#define UNIT 6

static bool operator==(edge_t const & a, edge_t const & b) {
    return a.dst == b.dst && a.src == b.src;
}

static bool operator!=(edge_t const & a, edge_t const & b) {
    return !(a.dst == b.dst && a.src == b.src);
}

static bool operator<(edge_t const & a, edge_t const & b) {
    return (a.src == b.src) ? a.dst < b.dst : a.src < b.src;
}

int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl
            << argv[0] << " <type> <input_folder_path> <output_folder_path> <db_name>" << std::endl
            << "type: 0 tsv, 1 adj6" << std::endl;
        return 0;
    }

    bool is_adj6 = false;
    if (std::string(argv[1]) == "0") { is_adj6 = false; }
    else if (std::string(argv[1]) == "1") { is_adj6 = true; }
    else { return 0; }

    auto const inFolder = fs::path(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
    auto const outFolder = fs::path(fs::path(std::string(argv[3]) + "/").parent_path().string() + "/");
    auto const outDBName = std::string(argv[4]);

    std::vector<edge_t> edgelist;
    if (is_adj6) {
        readADJ6(inFolder, edgelist);
    } else {
        readTSV(inFolder, edgelist);
    }

    std::cout << ">>> Create Lower Triangular Matrix" << std::endl;
    tbb::parallel_for_each(edgelist.begin(), edgelist.end(),
        [](edge_t & e) {
            if (e.src < e.dst) {
                std::swap(e.src, e.dst);
            }
        });

    std::cout << "complete: edge swap" << std::endl;
    
    tbb::parallel_sort(edgelist.begin(), edgelist.end(),
        [](const edge_t & e1, const edge_t & e2){ return e1 < e2; });

    std::cout << "complete: edge sort" << std::endl;

    tbb::concurrent_vector<bool> change(edgelist.size());

    change.back() = false;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, edgelist.size() - 1),
        [&edgelist, &change](tbb::blocked_range<size_t> const r) {
            for (auto i = r.begin(); i < r.end(); ++i) {
                auto curr = edgelist[i];
                auto next = edgelist[i+1];

                if (curr.src == curr.dst) {
                    change[i] = true;
                } else {
                    change[i] = (curr == next) ? true : false;
                }
            }
        }, tbb::auto_partitioner());

    if (edgelist.back().src == edgelist.back().dst) {
        change.back() = true;
    }

    std::cout << "complete: check self loop and duplicated edges" << std::endl;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, edgelist.size() - 1),
        [&edgelist, &change](tbb::blocked_range<size_t> const r) {
            for (auto i = r.begin(); i < r.end(); ++i) {
                if (change[i]) {
                    edgelist[i] = EDGEMAX;
                }
            }
        }, tbb::auto_partitioner());

    if (change.back()) {
        edgelist.back() = EDGEMAX;
    }

    std::cout << "complete: change self loop and duplicated edges to <INTMAX, INTMAX>" << std::endl;

    tbb::parallel_sort(edgelist.begin(), edgelist.end(),
        [](const edge_t & e1, const edge_t & e2){ return e1 < e2; });

    std::cout << "complete: sort edges" << std::endl;
    
    [](std::vector<edge_t>& el) {
        size_t maxcount = 0;
        for (size_t i = el.size() - 1; i >= 0; i--) {
            auto& e = el[i];
            if (e != EDGEMAX) {
                el.resize(el.size() - maxcount);
                break;
            }
            maxcount++;
        }
    }(edgelist);

    std::cout << "complete: remove <INTMAX, INTMAX>" << std::endl;

    writeGCSR(edgelist, outFolder, outDBName);

    return 0;
}