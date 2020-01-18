#include <tbb/tbb.h>
#include <nlohmann/json.hpp>
#include <grid.h>

#include <vector>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

#define EDGEMAX edge_t{0xFFFFFFFF, 0xFFFFFFFF}
#define UNIT 6
#define GRID_WIDTH 64

static void readADJ6(fs::path const & folder, std::vector<edge_t> & edgelist) {
    std::vector<char> temp;

    std::fstream fs;
    auto bigendian = [&temp](uint64_t const i) -> uint64_t {
        return
            (uint64_t(temp[i+0]) << (8*5)) +
            (uint64_t(temp[i+1]) << (8*4)) +
            (uint64_t(temp[i+2]) << (8*3)) +
            (uint64_t(temp[i+3]) << (8*2)) +
            (uint64_t(temp[i+4]) << (8*1)) +
            (uint64_t(temp[i+5]) << (8*0));};

    uint64_t estimated = 0;

    for (auto & p : fs::recursive_directory_iterator(folder)) {
        fs.open(p.path());
        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        estimated += filesize / UNIT;
        fs.close();
    }

    edgelist.resize(estimated);

    uint64_t position = 0;
    for (auto & p : fs::recursive_directory_iterator(folder)) {
        fs.open(p.path());
        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        temp.resize(filesize);
        std::cout << p.path().string() << std::endl;
        fs.read(temp.data(), filesize);
        fs.close();

        for (uint64_t i = 0; i < filesize;) {
            uint64_t src = bigendian(i);
            i+=UNIT;
            uint64_t size = bigendian(i);
            i+=UNIT;
            for (uint64_t j = 0; j < size; j++) {
                uint64_t dst = bigendian(i);
                edgelist[position] = edge_t{vertex_t(src), vertex_t(dst)};
                position++;
                i+=UNIT;
            }
        }
    }

    edgelist.resize(position);
}

static void writeGCSR(std::vector<edge_t> const & edgelist, fs::path const & folder, std::string const & dbname) {
    // make CSR

    auto maxvertex = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, edgelist.size()),
        0,
        [&edgelist](tbb::blocked_range<size_t> r, vertex_t max) -> vertex_t {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (edgelist[i].src > max) { max = edgelist[i].src; }
                if (edgelist[i].dst > max) { max = edgelist[i].dst; }
            }
            return max;
        },
        [](vertex_t const & a, vertex_t const & b) -> vertex_t {
            return (a < b) ? b : a;
        });

    auto grids = std::ceil(maxvertex / float(GRID_WIDTH));
    auto rc2i = [&grids](size_t const row, size_t const col) { return row * grids + col; };

    std::vector<std::vector<std::vector<vertex_t>>> grid;

    for (uint32_t r = 0; r < grids; r++) {
        for (uint32_t c = 0; c < grids; c++) {
            grid[rc2i(r, c)].resize(GRID_WIDTH);
        }
    }

    for (auto & e : edgelist) {
        auto const gRow = e.src / GRID_WIDTH;
        auto const gRowOff = e.src % GRID_WIDTH;
        auto const gCol = e.dst / GRID_WIDTH;
        auto const gColOff = e.dst % GRID_WIDTH;
        grid[rc2i(gRow, gCol)][gRowOff].push_back(gColOff);
    }

    for (auto & e : edgelist) {

    }
}

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cerr << "usage" << std::endl << argv[0] << " <input_folder_path> <output_folder_path> <db_name>" << std::endl;
        return 0;
    }

    auto const inFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const outFolder = fs::path(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
    auto const outDBName = std::string(argv[3]);

    tbb::task_scheduler_init();

    std::vector<edge_t> edgelist;
    readADJ6(inFolder, edgelist);

    tbb::parallel_for_each(edgelist.begin(), edgelist.end(),
        [](edge_t & e) {
            if (e.src < e.dst) {
                std::swap(e.src, e.dst);
            }
        });
    
    tbb::parallel_sort(edgelist.begin(), edgelist.end(),
        [](const edge_t & e1, const edge_t & e2){ return e1 < e2; });

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

    tbb::parallel_sort(edgelist.begin(), edgelist.end(),
        [](const edge_t & e1, const edge_t & e2){ return e1 < e2; });
    
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

    writeGCSR(edgelist, outFolder, outDBName);

    return 0;
}