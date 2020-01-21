#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/concurrent_vector.h>

#include <nlohmann/json.hpp>
#include <grid.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <iomanip>

using json = nlohmann::json;

#define EDGEMAX edge_t{0xFFFFFFFF, 0xFFFFFFFF}
#define UNIT 6

vertex_t constexpr GRID_WIDTH = (1<<24);

static bool operator==(edge_t const & a, edge_t const & b) {
    return a.dst == b.dst && a.src == b.src;
}

static bool operator!=(edge_t const & a, edge_t const & b) {
    return !(a.dst == b.dst && a.src == b.src);
}

static bool operator<(edge_t const & a, edge_t const & b) {
    return (a.src == b.src) ? a.dst < b.dst : a.src < b.src;
}

static size_t linecountTSV(fs::path const & file) {
    std::ifstream fs;
    fs.open(file);

    auto state_backup = fs.rdstate();
    fs.clear();
    auto pos_backup = fs.tellg();

    fs.seekg(0, std::ios::beg);
    size_t lf_count = std::count(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>(), '\n');
    fs.unget();
    if (fs.get() != '\n' ) {
        ++lf_count;
    }
    fs.clear();
    fs.seekg(pos_backup);
    fs.setstate(state_backup);
    fs.close();

    return lf_count;
}

static void readTSV(fs::path const & file, std::vector<edge_t> & edgelist) {
    std::cout << ">>> Read TSV Files" << std::endl;

    std::string temp;

    edgelist.resize(linecountTSV(file));

    std::ifstream fs;
    fs.open(file);
    fs.seekg(0, std::ios::end);
    uint64_t const filesize = fs.tellg();
    fs.seekg(0, std::ios::beg);
    temp.resize(filesize);
    fs.read(temp.data(), filesize);
    fs.close();

    std::cout << "complete: allocate memory of edgelist and buffer for TSV files" << std::endl;

    size_t edges = 0;
    std::string current_line;
    std::istringstream iss(temp);
    while (std::getline(iss, current_line)) {
        if (current_line[0] == '#') {
            continue;
        }

        char *ptr, *str = const_cast<char*>(current_line.c_str());
        ptr = strtok(str, "\t");
        edgelist[edges].src = atoi(ptr);
        ptr = strtok(nullptr, "\t");
        edgelist[edges].dst = atoi(ptr);
        edges++;
    }

    edgelist.resize(edges);
}

static void writeGCSR(std::vector<edge_t> const & edgelist, fs::path const & folder, std::string const & dbname) {
    std::cout << ">>> Dividing Grid" << std::endl;

    vertex_t maxvertex = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, edgelist.size()),
        0,
        [&edgelist](tbb::blocked_range<size_t> r, vertex_t max) -> vertex_t {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                max = (edgelist[i].src > max) ? edgelist[i].src : max;
                max = (edgelist[i].dst > max) ? edgelist[i].dst : max;
            }
            return max;
        },
        [](vertex_t const & a, vertex_t const & b) -> vertex_t {
            return (a < b) ? b : a;
        });

    auto grids = vertex_t(std::ceil(float(maxvertex) / float(GRID_WIDTH)));

    std::cout << "complete: find max vertex id" << std::endl;

    auto rc2i = [&grids](vertex_t const row, vertex_t const col) ->vertex_t{ return row * grids + col; };
    auto i2r = [&grids](vertex_t const idx) ->vertex_t { return idx / grids; };
    auto i2c = [&grids](vertex_t const idx) ->vertex_t{ return idx % grids; };

    std::vector<std::vector<std::vector<vertex_t>>> adjlist;
    adjlist.resize(grids * grids);

    for (uint32_t r = 0; r < grids; r++) {
        for (uint32_t c = 0; c < grids; c++) {
            adjlist[rc2i(r, c)].resize(GRID_WIDTH);
        }
    }

    std::cout << "complete: allocate memory space for grid-divided adjacency list" << std::endl;

    for (auto & e : edgelist) {
        auto const gRow = e.src / GRID_WIDTH;
        auto const gCol = e.dst / GRID_WIDTH;

        auto const gRowOff = e.src % GRID_WIDTH;
        auto const gColOff = e.dst % GRID_WIDTH;

        adjlist[rc2i(gRow, gCol)][gRowOff].push_back(gColOff);
    }

    std::cout << "complete: separate edges from edgelist to grid-divided adjacency list" << std::endl;

    // set memory
    std::vector<gridData_t> out(grids * grids);
    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];
        o.ptr.resize(GRID_WIDTH + 1);
        std::fill(o.ptr.begin(), o.ptr.end(), vertex_t(0));

        size_t columns = 0;
        for (auto & row : a) {
            columns += row.size();
        }

        o.col.resize(columns);
    }

    std::cout << "complete: allocate output memory (ptr, col) and fill zeroes" << std::endl;

    // fill ptr
    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];

        // prefix sum (exclusive scan)
        for (size_t j = 1; j < a.size()+1; j++) {
            o.ptr[j] = o.ptr[j-1] + a[j-1].size();
        }
    }

    std::cout << "complete: fill ptr using exclusive scan prefix sum" << std::endl;

    // fill col
    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];
        size_t k = 0;
        for (auto & row : a) {
            for (auto & col : row) {
                o.col[k++] = col;
            }
        }
    }

    std::cout << "complete: fill col" << std::endl;

    std::cout << ">>> Saving Files" << std::endl;
    // write file
    auto rootfolder = folder / (dbname + "/");
    fs::create_directory(rootfolder);

    std::ofstream of;
    json j;

    j["dataname"] = dbname;
    j["ext"]["ptr"] = "ptr";
    j["ext"]["col"] = "col";

    for (size_t i = 0; i < out.size(); i++) {
        auto & o = out[i];

        char _name[8];
        auto row = i2r(i), col = i2c(i);
        snprintf(_name, 8, "%03d-%03d", row, col);

        j["grids"][i]["filename"] = std::string(_name);
        j["grids"][i]["row"] = row;
        j["grids"][i]["col"] = col;

        of.open(rootfolder.string() + std::string(_name) + ".col", std::ios::out | std::ios::binary);
        of.write((char*)o.col.data(), sizeof(vertex_t) * o.col.size());
        of.close();

        of.open(rootfolder.string() + std::string(_name) + ".ptr", std::ios::out | std::ios::binary);
        of.write((char*)o.ptr.data(), sizeof(vertex_t) * o.ptr.size());
        of.close();
    }

    std::cout << "complete: write col and ptr files" << std::endl;

    of.open(rootfolder.string() + std::string("meta.json"), std::ios::out);
    of << std::setw(4) << j << std::endl;
    of.close();

    std::cout << "complete: write meta.json" << std::endl;
}

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cerr << "usage" << std::endl << argv[0] << " <input_file_path> <output_folder_path> <db_name>" << std::endl;
        return 0;
    }

    auto const inFile = fs::path(std::string(argv[1]));
    auto const outFolder = fs::path(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
    auto const outDBName = std::string(argv[3]);

    std::vector<edge_t> edgelist;
    readTSV(inFile, edgelist);

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