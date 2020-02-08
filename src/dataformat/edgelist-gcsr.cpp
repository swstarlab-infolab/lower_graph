#include <dataformat.h>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

using json = nlohmann::json;

void writeGCSR(std::vector<edge_t> const & edgelist, fs::path const & folder, std::string const & dbname) {
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

    auto grids = vertex_t(std::ceil(float(maxvertex) / float(FORMAT_GRID_WIDTH)));

    std::cout << "complete: find max vertex id" << std::endl;

    auto rc2i = [&grids](vertex_t const row, vertex_t const col) ->vertex_t{ return row * grids + col; };
    auto i2r = [&grids](vertex_t const idx) ->vertex_t { return idx / grids; };
    auto i2c = [&grids](vertex_t const idx) ->vertex_t{ return idx % grids; };

    std::vector<std::vector<std::vector<vertex_t>>> adjlist;
    adjlist.resize(grids * grids);

    for (uint32_t r = 0; r < grids; r++) {
        for (uint32_t c = 0; c < grids; c++) {
            adjlist[rc2i(r, c)].resize(FORMAT_GRID_WIDTH);
        }
    }

    std::cout << "complete: allocate memory space for grid-divided adjacency list" << std::endl;

    for (auto & e : edgelist) {
        auto const gRow = e.src / FORMAT_GRID_WIDTH;
        auto const gCol = e.dst / FORMAT_GRID_WIDTH;

        auto const gRowOff = e.src % FORMAT_GRID_WIDTH;
        auto const gColOff = e.dst % FORMAT_GRID_WIDTH;

        adjlist[rc2i(gRow, gCol)][gRowOff].push_back(gColOff);
    }

    std::cout << "complete: separate edges from edgelist to grid-divided adjacency list" << std::endl;

    // set memory
    std::vector<gridData_t> out(grids * grids);
    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];
        o.ptr.resize(FORMAT_GRID_WIDTH + 1);
        std::fill(o.ptr.begin(), o.ptr.end(), vertex_t(0));

        size_t columns = 0;
        for (auto & row : a) {
            columns += row.size();
        }

        o.col.resize(columns);
    }


    std::cout << "complete: allocate output memory (ptr, col) and fill zeroes" << std::endl;

    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];

        for (size_t ridx = 0; ridx < a.size(); ridx++) {
            if (a[ridx].size()) {
                o.row.push_back(ridx);
            }
        }
    }
    
    std::cout << "complete: rows" << std::endl;

    // fill ptr
    for (size_t i = 0; i < adjlist.size(); i++) {
        auto & o = out[i];
        auto & a = adjlist[i];

        // prefix sum (exclusive scan)
        for (size_t j = 1; j < a.size()+1; j++) {
            o.ptr[j] = o.ptr[j-1] + a[j-1].size();
        }
    }

    std::cout << "complete: fill ptr" << std::endl;

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
    j["ext"]["row"] = "row";
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

        of.open(rootfolder.string() + std::string(_name) + ".row", std::ios::out | std::ios::binary);
        of.write((char*)o.row.data(), sizeof(vertex_t) * o.col.size());
        of.close();

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