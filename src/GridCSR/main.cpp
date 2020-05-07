#include <GridCSR/GridCSR.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include <nlohmann/json.hpp>
using JSON = nlohmann::json;

#define EXPAND(x) x

// Count the number of arguments
#define ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define SEQ_N 8, 7, 6, 5, 4, 3, 2, 1, 0
#define NARG_(...) EXPAND(ARG_N(__VA_ARGS__))
#define NARG(...) NARG_(__VA_ARGS__, SEQ_N())

// marshal
#define SAVE_2(J, x1) J[#x1] = this->x1
#define SAVE_3(J, x1, x2) J[#x1][#x2] = this->x1.x2
#define SAVE_4(J, x1, x2, x3) J[#x1][#x2][#x3] = this->x1.x2.x3
#define _SAVECALL(N) SAVE_##N
#define MCALL(N) _SAVECALL(N)
#define SAVE(...) EXPAND(MCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))

// unmarshal num
#define LOAD_2(J, x1) this->x1 = J[#x1].get<decltype(this->x1)>();
#define LOAD_3(J, x1, x2) this->x1.x2 = J[#x1][#x2].get<decltype(this->x1.x2)>();
#define LOAD_4(J, x1, x2, x3) this->x1.x2.x3 = J[#x1][#x2][#x3].get<decltype(this->x1.x2.x3)>();
#define _LOADCALL(N) LOAD_##N
#define LOADCALL(N) _LOADCALL(N)
#define LOAD(...) EXPAND(LOADCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))

/*
// unmarshal string
#define LOAD_2(J, x1) this->x1 = J[#x1].get<std::string>();
#define LOAD_3(J, x1, x2) this->x1.x2 = J[#x1][#x2].get<std::string>();
#define LOAD_4(J, x1, x2, x3) this->x1.x2.x3 = J[#x1][#x2][#x3].get<std::string>();
#define _LOADCALL(N) LOAD_##N
#define LOADCALL(N) _LOADCALL(N)
#define LOAD(...) EXPAND(LOADCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))
*/

void GridCSR::MetaData::Save(GridCSR::FS::path const & filePath) {
    
    JSON j;

    SAVE(j, dataname);
    SAVE(j, extension, row);
    SAVE(j, extension, ptr);
    SAVE(j, extension, col);
    SAVE(j, info, count, row);
    SAVE(j, info, count, col);
    SAVE(j, info, width, row);
    SAVE(j, info, width, col);
    SAVE(j, info, max_vid);

    for (size_t i = 0; i < this->grid.each.size(); i++) {
        auto const & g = this->grid.each[i];
        j["grid"][i]["name"] = g.name;
        j["grid"][i]["index"]["row"] = g.index.row;
        j["grid"][i]["index"]["col"] = g.index.col;
    }

    std::ofstream f;
    f.open(filePath);
    f << std::setw(4) << j << std::endl;
    f.close();
}

void GridCSR::MetaData::Load(GridCSR::FS::path const & filePath) {
    std::ifstream f(filePath);
    JSON j;
    f >> j;
    f.close();

    LOAD(j, dataname);
    LOAD(j, extension, row);
    LOAD(j, extension, ptr);
    LOAD(j, extension, col);
    LOAD(j, info, count, row);
    LOAD(j, info, count, col);
    LOAD(j, info, width, row);
    LOAD(j, info, width, col);
    LOAD(j, info, max_vid);

    this->grid.each.resize(j["grid"].size());

    for (size_t i = 0; i < j["grid"].size(); i++) {
        auto & g = this->grid.each[i];
        g.name = j["grid"][i]["name"];
        g.index.row = j["grid"][i]["index"]["row"].get<decltype(g.index.row)>();
        g.index.col = j["grid"][i]["index"]["col"].get<decltype(g.index.col)>();
    }
}