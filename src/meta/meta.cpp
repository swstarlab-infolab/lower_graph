#include "../meta.h"

#include <fstream>
#include <iostream>
#include <iomanip>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define EXPAND(x) x

// Count the number of arguments
#define ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define SEQ_N 8, 7, 6, 5, 4, 3, 2, 1, 0
#define NARG_(...) EXPAND(ARG_N(__VA_ARGS__))
#define NARG(...) NARG_(__VA_ARGS__, SEQ_N())

// marshal
#define M_2(J, x1) J[#x1] = this->x1
#define M_3(J, x1, x2) J[#x1][#x2] = this->x1.x2
#define M_4(J, x1, x2, x3) J[#x1][#x2][#x3] = this->x1.x2.x3
#define _MCALL(N) M_##N
#define MCALL(N) _MCALL(N)
#define M(...) EXPAND(MCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))

// unmarshal num
#define UN_2(J, x1) this->x1 = J[#x1].get<decltype(this->x1)>();
#define UN_3(J, x1, x2) this->x1.x2 = J[#x1][#x2].get<decltype(this->x1.x2)>();
#define UN_4(J, x1, x2, x3) this->x1.x2.x3 = J[#x1][#x2][#x3].get<decltype(this->x1.x2.x3)>();
#define _UNCALL(N) UN_##N
#define UNCALL(N) _UNCALL(N)
#define UN(...) EXPAND(UNCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))

// unmarshal string
#define US_2(J, x1) this->x1 = J[#x1].get<std::string>();
#define US_3(J, x1, x2) this->x1.x2 = J[#x1][#x2].get<std::string>();
#define US_4(J, x1, x2, x3) this->x1.x2.x3 = J[#x1][#x2][#x3].get<std::string>();
#define _USCALL(N) US_##N
#define USCALL(N) _USCALL(N)
#define US(...) EXPAND(USCALL(NARG(__VA_ARGS__)) (__VA_ARGS__))

void meta_t::marshal_to_file(fs::path const & filePath) {
    
    json j;

    M(j, dataname);
    M(j, extension, row);
    M(j, extension, ptr);
    M(j, extension, col);
    M(j, info, count, row);
    M(j, info, count, col);
    M(j, info, width, row);
    M(j, info, width, col);
    M(j, info, max_vid);

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

void meta_t::unmarshal_from_file(fs::path const & filePath) {
    std::ifstream f(filePath);
    json j;
    f >> j;
    f.close();

    US(j, dataname);
    US(j, extension, row);
    US(j, extension, ptr);
    US(j, extension, col);
    UN(j, info, count, row);
    UN(j, info, count, col);
    UN(j, info, width, row);
    UN(j, info, width, col);
    UN(j, info, max_vid);

    this->grid.each.resize(j["grid"].size());

    for (size_t i = 0; i < j["grid"].size(); i++) {
        auto & g = this->grid.each[i];
        g.name = j["grid"][i]["name"];
        g.index.row = j["grid"][i]["index"]["row"].get<decltype(g.index.row)>();
        g.index.col = j["grid"][i]["index"]["col"].get<decltype(g.index.col)>();
    }
}