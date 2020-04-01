#include "converter.h"
#include "../meta.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <tbb/parallel_for_each.h>

#include <string>
#include <iostream>

#include <fstream>

void GridCSRConverter::storeGCSR(fs::path const & folderPath, std::string const & dataName) {
    meta_t m;

    m.dataname = dataName;
    m.extension.row = "row";
    m.extension.ptr = "ptr";
    m.extension.col = "col";

    m.info.width.row = this->gridWidth;
    m.info.width.col = this->gridWidth;

    m.info.count.row = this->gridCount;
    m.info.count.col = this->gridCount;

    m.info.max_vid = this->maxVID;

    size_t i = 0;
    for (uint32_t row = 0; row < this->gridCount; row++) {
        for (uint32_t col = 0; col < this->gridCount; col++) {
            auto & o = this->out.col[row][col];
            if (o.size()) {
                i++;
            }
        }
    }

    m.grid.each.resize(i);

    i = 0;
    for (uint32_t row = 0; row < this->gridCount; row++) {
        for (uint32_t col = 0; col < this->gridCount; col++) {
            auto & o = this->out.col[row][col];
            if (o.size()) {
                char _name[128];
                //sprintf(_name, "%015d-%015d", row, col);
                sprintf(_name, "%d-%d", row, col);
                m.grid.each[i].name = std::string(_name);
                m.grid.each[i].index.row = row;
                m.grid.each[i].index.col = col;
                i++;
            }
        }
    }

    auto rootFolder = folderPath / (dataName + "/");
    fs::create_directory(rootFolder);

    m.marshal_to_file(rootFolder.string() + std::string("meta.json"));

    // Write Row
    tbb::parallel_for_each(m.grid.each.begin(), m.grid.each.end(),
        [this, &m, &rootFolder](decltype(m.grid.each.front()) const & i){
            std::ofstream of;
            auto & o = this->out.row[i.index.row][i.index.col];
            of.open(rootFolder.string() + std::string(i.name) + "." + m.extension.row, std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });

    // Write Ptr
    tbb::parallel_for_each(m.grid.each.begin(), m.grid.each.end(),
        [this, &m, &rootFolder](decltype(m.grid.each.front()) const & i){
            std::ofstream of;
            auto & o = this->out.ptr[i.index.row][i.index.col];
            of.open(rootFolder.string() + std::string(i.name) + "." + m.extension.ptr, std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });

    // Write Col
    tbb::parallel_for_each(m.grid.each.begin(), m.grid.each.end(),
        [this, &m, &rootFolder](decltype(m.grid.each.front()) const & i){
            std::ofstream of;
            auto & o = this->out.col[i.index.row][i.index.col];
            of.open(rootFolder.string() + std::string(i.name) + "." + m.extension.col, std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });
}