#include "converter.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <fstream>
#include <iomanip>

#include <tbb/parallel_for_each.h>

#include <string>
#include <iostream>

void GridCSRConverter::write(fs::path const & folderPath, std::string const & dataName) {
    json j;

    j["dataname"] = dataName;

    j["extension"]["row"] = "row";
    j["extension"]["ptr"] = "ptr";
    j["extension"]["col"] = "col";

    j["grid-info"]["width"]["row"] = this->gridWidth;
    j["grid-info"]["width"]["col"] = this->gridWidth;

    j["grid-info"]["count"]["row"] = this->gridCount;
    j["grid-info"]["count"]["col"] = this->gridCount;

    j["grid-info"]["max-vid"] = this->maxVID;

    size_t i = 0;
    for (uint32_t row = 0; row < this->gridCount; row++) {
        for (uint32_t col = 0; col < this->gridCount; col++) {
            auto & o = this->out.col[row][col];
            if (o.size()) {
                char _name[128];
                //sprintf(_name, "%015d-%015d", row, col);
                sprintf(_name, "%d-%d", row, col);
                j["grid"][i]["name"] = std::string(_name);
                j["grid"][i]["index"]["row"] = row;
                j["grid"][i]["index"]["col"] = col;
                i++;
            }
        }
    }

    auto rootFolder = folderPath / (dataName + "/");
    fs::create_directory(rootFolder);

    // Write JSON
    std::ofstream of;

    of.open(rootFolder.string() + std::string("meta.json"), std::ios::out);
    of << std::setw(4) << j << std::endl;
    of.close();

    // Write Row
    tbb::parallel_for_each(j["grid"].begin(), j["grid"].end(),
        [this, &j, &rootFolder](json const & i){
            std::ofstream of;
            auto & o = this->out.row[i["index"]["row"]][i["index"]["col"]];
            of.open(rootFolder.string() + std::string(i["name"]) + ".row", std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });

    // Write Ptr
    tbb::parallel_for_each(j["grid"].begin(), j["grid"].end(),
        [this, &j, &rootFolder](json const & i){
            std::ofstream of;
            auto & o = this->out.ptr[i["index"]["row"]][i["index"]["col"]];
            of.open(rootFolder.string() + std::string(i["name"]) + ".ptr", std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });

    // Write Col
    tbb::parallel_for_each(j["grid"].begin(), j["grid"].end(),
        [this, &j, &rootFolder](json const & i){
            std::ofstream of;
            auto & o = this->out.col[i["index"]["row"]][i["index"]["col"]];
            of.open(rootFolder.string() + std::string(i["name"]) + ".col", std::ios::out | std::ios::binary);
            of.write((char*)o.data(), sizeof(vertex_t) * o.size());
            of.close();
        });
}