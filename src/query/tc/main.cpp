#include <tbb/tbb.h>
#include <nlohmann/json.hpp>

#include <grid.h>
#include "common.h"

#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <algorithm>

using json = nlohmann::json;

static void readMeta(fs::path const & path, std::vector<gridInfo_t> & info) {
    // Read JSON file as string
    std::fstream fs;
    fs.open(path);
    fs.seekg(0, std::ios::end);
    auto const filesize = fs.tellg();
    fs.seekg(0, std::ios::beg);
    std::string input;
    input.resize(filesize);
    fs.read(input.data(), filesize);
    fs.close();
    
    // Parsing JSON from string
    auto j = json::parse(input);

    // Generate file path from metadata JSON file
    info.resize(j["grid-list"].size());
    auto folder = fs::path(path.parent_path().string() + "/");

    uint32_t i = 0;
    for (auto & l : j["grid-list"]) {
        auto & g = info[i];
        auto basicstr = folder.string() + std::string(l["file-name"]) + ".";
        g.path.row = fs::path(basicstr + std::string(j["ext"]["row"]));
        g.path.ptr = fs::path(basicstr + std::string(j["ext"]["column-pointer"]));
        g.path.col = fs::path(basicstr + std::string(j["ext"]["column"]));
        if (!fs::exists(g.path.row)) {
            std::cerr << "Not exists: " << g.path.row.string() << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!fs::exists(g.path.ptr)) {
            std::cerr << "Not exists: " << g.path.ptr.string() << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!fs::exists(g.path.col)) {
            std::cerr << "Not exists: " << g.path.col.string() << std::endl;
            exit(EXIT_FAILURE);
        }
        g.pos.row = l["row-position"];
        g.pos.col = l["column-position"];
        i++;
    }
}

static void readData(std::vector<gridInfo_t> const & info, std::vector<gridData_t> & data) {
    auto grids = info.size();
    data.resize(grids);

    std::fstream fs;

    auto load = [&fs](fs::path const & p, std::vector<vertex_t> & d){
        fs.open(p);
        fs.seekg(0, std::ios::end);
        auto const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        d.resize(filesize);
        fs.read((char*)d.data(), filesize);
        fs.close();
    };

    for (uint32_t i = 0; i < grids; i++) {
        load(info[i].path.row, data[i].row);
        load(info[i].path.ptr, data[i].ptr);
        load(info[i].path.col, data[i].col);
    }
}

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const pathMeta = fs::path(pathFolder.string() + "meta.json");

    std::vector<gridInfo_t> gridInfo;
    readMeta(pathMeta, gridInfo);

    //std::vector<gridData_t> gridData;
    //readData(gridInfo, gridData);

    //launch(gridData);

    return 0;
}