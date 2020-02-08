#include <dataformat.h>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <algorithm>

using json = nlohmann::json;

static void readMeta(fs::path const & path, std::vector<gridInfo_t> & info) {
    //std::cout << ">>> Read and Parse Metadata" << std::endl;
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
    info.resize(j["grids"].size());
    auto folder = fs::path(path.parent_path().string() + "/");

    uint32_t i = 0;
    for (auto & l : j["grids"]) {
        auto & g = info[i];
        auto basicstr = folder.string() + std::string(l["filename"]) + ".";

        g.path.row = fs::path(basicstr + std::string(j["ext"]["row"]));
        g.path.ptr = fs::path(basicstr + std::string(j["ext"]["ptr"]));
        g.path.col = fs::path(basicstr + std::string(j["ext"]["col"]));

        if (!fs::exists(g.path.row)) {
            std::cerr << "Not exists: " << g.path.ptr.string() << std::endl;
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

        g.pos.row = l["row"];
        g.pos.col = l["col"];

        i++;
    }
}

void readGCSR(fs::path const & meta_path, std::vector<gridInfo_t> & info, std::vector<gridData_t> & data) {
    //std::cout << ">>> Read Grid Data" << std::endl;
    readMeta(meta_path, info);

    auto grids = info.size();
    data.resize(grids);

    std::fstream fs;

    auto load = [&fs](fs::path const & p, std::vector<vertex_t> & d){
        fs.open(p);
        fs.seekg(0, std::ios::end);
        auto const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        d.resize(filesize / sizeof(vertex_t));
        fs.read((char*)d.data(), filesize);
        fs.close();
        //std::cout << p.string() << " size: " << filesize << " bytes" << std::endl;
    };

    for (uint32_t i = 0; i < grids; i++) {
        load(info[i].path.row, data[i].row);
        load(info[i].path.ptr, data[i].ptr);
        load(info[i].path.col, data[i].col);
    }
}