#include "main.cuh"

#include <fstream>
#include "../meta.h"

void managed_t::init(fs::path const & folderPath) {
    meta_t meta;
    meta.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));

    std::ifstream f;
    this->graph.resize(meta.info.count.row);
    for (auto & g : this->graph) {
        g.resize(meta.info.count.row);
    }

    auto loader = [](std::ifstream & f, fs::path const & path, CudaManagedMemory<vertex_t> & mem){
        f.open(path);
        f.seekg(0, std::ios::end);
        auto const fileSize = f.tellg();
        f.seekg(0, std::ios::beg);
        mem.mallocByte(fileSize); CUDACHECK();
        f.read((char*)mem.data(), fileSize);
        f.close();
    };

    for (auto i = 0; i < meta.grid.each.size(); i++) {
        auto const baseString = folderPath.string() + std::string(meta.grid.each[i].name) + ".";

        auto const pathRow = fs::path(baseString + meta.extension.row);
        auto const pathPtr = fs::path(baseString + meta.extension.ptr);
        auto const pathCol = fs::path(baseString + meta.extension.col);

        if (!(fs::exists(pathRow) && fs::exists(pathPtr) && fs::exists(pathCol))) {
            printf("Not exists: %s\n", meta.grid.each[i].name.c_str());
            exit(EXIT_FAILURE);
        }

        size_t const rowIndex = meta.grid.each[i].index.row;
        size_t const colIndex = meta.grid.each[i].index.col;

        std::ifstream f;
        loader(f, pathRow, this->graph[rowIndex][colIndex].row);
        loader(f, pathPtr, this->graph[rowIndex][colIndex].ptr);
        loader(f, pathCol, this->graph[rowIndex][colIndex].col);
    }

    cudaDeviceSynchronize();
}