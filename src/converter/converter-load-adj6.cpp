#include "converter.h"

#include <iostream>
#include <fstream>

#define UNIT 6

void GridCSRConverter::loadAdj6(fs::path const & folderPath) {
    uint64_t estimated = 0;

    std::ifstream fs;
    for (auto & p : fs::recursive_directory_iterator(folderPath)) {
        fs.open(p.path());

        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);

        estimated += filesize / UNIT;

        fs.close();
    }

    std::vector<uint8_t> temp;

    auto bigendian = [&temp](uint64_t const i) -> uint64_t {
        return
            (uint64_t(temp[i+0]) << (8*5)) +
            (uint64_t(temp[i+1]) << (8*4)) +
            (uint64_t(temp[i+2]) << (8*3)) +
            (uint64_t(temp[i+3]) << (8*2)) +
            (uint64_t(temp[i+4]) << (8*1)) +
            (uint64_t(temp[i+5]) << (8*0));};

    uint64_t position = 0;
    for (auto & p : fs::recursive_directory_iterator(folderPath)) {
        fs.open(p.path());

        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);

        std::cout << p.path().string() << std::endl;

        temp.resize(filesize);
        fs.read((char*)temp.data(), filesize);

        fs.close();

        for (size_t i = 0; i < filesize;) {
            vertex_t src = bigendian(i);
            i+=UNIT;

            size_t size = bigendian(i);
            i+=UNIT;

            for (size_t j = 0; j < size; j++) {
                vertex_t dst = bigendian(i);
                i+=UNIT;

                this->insert(edge_t{src, dst});
                position++;
            }
        }
    }
}