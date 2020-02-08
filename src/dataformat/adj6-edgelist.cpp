#include <dataformat.h>

#include <iostream>
#include <fstream>

#define UNIT 6

void readADJ6(fs::path const & folder, std::vector<edge_t> & edgelist) {
    std::cout << ">>> Read ADJ6 Files" << std::endl;

    uint64_t estimated = 0;

    std::ifstream fs;
    for (auto & p : fs::recursive_directory_iterator(folder)) {
        fs.open(p.path());

        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);

        estimated += filesize / UNIT;

        fs.close();
    }

    edgelist.resize(estimated);

    std::cout << "complete: allocate memory of edgelist and buffer for ADJ6 files" << std::endl;

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
    for (auto & p : fs::recursive_directory_iterator(folder)) {
        fs.open(p.path());

        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);

        std::cout << p.path().string() << std::endl;

        temp.resize(filesize);
        fs.read((char*)temp.data(), filesize);

        fs.close();

        for (uint64_t i = 0; i < filesize;) {
            uint64_t src = bigendian(i);
            i+=UNIT;

            uint64_t size = bigendian(i);
            i+=UNIT;

            for (uint64_t j = 0; j < size; j++) {
                uint64_t dst = bigendian(i);
                i+=UNIT;

                edgelist[position] = edge_t{vertex_t(src), vertex_t(dst)};
                position++;
            }
        }
    }

    edgelist.resize(position);

    std::cout << "complete: file read and parse to fill edgelist" << std::endl;
}