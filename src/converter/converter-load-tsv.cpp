#include "converter.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

void GridCSRConverter::loadTSV(fs::path const & folder) {
    std::string temp;

    for (auto & p : fs::recursive_directory_iterator(folder)) {

        std::ifstream fs;
        fs.open(p.path());

        fs.seekg(0, std::ios::end);
        uint64_t const filesize = fs.tellg();
        fs.seekg(0, std::ios::beg);

        temp.resize(filesize);
        fs.read(temp.data(), filesize);
        fs.close();

        std::string current_line;
        std::istringstream iss(temp);
        while (std::getline(iss, current_line)) {
            if (current_line[0] == '#') {
                continue;
            }

            char *ptr, *str = const_cast<char*>(current_line.c_str());
            ptr = strtok(str, "\t");
            vertex_t src = atoi(ptr);
            ptr = strtok(nullptr, "\t");
            vertex_t dst = atoi(ptr);
            this->insert(edge_t{src, dst});
        }
    }
}