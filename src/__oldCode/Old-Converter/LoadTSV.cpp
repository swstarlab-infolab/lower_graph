#include "Converter.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

void GridCSRConverter::loadTSV(GridCSR::FS::path const & folder) {
    std::string temp;

    for (auto & p : GridCSR::FS::recursive_directory_iterator(folder)) {

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
            GridCSR::Vertex src = atoi(ptr);
            ptr = strtok(nullptr, "\t");
            GridCSR::Vertex dst = atoi(ptr);
            if (this->mode.is_directed) {
                using type = GridCSRConverterMode::SortingType;
                switch (this->mode.sort_type) {
                case type::not_sort:
                    this->insert(GridCSR::Edge{src, dst});
                    break;
                case type::lower_triangle:
                    this->insertLowerTriangle(GridCSR::Edge{src, dst});
                    break;
                case type::degree:
                    this->insertUndirectedDegree(GridCSR::Edge{src, dst});
                    break;
                }
            } else {
                this->insertUndirected(GridCSR::Edge{src, dst});
            }
        }
    }
}