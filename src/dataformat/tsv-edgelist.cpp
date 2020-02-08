#include <dataformat.h>

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

static size_t linecountTSV(fs::path const & file) {
    std::ifstream fs;
    fs.open(file);

    auto state_backup = fs.rdstate();
    fs.clear();
    auto pos_backup = fs.tellg();

    fs.seekg(0, std::ios::beg);
    size_t lf_count = std::count(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>(), '\n');
    fs.unget();
    if (fs.get() != '\n' ) {
        ++lf_count;
    }
    fs.clear();
    fs.seekg(pos_backup);
    fs.setstate(state_backup);
    fs.close();

    return lf_count;
}

void readTSV(fs::path const & folder, std::vector<edge_t> & edgelist) {
    std::cout << ">>> Read TSV Files" << std::endl;


    std::string temp;
    size_t edges = 0;

    for (auto & p : fs::recursive_directory_iterator(folder)) {
        edgelist.resize(edgelist.size() + linecountTSV(p.path()));

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
            edgelist[edges].src = atoi(ptr);
            ptr = strtok(nullptr, "\t");
            edgelist[edges].dst = atoi(ptr);
            edges++;
        }

        edgelist.resize(edges);
    }
}