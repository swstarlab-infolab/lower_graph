#include <dataformat.h>
#include "common.h"

#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <algorithm>


int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const pathMeta = fs::path(pathFolder.string() + "meta.json");

    std::vector<gridInfo_t> gridInfo;
    std::vector<gridData_t> gridData;
    readGCSR(pathMeta, gridInfo, gridData);

    launch(gridInfo, gridData);

    return 0;
}