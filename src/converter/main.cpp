#include "common.h"
#include "converter.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "usage" << std::endl
            << argv[0] << " <type> <input_folder_path> <grid_width_power> <output_folder_path> <db_name>" << std::endl
            << "type: tsv, adj6" << std::endl;
        return 0;
    }

    auto const inFolder = fs::path(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
    auto const grid_width_power = strtol(argv[3], nullptr, 10);
    auto const outFolder = fs::path(fs::path(std::string(argv[4]) + "/").parent_path().string() + "/");
    auto const outDataName = std::string(argv[5]);

    GridCSRConverter converter(1 << grid_width_power);

    if (std::string(argv[1]) == "tsv") {
        converter.loadTSV(inFolder);
    } else if (std::string(argv[1]) == "adj6") {
        converter.loadAdj6(inFolder);
    } else {
        return 0;
    }

    converter.run();
    converter.storeGCSR(outFolder, outDataName);

    return 0;
}