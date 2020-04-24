#include <string>

#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include "device-setting.cuh"

#include "tc.h"
#include "../common.h"

int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <streams> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("Start in-memory allocation\n");
    std::vector<device_setting_t> dev(deviceCount);
    auto gridWidth = getGridWidth(pathFolder);
    for (uint32_t i = 0; i < dev.size(); i++) {
        dev[i].init(i, streamCount, blockCount, threadCount, gridWidth);
        printf("Complete in-memory allocation: GPU%d\n", i);
    }

    printf("Start unified memory allocation\n");
    unified_setting_t umem;
    umem.load_graph(pathFolder);
    printf("Complete unified memory allocation\n");

    printf("Launch CUDA kernel\n");
    launch(dev, umem);
    printf("Complete CUDA kernel\n");

    return 0;
}