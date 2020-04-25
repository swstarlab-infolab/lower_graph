
#include <string>

#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include "device-setting.cuh"
#include <thread>

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
    std::vector<device_setting_t> dev(deviceCount);
    std::vector<std::thread> p(deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        p[i] = std::thread([&dev, i, &streamCount, &blockCount, &threadCount, &pathFolder]{
            dev[i].init(i, streamCount, blockCount, threadCount, pathFolder);
        });
    }

    for (int i = 0; i < deviceCount; i++) {
        p[i].join();
    }

    launch(dev);

    return 0;
}