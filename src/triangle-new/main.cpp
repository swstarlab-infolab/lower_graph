
#include <string>

#include <fstream>
#include <iostream>
#include <vector>

#include <algorithm>
#include <cuda_runtime.h>
#include "device-setting.cuh"

#include "tc.h"
#include "../common.h"

#define CUDA_BLOCKS 80
#define CUDA_THREADS 1024
#define CUDA_STREAMS 1

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path>" << std::endl;
        return 0;
    }

    auto const pathFolder = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::vector<device_setting_t> dev(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        dev[i].init(i, CUDA_STREAMS, CUDA_BLOCKS, CUDA_THREADS, pathFolder);
    }

    cudaSetDevice(0); CUDACHECK();
    cudaDeviceReset(); CUDACHECK();
    launch(dev);

    return 0;
}