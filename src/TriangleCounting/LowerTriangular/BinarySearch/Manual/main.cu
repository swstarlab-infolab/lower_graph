#include <iostream>

#include "main.cuh"
#include <GridCSR/GridCSR.h>

#include <cub/device/device_scan.cuh>
#include <device_launch_parameters.h>
#include <algorithm>

using memory_list_t = std::vector<grid_pos_t>>;
using memory_lists_t = std::vector<memory_list_t>;

decltype(GridCSR::MetaData::info.width.row) getGridWidth(GridCSR::FS::path const & folderPath) {
    GridCSR::MetaData meta;
    meta.Load(GridCSR::FS::path(folderPath.string() + "meta.json"));
    return meta.info.width.row;
}

static void bringNewMemory(managed_t & managed, memory_list_t & devMemList, grid_pos_t & target) {
    if (std::find(devMemList.begin(), devMemList.end(), target) != devMemList.end()) {
        // find
        return;
    } else {
        copyToGPU();
    }
}

void launch(
    Graph & managed,
    DeviceMemory & devices,
    int const blockCount,
    int const threadCount
)
{
    std::vector<count_t> globalCount(devices.size() * devices.front().stream.size());

    size_t streamIndex = 0;
    size_t deviceIndex = 0;

    auto next = [&devices, &deviceIndex, &streamIndex]() {
        streamIndex++;
        if (streamIndex / devices[deviceIndex].stream.size()) {
            streamIndex = 0;
            deviceIndex++;
            if (deviceIndex / devices.size()) {
                deviceIndex = 0;
            }
        }
    };


    auto start = std::chrono::system_clock::now();

    for (size_t row = 0; row < managed.graph.size(); row++) {
        for (size_t col = 0; col <= row; col++) {
            for (size_t i = col; i <= row; i++) {
                auto & device = devices[deviceIndex];
                cudaSetDevice(device.deviceID); CUDACHECK();

                auto & mem = device.streamMemory[streamIndex];
                auto & stream = device.stream[streamIndex];

                //auto const & G0 = managed.graph[i][col];
                //auto const & G1 = managed.graph[row][col];
                //auto const & G2 = managed.graph[row][i];

                if (!(G0.row.count() && G1.row.count() && G2.row.count())) { continue; }

                fprintf(stdout, "GPU%d, (%ld, %ld) (%ld, %ld), (%ld, %ld)\n", device.deviceID, i, col, row, col, row, i);


                genLookupTemp <<<blockCount, threadCount, 0, stream>>> (G0, mem.lookup.temp);
                CUDACHECK();

                size_t byte = mem.cub.byte();
                cub::DeviceScan::ExclusiveSum(
                    (void*)mem.cub.data(),
                    byte,
                    mem.lookup.temp.data(),
                    mem.lookup.G0.data(),
                    mem.lookup.G0.count());
                    //stream);
                CUDACHECK();

                resetLookupTemp <<<blockCount, threadCount, 0, stream>>> (G0, mem.lookup.temp);
                CUDACHECK();

                genLookupTemp <<<blockCount, threadCount, 0, stream>>> (G2, mem.lookup.temp);
                CUDACHECK();

                cub::DeviceScan::ExclusiveSum(
                    (void*)mem.cub.data(),
                    byte,
                    mem.lookup.temp.data(),
                    mem.lookup.G2.data(),
                    mem.lookup.G2.count(),
                    stream);
                CUDACHECK();

                resetLookupTemp <<<blockCount, threadCount, 0, stream>>> (G2, mem.lookup.temp);
                CUDACHECK();

                kernel <<<blockCount, threadCount, 0, stream>>> (
                    G0, G1, G2,
                    mem.lookup.G0,
                    mem.lookup.G2,
                    mem.count);
                CUDACHECK();

                next();
            }
        }
    }

    CUDACHECK();

    for (auto & device : devices) {
        cudaSetDevice(device.deviceID); CUDACHECK();
        for (size_t s = 0; s < device.stream.size(); s++) {
            device.streamMemory[s].count.copyD2H(
                &globalCount[device.stream.size() * device.deviceID + s],
                device.stream[s]); CUDACHECK();
        }
    }

    fprintf(stdout, "WAIT KERNEL COMPLETION\n");

    for (auto & device : devices) {
        cudaSetDevice(device.deviceID); CUDACHECK();
        for (size_t i = 0; i < device.stream.size(); i++) {
            cudaStreamSynchronize(device.stream[i]); CUDACHECK();
        }
    }

    for (int i = 1; i < devices.size(); i++) {
        globalCount.front() += globalCount[i];
    }

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    std::cout << globalCount.front() << "," << elapsed.count() << std::endl;

    CUDACHECK();
}


int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <streams> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const folderPath = GridCSR::FS::path(GridCSR::FS::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount); CUDACHECK();

    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) { continue; }
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess) {
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    memory_list_t devMemList(deviceCount);

    devices_t devices(deviceCount);
    auto gridWidth = getGridWidth(folderPath);
    for (auto gpuIndex = 0; gpuIndex < devices.size(); gpuIndex++){
        devices[gpuIndex].init(gpuIndex, streamCount, gridWidth);
    }

    managed_t managed;
    managed.init(folderPath);

    launch(managed, devices, devMemlist, blockCount, threadCount);

    return 0;
}