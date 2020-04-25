#include <iostream>

#include "main.cuh"
#include "../meta.h"

#include <cub/device/device_scan.cuh>

decltype(meta_t::info.width.row) getGridWidth(fs::path const & folderPath) {
    meta_t meta;
    meta.unmarshal_from_file(fs::path(folderPath.string() + "meta.json"));
    return meta.info.width.row;
}

void launch(managed_t & managed, devices_t & devices, int const blockCount, int const threadCount) {
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

    // THIS FUNCTION IS KEY POINT FOR HIGH SPEED !!!
    auto setGrid = [](int const did, grid_t const & g, cudaStream_t & s){
        auto _tmpfunc = [](int const did, CudaManagedMemory<vertex_t> const & arr, cudaStream_t & s){
            // Especially these two lines!!!
            arr.advise(did, cudaMemoryAdvise::cudaMemAdviseSetReadMostly);
            arr.prefetchAsync(did, s);
        };
        _tmpfunc(did, g.row, s);
        _tmpfunc(did, g.ptr, s);
        _tmpfunc(did, g.col, s);
    };

    auto start = std::chrono::system_clock::now();

    for (size_t row = 0; row < managed.graph.size(); row++) {
        for (size_t col = 0; col <= row; col++) {
            for (size_t i = col; i <= row; i++) {
                auto & device = devices[deviceIndex];
                auto const & G0 = managed.graph[i][col];
                auto const & G1 = managed.graph[row][col];
                auto const & G2 = managed.graph[row][i];

                auto & mem = device.streamMemory[streamIndex];
                auto & stream = device.stream[streamIndex];

                cudaSetDevice(device.deviceID); CUDACHECK();

                if (!(G0.row.count() && G1.row.count() && G2.row.count())) { continue; }

                // launch kernels

                setGrid(device.deviceID, G0, stream);
                setGrid(device.deviceID, G1, stream);
                setGrid(device.deviceID, G2, stream);

                genLookupTemp <<<blockCount, threadCount, 0, stream>>> (G0, mem.lookup.temp);

                size_t byte = mem.cub.byte();
                cub::DeviceScan::ExclusiveSum(
                    (void*)mem.cub.data(),
                    byte,
                    mem.lookup.temp.data(),
                    mem.lookup.G0.data(),
                    mem.lookup.G0.count(),
                    stream);

                resetLookupTemp <<<blockCount, threadCount, 0, stream>>> (G0, mem.lookup.temp);

                genLookupTemp <<<blockCount, threadCount, 0, stream>>> (G2, mem.lookup.temp);

                cub::DeviceScan::ExclusiveSum(
                    (void*)mem.cub.data(),
                    byte,
                    mem.lookup.temp.data(),
                    mem.lookup.G2.data(),
                    mem.lookup.G2.count(),
                    stream);

                resetLookupTemp <<<blockCount, threadCount, 0, stream>>> (G2, mem.lookup.temp);

                G1.row.prefetchAsync(deviceIndex, stream);
                G1.ptr.prefetchAsync(deviceIndex, stream);
                G1.col.prefetchAsync(deviceIndex, stream);
                kernel <<<blockCount, threadCount, 0, stream>>> (
                    G0, G1, G2,
                    mem.lookup.G0,
                    mem.lookup.G2,
                    mem.count);

                next();
            }
        }
    }


    for (auto & device : devices) {
        cudaSetDevice(device.deviceID); CUDACHECK();
        for (size_t s = 0; s < device.stream.size(); s++) {
            device.streamMemory[s].count.copyD2H(
                &globalCount[device.stream.size() * device.deviceID + s],
                device.stream[s]); CUDACHECK();
        }
    }

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
}

int main(int argc, char * argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <streams> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount); CUDACHECK();

    auto p = [](int src, int dst){
        cudaSetDevice(src); CUDACHECK();
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, src, dst); CUDACHECK();
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(dst, 0); CUDACHECK();
            //printf("Peer Access Enabled: GPU%d -> GPU%d\n", src, dst);
        }
    };

    for (auto i = 0; i < deviceCount; i++) {
        for (auto j = 0; j < i; j++) {
            p(i, j);
            p(j, i);
        }
    }

    devices_t devices(deviceCount);
    auto gridWidth = getGridWidth(folderPath);
    for (auto gpuIndex = 0; gpuIndex < devices.size(); gpuIndex++){
        devices[gpuIndex].init(gpuIndex, streamCount, gridWidth);
    }

    managed_t managed;
    managed.init(folderPath);

    launch(managed, devices, blockCount, threadCount);

    return 0;
}