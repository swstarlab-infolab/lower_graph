#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <GridCSR/GridCSR.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>
#include <thread>

#include "manager.h"
#include "context.h"
#include <boost/fiber/buffered_channel.hpp>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
        return 0;
    }

    auto const folderPath  = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount  = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    // get total GPUs
    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);

    // prepare context
    Context ctx;
    ctx.meta.Load(folderPath / "meta.json");

    // prepare channels
    typename Manager::chanCmdReq cmdReq(16);

    std::vector<std::shared_ptr<Manager::chanCmdRes>> cmdRes(deviceCount);
    for (auto i = 0; i < deviceCount; i++) {
        cmdRes[i] = std::shared_ptr<Manager::chanCmdRes>(
            []{
                auto * p = new Manager::chanCmdRes(16);
                printf("chanCmdRes create: %p\n", p);
                return p;
            }(),
            [](void* p){
                printf("chanCmdRes delete: %p\n");
                delete p;
            });
    }

    Manager::chanLoadReq loadReq(16);

    std::vector<std::shared_ptr<Manager::chanLoadRes>> loadRes(deviceCount);
    for (auto i = 0; i < deviceCount; i++) {
        loadRes[i] = std::shared_ptr<Manager::chanLoadRes>(
            []{
                auto * p = new Manager::chanLoadRes(16);
                printf("chanLoadRes create: %p\n", p);
                return p;
            }(),
            [](void* p){
                printf("chanLoadRes delete: %p\n");
                delete p;
            });
    }

    // prepare thread
    std::thread cmdMgr;
    std::vector<std::thread> gpuMgr(deviceCount);

    // launch thread
    cmdMgr = std::thread([&]{ Manager::commander(ctx, cmdReq, cmdRes); });
    for (size_t i = 0; i < gpuMgr.size(); i++) {
        gpuMgr[i] = std::thread([&, i]{
            Manager::Execute::GPU(
                ctx,
                cmdReq,
                *cmdRes[i].get(),
                loadReq,
                *loadRes[i].get(),
                i); });
    }

    // join thread
    cmdMgr.join();
    for (size_t i = 0; i < gpuMgr.size(); i++) {
        gpuMgr[i].join();
    }

    return 0;
}