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
#include <boost/fiber/all.hpp>

template <typename Type, size_t Size=128>
auto makeChan() {
    return std::shared_ptr<Type>(
        new Type(Size),
        [](Type* p){
            if (p != nullptr) {
                delete p;
            }
        });
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
        return 0;
    }

    auto const folderPath  = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    //auto const folderPath  = boost::filesystem::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
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
        cmdRes[i] = makeChan<Manager::chanCmdRes>();
    }

    Manager::chanLoadReq loadReq(16);

    std::vector<std::shared_ptr<Manager::chanLoadRes>> loadRes(deviceCount);
    for (auto i = 0; i < deviceCount; i++) {
        loadRes[i] = makeChan<Manager::chanLoadRes>();
    }

    // prepare thread
    std::vector<boost::fibers::fiber> gpuMgr(deviceCount);

    // launch thread
    auto cmdMgr = boost::fibers::fiber([&]{ Manager::commander(ctx, cmdReq, cmdRes); });
    for (size_t i = 0; i < gpuMgr.size(); i++) {
        gpuMgr[i] = boost::fibers::fiber([&ctx, &cmdReq, &cmdRes, &loadReq, &loadRes, i]{
            Manager::Execute::GPU(
                ctx,
                cmdReq,
                cmdRes[i].get(),
                loadReq,
                loadRes[i].get(),
                i); });
    }

    // join thread
    cmdMgr.join();
    for (size_t i = 0; i < gpuMgr.size(); i++) {
        gpuMgr[i].join();
    }

    return 0;
}