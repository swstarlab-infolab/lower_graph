#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <concurrentqueue/blockingconcurrentqueue.h>
#include <GridCSR/GridCSR.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>
#include <thread>

#include "manager.h"
#include "context.h"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
        return 0;
    }

    auto const folderPath  = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount  = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);


    Context ctx;
    ctx.meta.Load(folderPath / "meta.json");

    Channel<Manager::MessageType::CommandReq> cmdReq;
    Channel<Manager::MessageType::CommandRes> cmdRes;
    Channel<Manager::MessageType::LoadReq> loadReq;
    Channel<Manager::MessageType::LoadRes> loadRes;

    std::thread cmdMgr([&]{ Manager::commander(ctx, cmdReq, cmdRes); });
    std::thread gpuMgr([&]{ Manager::Execute::GPU(ctx, cmdReq, cmdRes, loadReq, loadRes); });

    cmdMgr.join();
    gpuMgr.join();

    cudaDeviceReset();

    return 0;
}

/*
int main(int argc, char* argv[]) {
    GlobalContext globalCtx;
    CPUContext cpuCtx;
    std::vector<GPUContext> gpuCtx(devCount);

    std::thread cpuMgr;
    std::vector<std::thread> gpuMgr(devCount);

    initGlobal(globalCtx, folderPath);

    auto jobMgr = std::thread([&]{
        auto const MAXROW = globalCtx.meta.info.count.row;
        auto const MAXJOB = ((MAXROW) * (MAXROW + 1) * (MAXROW + 2)) / 6;

        size_t job = 0;
        for (size_t row = 0; row < MAXROW; row++) {
            for (size_t col = 0; col <= row; col++) {
                for (size_t i = col; i <= row; i++) {
                    QueuePairElement<JobCommandReq> req;
                    req.data.G[0] = {i, col};
                    req.data.G[1] = {row, col};
                    req.data.G[2] = {row, i};
                    req.end = (job == MAXJOB - 1);
                    qpCommand.req.enqueue(req);

                    if (req.end) {
                        goto RES_WAIT;
                    }

                    job++;
                }
            }
        }

        RES_WAIT:
        while (true) {
            QueuePairElement<JobCommandRes> res;
            qpCommand.res.wait_dequeue(res);

            if (res.data.success) {
                printf("Success (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                    res.data.G[0].row,
                    res.data.G[0].col,
                    res.data.G[1].row,
                    res.data.G[1].col,
                    res.data.G[2].row,
                    res.data.G[2].col,
                    res.data.triangle,
                    res.data.elapsed);
            } else {
                printf("Failed (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                    res.data.G[0].row,
                    res.data.G[0].col,
                    res.data.G[1].row,
                    res.data.G[1].col,
                    res.data.G[2].row,
                    res.data.G[2].col,
                    res.data.triangle,
                    res.data.elapsed);
            }

            if (res.end) { break; }
        }
    });

    auto gpuMgr = std::thread([&]{
        while (true) {
            QueuePairElement<JobCommandReq> req;
            qpCommand.req.wait_dequeue(req);

            QueuePairElement<JobCommandRes> res;
            res.data.G[0].row = req.data.G[0].row;
            res.data.G[0].col = req.data.G[0].col;
            res.data.G[1].row = req.data.G[1].row;
            res.data.G[1].col = req.data.G[1].col;
            res.data.G[2].row = req.data.G[2].row;
            res.data.G[2].col = req.data.G[2].col;


            res.data.triangle = 0;
            res.data.elapsed = 1.0f;
            res.data.success = true;

            res.end = req.end;
            qpCommand.res.enqueue(res);

            if (res.end) { break; }
        }
    });

    jobMgr.join();
    gpuMgr.join();

    return 0;

    ///////////////////////


/*
    cpuMgr = std::thread([&]{
        TryCatch(
            initCPU(cpuCtx);
        );
    });

    for (int i = 0; i < devCount; i++) {
        gpuMgr[i] = std::thread([&, i]{ 
            TryCatch(
                gpuCtx[i].gpuID = i;
                initGPU(gpuCtx[i]);
            );
        });
    }

    // wait
    cpuMgr.join();
    for (int i = 0; i < devCount; i++) {
        gpuMgr[i].join();
    }

    cpuMgr = std::thread([&]{
        TryCatch(
            mainCPU(globalCtx, cpuCtx);
        );
    });

    for (int i = 0; i < devCount; i++) {
        gpuMgr[i] = std::thread([&, i]{ 
            TryCatch(
                mainGPU(globalCtx, gpuCtx[i]);
            );
        });
    }

    // wait
    cpuMgr.join();
    for (int i = 0; i < devCount; i++) {
        gpuMgr[i].join();
    }
    */