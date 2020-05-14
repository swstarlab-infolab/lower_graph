// stream
#include <iostream>
#include <fstream>

// data structure
#include <vector>
#include <map>

// multithread
#include <thread>
#include <mutex>

// cuda
#include <cuda_runtime.h>

// custom global header
#include <cub/device/device_scan.cuh>
#include <GridCSR/GridCSR.h>

// custom local header
#include "main.cuh"
#include "GPUMemory.cuh"
#include "CPUMemory.h"
#include "error.h"
#include <taskflow/taskflow.hpp>

// memory management
#include <stdlib.h>



int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "usage" << std::endl << argv[0] << " <folder_path> <streams> <blocks> <threads>" << std::endl;
        return 0;
    }

    auto const folderPath = GridCSR::FS::path(GridCSR::FS::path(std::string(argv[1]) + "/").parent_path().string() + "/");
    auto const streamCount = strtol(argv[2], nullptr, 10);
    auto const blockCount = strtol(argv[3], nullptr, 10);
    auto const threadCount = strtol(argv[4], nullptr, 10);


    int devCount = -1;
    ThrowCuda(cudaGetDeviceCount(&devCount));

    GlobalContext globalCtx;
    CPUContext cpuCtx;
    std::vector<GPUContext> gpuCtx(devCount);


    /*
    std::thread cpuMgr;
    std::vector<std::thread> gpuMgr(devCount);
    */

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

    return 0;
}