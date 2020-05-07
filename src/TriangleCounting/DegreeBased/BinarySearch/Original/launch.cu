#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <array>

#include "tc.h"
#include "device-setting.cuh"

#include <cmath>

__device__ static uint32_t ulog2floor(uint32_t x) {
    uint32_t r, q;
    r = (x > 0xFFFF) << 4; x >>= r;
    q = (x > 0xFF  ) << 3; x >>= q; r |= q;
    q = (x > 0xF   ) << 2; x >>= q; r |= q;
    q = (x > 0x3   ) << 1; x >>= q; r |= q;
                                   
    return (r | (x >> 1));
}

__device__ static void intersection(
    GridCSR::Vertex const * Arr,
    uint32_t const ArrLen,
    GridCSR::Vertex const candidate,
    count_t * count)
{
    //auto const maxLevel = uint32_t(ceil(log2(ArrLen + 1))) - 1;
    // ceil(log2(a)) == floor(log2(a-1))+1
    auto const maxLevel = ulog2floor(ArrLen);

    int now = (ArrLen - 1) >> 1;

    for (uint32_t level = 0; level <= maxLevel; level++) {
        auto const movement = 1 << (maxLevel - level - 1);

        if (now < 0) {
            now += movement;
        } else if (ArrLen <= now) {
            now -= movement;
        } else {
            if (Arr[now] < candidate) {
                now += movement;
            } else if (candidate < Arr[now]) {
                now -= movement;
            } else {
                (*count)++;
                break;
            }
        }
    }
}

__device__ static int binarySearchPosition(
    GridCSR::Vertex const * Arr,
    uint32_t const ArrLen,
    GridCSR::Vertex const candidate)
{
    //auto const maxLevel = uint32_t(ceil(log2(ArrLen + 1))) - 1;
    // ceil(log2(a)) == floor(log2(a-1))+1
    auto const maxLevel = ulog2floor(ArrLen);

    int now = (ArrLen - 1) >> 1;

    for (uint32_t level = 0; level <= maxLevel; level++) {
        auto const movement = 1 << (maxLevel - level - 1);

        if (now < 0) {
            now += movement;
        } else if (ArrLen <= now) {
            now -= movement;
        } else {
            if (Arr[now] < candidate) {
                now += movement;
            } else if (candidate < Arr[now]) {
                now -= movement;
            } else {
                return now;
            }
        }
    }

    return -1;
}

struct kernelParameter {
    struct {
        struct {
            GridCSR::Vertex *row, *ptr, *col;
            GridCSR::Vertex rows, ptrs, cols;
        } p, a, b;
    } G;

    count_t *count;
};

__global__ static void kernel(kernelParameter kp)
{
    count_t mycount = 0;

    auto const & G = kp.G;
    __shared__ int SHARED[1024];

    for (GridCSR::Vertex prowIter = blockIdx.x; prowIter < G.p.rows; prowIter+=gridDim.x) {
        int const apos = binarySearchPosition(G.a.row, G.a.rows, G.p.row[prowIter]);
        if (apos == -1) { continue; }
        int const alen = G.a.ptr[apos+1] - G.a.ptr[apos];

        auto const Gpptr_s = G.p.ptr[prowIter];
        auto const Gpptr_e = G.p.ptr[prowIter+1];

        for (GridCSR::Vertex pcolIter = Gpptr_s; pcolIter < Gpptr_e; pcolIter+=blockDim.x) {
            SHARED[threadIdx.x]
                    = (pcolIter + threadIdx.x < Gpptr_e) ?
                        binarySearchPosition(G.b.row, G.b.rows, G.p.col[pcolIter+threadIdx.x]) : -2;

            __syncthreads();

            for (uint32_t t = 0; t < blockDim.x; t++) {
                int const bpos = SHARED[t];
                if (bpos == -2) { break;} // very important for runtime. (x2 speedup)
                if (bpos == -1) { continue; }

                int const blen = G.b.ptr[bpos+1] - G.b.ptr[bpos];

                if (alen > blen) {
                    for (GridCSR::Vertex bcolIter = G.b.ptr[bpos]+threadIdx.x; bcolIter < G.b.ptr[bpos+1]; bcolIter+=blockDim.x) {
                        intersection(&G.a.col[G.a.ptr[apos]], alen, G.b.col[bcolIter], &mycount);
                    }
                } else {
                    for (GridCSR::Vertex acolIter = G.a.ptr[apos]+threadIdx.x; acolIter < G.a.ptr[apos+1]; acolIter+=blockDim.x) {
                        intersection(&G.b.col[G.b.ptr[bpos]], blen, G.a.col[acolIter], &mycount);
                    }
                }
            }

            __syncthreads();
        }
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(kp.count, mycount); }
}

void launch(std::vector<device_setting_t> & dev) {
    std::vector<count_t> globalCount(dev.size() * dev.front().gpu.setting.stream.size());

    size_t streamIndex = 0;
    size_t deviceIndex = 0;

    auto next = [&dev, &deviceIndex, &streamIndex]() {
        streamIndex++;
        if (streamIndex / dev[deviceIndex].gpu.setting.stream.size()) {
            streamIndex = 0;
            deviceIndex++;
            if (deviceIndex / dev.size()) {
                deviceIndex = 0;
            }
        }
    };

    auto const gridCount = dev.front().mem.graph_meta.info.count.row;
    auto const gridWidth = dev.front().mem.graph_meta.info.width.row;

    auto start = std::chrono::system_clock::now();


    for (size_t row = 0; row < gridCount; row++) {
        for (size_t col = 0; col < gridCount; col++) {
            auto & d = dev[deviceIndex];
            cudaSetDevice(d.gpu.meta.index); CUDACHECK();

            auto & mem = d.mem.stream[streamIndex];
            auto & setting = d.gpu.setting;

            auto const & Gp = d.mem.graph[row][col];

            for (size_t col2 = 0; col2 < gridCount; col2++) {
                auto const & Ga = d.mem.graph[row][col2];
                auto const & Gb = d.mem.graph[col][col2];

                kernelParameter kp;

                kp.G.p.row = Gp.row.ptr; kp.G.p.ptr = Gp.ptr.ptr; kp.G.p.col = Gp.col.ptr;
                kp.G.a.row = Ga.row.ptr; kp.G.a.ptr = Ga.ptr.ptr; kp.G.a.col = Ga.col.ptr;
                kp.G.b.row = Gb.row.ptr; kp.G.b.ptr = Gb.ptr.ptr; kp.G.b.col = Gb.col.ptr;

                kp.G.p.rows = Gp.row.count; kp.G.p.ptrs = Gp.ptr.count; kp.G.p.cols = Gp.col.count; 
                kp.G.a.rows = Ga.row.count; kp.G.a.ptrs = Ga.ptr.count; kp.G.a.cols = Ga.col.count; 
                kp.G.b.rows = Gb.row.count; kp.G.b.ptrs = Gb.ptr.count; kp.G.b.cols = Gb.col.count; 

                kp.count = mem.count.ptr;

                kernel<<<setting.block, setting.thread, 0, setting.stream[streamIndex]>>>(kp);

            }
            next();
        }
    }

    for (auto & d : dev) {
        cudaSetDevice(d.gpu.meta.index); CUDACHECK();
        for (size_t i = 0; i < d.gpu.setting.stream.size(); i++) {
            d.mem.stream[i].count.copy_d2h_async(&globalCount[d.gpu.setting.stream.size() * d.gpu.meta.index + i], d.gpu.setting.stream[i]); CUDACHECK();
        }
    }

    for (auto & d : dev) {
        cudaSetDevice(d.gpu.meta.index); CUDACHECK();
        for (size_t i = 0; i < d.gpu.setting.stream.size(); i++) {
            cudaStreamSynchronize(d.gpu.setting.stream[i]); CUDACHECK();
        }
    }

    for (size_t i = 1; i < globalCount.size(); i++) {
        globalCount.front() += globalCount[i];
    }

    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    std::cout << globalCount.front() << "," << elapsed.count() << std::endl;
}