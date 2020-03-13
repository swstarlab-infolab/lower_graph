#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <array>

#include "../common.h"
#include "tc.h"
#include "device-setting.cuh"

#include <cub/device/device_scan.cuh>
#include <tbb/parallel_for_each.h>

__global__ void gen_lookup_temp(
    vertex_t const * row,
    vertex_t const * ptr,
    uint32_t const rowSize,
    lookup_t * lookup_temp)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rowSize; i += gridDim.x * blockDim.x) {
        lookup_temp[row[i]] = ptr[i+1] - ptr[i];
    }
}

__global__ void reset_lookup_temp(
    vertex_t const * row,
    uint32_t const rowSize,
    lookup_t * lookup_temp)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rowSize; i += gridDim.x * blockDim.x) {
        lookup_temp[row[i]] = 0;
    }
}

__device__ static void bitmap_set(
    bitmap_t* bm0,
    bitmap_t* bm1,
    const vertex_t vid)
{
	atomicOr(&bm0[vid >> EXP_BITMAP0], 1 << ((vid >> (EXP_BITMAP0-EXP_BITMAP1)) & 31));
	atomicOr(&bm1[vid >> EXP_BITMAP1], 1 << (vid & 31));
}

__device__ static bool bitmap_check(bitmap_t const * bm0, bitmap_t const * bm1, const vertex_t vid) {
    if (bm0[vid >> EXP_BITMAP0] & (1 << ((vid >> (EXP_BITMAP0 - EXP_BITMAP1) & 31)))) {
        return bm1[vid >> EXP_BITMAP1] & (1 << (vid & 31));
    } else {
        return false;
    }
}

__global__ static void kernel(
    lookup_t const * lookupG0, vertex_t const * G0Col,
    vertex_t const * G1Row, vertex_t const * G1Ptr, vertex_t const * G1Col,
    lookup_t const * lookupG2, vertex_t const * G2Col,
    count_t const G1RowSize,
    count_t const gridWidth,
    bitmap_t * bitmapLV0, bitmap_t * bitmapLV1,
    count_t * count)
{
    bitmap_t * mybm0 = &bitmapLV0[(gridWidth >> EXP_BITMAP0) * blockIdx.x];
    bitmap_t * mybm1 = &bitmapLV1[(gridWidth >> EXP_BITMAP1) * blockIdx.x];
    count_t mycount = 0;

    //__shared__ int SHARED[1024];
    __shared__ extern int SHARED[];

    for (uint32_t g1row_iter = blockIdx.x; g1row_iter < G1RowSize; g1row_iter += gridDim.x) {

        // This makes huge difference!!!
        // Without "Existing Row" information: loop all 2^24 and check it all
        // With "Existing Row" information: extremely faster than without-version
        auto const g1row = G1Row[g1row_iter];

        if (lookupG2[g1row] == lookupG2[g1row + 1]) { continue; }

        auto const g1col_idx_s = G1Ptr[g1row_iter], g1col_idx_e = G1Ptr[g1row_iter+1];

        // generate bitmap
        for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e; g1col_idx += blockDim.x) {
            bitmap_set(mybm0, mybm1, G1Col[g1col_idx]);
        }

        auto const g2col_s = lookupG2[g1row], g2col_e = lookupG2[g1row+1];

        for (uint32_t g2col_idx = g2col_s; g2col_idx < g2col_e; g2col_idx += blockDim.x) {
            SHARED[threadIdx.x] = (g2col_idx + threadIdx.x < g2col_e) ? (int)G2Col[g2col_idx + threadIdx.x] : -1;

            __syncthreads();

            for (uint32_t s = 0; s < blockDim.x; s++) {
                int const g2col = SHARED[s];
                if (g2col == -1) { break; }
                if (lookupG0[g2col] == lookupG0[g2col + 1]) { continue; }

                auto const g0col_idx_s = lookupG0[g2col], g0col_idx_e = lookupG0[g2col+1];

                for (uint32_t g0col_idx = g0col_idx_s + threadIdx.x; g0col_idx < g0col_idx_e; g0col_idx += blockDim.x) {
                    if (bitmap_check(mybm0, mybm1, G0Col[g0col_idx])) {
                        mycount++;
                    }
                }
            }
            __syncthreads();
        }

        // reset bitmap
        for (uint32_t g1col_idx = g1col_idx_s + threadIdx.x; g1col_idx < g1col_idx_e; g1col_idx += blockDim.x) {
            auto const c = G1Col[g1col_idx];
            mybm0[c >> EXP_BITMAP0] = 0;
            mybm1[c >> EXP_BITMAP1] = 0;
        }

        __syncthreads();
    }

    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
		mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
	}

	if ((threadIdx.x & 31) == 0) { atomicAdd(count, mycount); }
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
        for (size_t col = 0; col <= row; col++) {
            for (size_t i = col; i <= row; i++) {
                auto & d = dev[deviceIndex];


                auto const & G0 = d.mem.graph[i][col];
                auto const & G1 = d.mem.graph[row][col];
                auto const & G2 = d.mem.graph[row][i];

                auto & mem = d.mem.stream[streamIndex];

                auto & setting = d.gpu.setting;

                cudaSetDevice(d.gpu.meta.index); CUDACHECK();

                if (!(G0.row.count && G1.row.count && G2.row.count)) { continue; }

                gen_lookup_temp <<<setting.block, setting.thread, 0, setting.stream[streamIndex]>>> (
                    G0.row.ptr,
                    G0.ptr.ptr, 
                    G0.row.count,
                    mem.lookup.temp.ptr
                );

                cub::DeviceScan::ExclusiveSum(
                    mem.cub.ptr,
                    mem.cub.byte,
                    mem.lookup.temp.ptr,
                    mem.lookup.G0.ptr,
                    mem.lookup.G0.count,
                    setting.stream[streamIndex]);

                reset_lookup_temp <<<setting.block, setting.thread, 0, setting.stream[streamIndex]>>> (
                    G0.row.ptr, 
                    G0.row.count,
                    mem.lookup.temp.ptr
                );

                gen_lookup_temp <<<setting.block, setting.thread, 0, setting.stream[streamIndex]>>> (
                    G2.row.ptr, 
                    G2.ptr.ptr, 
                    G2.row.count,
                    mem.lookup.temp.ptr
                );

                cub::DeviceScan::ExclusiveSum(
                    mem.cub.ptr,
                    mem.cub.byte,
                    mem.lookup.temp.ptr,
                    mem.lookup.G2.ptr,
                    mem.lookup.G2.count,
                    setting.stream[streamIndex]);

                reset_lookup_temp <<<setting.block, setting.thread, 0, setting.stream[streamIndex]>>> (
                    G2.row.ptr,
                    G2.row.count,
                    mem.lookup.temp.ptr
                );

                kernel <<<setting.block, setting.thread, setting.thread, setting.stream[streamIndex]>>> (
                    mem.lookup.G0.ptr, G0.col.ptr,
                    G1.row.ptr, G1.ptr.ptr, G1.col.ptr,
                    mem.lookup.G2.ptr, G2.col.ptr,
                    G1.row.count,
                    gridWidth,
                    mem.bitmap.lv0.ptr, mem.bitmap.lv1.ptr,
                    mem.count.ptr
                );

                next();
            }
        }
    }

    for (auto & d : dev) {
        cudaSetDevice(d.gpu.meta.index); CUDACHECK();
        for (size_t i = 0; i < d.gpu.setting.stream.size(); i++) {
            d.mem.stream[i].count.copy_d2h_async(&globalCount[dev.size() * d.gpu.meta.index + i], d.gpu.setting.stream[i]); CUDACHECK();
        }
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