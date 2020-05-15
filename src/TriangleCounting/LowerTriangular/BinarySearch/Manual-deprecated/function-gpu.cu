#include <cub/device/device_scan.cuh>

void mainGPU(GlobalContext & globalCtx, GPUContext & gpuCtx) {
/*
    auto tryAllocGrid = [&](Grid::Mem & mem){
        mem.ptr = (GridCSR::Vertex*)gpuCtx.buddy.allocate(mem.byte);
        mem.byte = mem.byte;
        return (mem.ptr != nullptr) ? true : false;
    };

    Grid tempGrid;

    if ([&]{
        if (!tryAllocGrid(tempGrid.row)) { return false; }
        if (!tryAllocGrid(tempGrid.ptr)) { return false; }
        if (!tryAllocGrid(tempGrid.col)) { return false; }
        return true;
    }()) {
        // success
        d.memTable = tempGrid;
        cudaMemcpy(d.grid.row.ptr, hostGrid.row.ptr, devGrid.row.byte, cudaMemcpyHostToDevice);
        cudaMemcpy(devGrid.ptr.ptr, hostGrid.ptr.ptr, devGrid.ptr.byte, cudaMemcpyHostToDevice);
        cudaMemcpy(devGrid.col.ptr, hostGrid.col.ptr, devGrid.col.byte, cudaMemcpyHostToDevice);
    } else {
        // failed. should remove one grid.
    }

    // kernel launch

    globalCtx.metaData

    */
}