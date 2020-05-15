#include "GlobalContext.h"
#include "error.h"

#include <stdio.h>

void GlobalContext::init(GridCSR::FS::path const & filePath);
    fprintf(stdout, "[GLOB ] Try load metadata\n");
    this->meta.Load(filePath / "meta.json");

    fprintf(stdout, "[GLOB ] Try set queue\n");
    int devCount = -1;
    ThrowCuda(cudaGetDeviceCount(&devCount));
    this->qLoad.res.resize(devCount);

    fprintf(stdout, "[GLOB ] init success\n");
}