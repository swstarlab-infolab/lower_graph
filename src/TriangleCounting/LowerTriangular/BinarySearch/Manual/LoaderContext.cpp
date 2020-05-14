#include "LoaderContext.h"

void LoaderContext::init() {
    auto memSize = 1024L * 1024L * 1024L * 128L;

    fprintf(stdout, "[CPU  ] Try allocate buffer memory: %s bytes\n", SIUnit(memSize).c_str());
    this->buffer.mallocByte(memSize);
    this->buddy.init(memrgn_t{ cpuCtx.buffer.data(), cpuCtx.buffer.byte() }, buddyAlign, buddyMinCof);
    fprintf(stdout, "[CPU  ] init success\n");
}