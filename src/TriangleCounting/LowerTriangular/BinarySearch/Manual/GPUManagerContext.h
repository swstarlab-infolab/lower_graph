#ifndef __GPUManagerContext_h__
#define __GPUManagerContext_h__

#include "GPUMemory.cuh"
#include "GPUMemoryBuddy.cuh"
#include "type.h"
#include <BuddySystem/BuddySystem.h>

struct GPUManagerContext {
    int gpuID;
    struct {
        struct {
            GPUMemory<Lookup> G0, G2, Gt;
        } lookup;

        GPUMemory<Lookup> cub;
        GPUMemory<Count> count;
    } mem;

    GPUMemory<void> buffer;
    hashed_buddy_system buddy;

    void init();
};

#endif