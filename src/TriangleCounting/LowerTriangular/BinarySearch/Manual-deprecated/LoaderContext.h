#ifndef __LoaderContext_h__
#define __LoaderContext_h__

#include "CPUMemory.h"
#include <BuddySystem/BuddySystem.h>

struct CPUContext {
    CPUMemory<void> buffer;
    buddy_system buddy;
    //MemTable memTable;
};

#endif