#ifndef __GlobalContext_h__
#define __GlobalContext_h__

#include <stdint.h>
#include <GridCSR/GridCSR.h>
#include "queue.h"
#include "type.h"

struct GlobalContext {
    struct JobCommandReq {
        struct {
            size_t row, col;
        } G[3];
    };

    struct JobCommandRes {
        struct {
            size_t row, col;
        } G[3];
        Count triangle;
        double elapsed;
        bool success;
    };

    struct JobLoadDataReq {
        size_t row, col;
    };

    struct JobLoadDataRes {
        size_t row, col;
        size_t byte;
        void * ptr;
    };

    // Read only
    GridCSR::MetaData meta;
    Queue::SrSr<JobCommandReq, JobCommandRes> qCommand;
    Queue::SrMr<JobLoadDataReq, JobLoadDataRes> qLoad;

    void init(GridCSR::FS::path const & filePath);
};

#endif