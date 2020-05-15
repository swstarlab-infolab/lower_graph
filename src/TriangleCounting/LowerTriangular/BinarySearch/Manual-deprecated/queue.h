#ifndef __queue_h__
#define __queue_h__

#include <concurrentqueue/blockingconcurrentqueue.h>
#include <vector>

struct Queue {
    template <typename Type>
    struct Elem {
        Type data;
        bool end;
    };

    // Single Request Single Response
    template <typename Request, typename Response>
    struct SrSr {
        moodycamel::BlockingConcurrentQueue<Elem<Request>> req;
        moodycamel::BlockingConcurrentQueue<Elem<Response>> res;
    };

    // Single Request Multiple Response
    template <typename Request, typename Response>
    struct SrMr {
        moodycamel::BlockingConcurrentQueue<Elem<Request>> req;
        std::vector<moodycamel::BlockingConcurrentQueue<Elem<Response>>> res;
    };
};

#endif