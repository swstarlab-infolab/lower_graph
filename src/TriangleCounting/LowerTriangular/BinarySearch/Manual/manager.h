#ifndef __MANAGER_H__
#define __MANAGER_H__

#include "context.h"
#include "type.h"
#include <vector>

#include <boost/fiber/buffered_channel.hpp>

namespace Manager {
namespace MessageType {
struct CommandReq {
    struct {
        size_t row, col;
    } G[3];
};

struct CommandRes {
    struct {
        size_t row, col;
    } G[3];
    Count triangle;
    double elapsed;
    int deviceID;
    bool success;
};

struct LoadReq {
    size_t row, col;
};

struct LoadRes {
    size_t row, col;
    size_t byte;
    void * ptr;
};
} // namespace MessageType

typedef boost::fibers::buffered_channel<MessageType::CommandReq> chanCmdReq;
typedef boost::fibers::buffered_channel<MessageType::CommandRes> chanCmdRes;
typedef boost::fibers::buffered_channel<MessageType::LoadReq> chanLoadReq;
typedef boost::fibers::buffered_channel<MessageType::LoadRes> chanLoadRes;

void commander(
    Context const & ctx,
    chanCmdReq & cmdReq,
    std::vector<std::shared_ptr<chanCmdRes>> & cmdRes);


void loader(
    Context const & ctx,
    chanLoadReq & loadReq,
    std::vector<std::shared_ptr<chanLoadRes>> & loadRes);

namespace Execute {
void GPU(
    Context const & ctx,
    chanCmdReq & cmdReq,
    chanCmdRes * cmdRes,
    chanLoadReq & loadReq,
    chanLoadRes * loadRes,
    int gpuID);
} // namespace Execute
} // namespace Manager

#endif