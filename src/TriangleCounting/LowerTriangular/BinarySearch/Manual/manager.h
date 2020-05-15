#pragma once

#include "context.h"
#include "channel.h"
#include "type.h"

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

void commander(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes);


template <size_t Size>
void loader(
    Context const & ctx,
    Channel<MessageType::LoadReq> & loadReq,
    std::array<Channel<MessageType::LoadRes>, Size> loadRes);

namespace Execute {
void CPU(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes,
    Channel<MessageType::LoadReq> & loadReq,
    Channel<MessageType::LoadRes> & loadRes);
void GPU(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes,
    Channel<MessageType::LoadReq> & loadReq,
    Channel<MessageType::LoadRes> & loadRes);
} // namespace Execute
} // namespace Manager