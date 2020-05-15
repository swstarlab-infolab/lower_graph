#include "../manager.h"
#include "../context.h"
#include "../channel.h"

namespace Manager {
namespace Execute {
void GPU(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes,
    Channel<MessageType::LoadReq> & loadReq,
    Channel<MessageType::LoadRes> & loadRes)
{
    typename MessageType::CommandReq req;
    typename MessageType::CommandRes res;

    ForChan(req, cmdReq, [&]{
        res.G[0].row = req.G[0].row;
        res.G[0].col = req.G[0].col;
        res.G[1].row = req.G[1].row;
        res.G[1].col = req.G[1].col;
        res.G[2].row = req.G[2].row;
        res.G[2].col = req.G[2].col;
        res.success = true;

        cmdRes << res;
    });

    Close(cmdRes);
}

void CPU(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes,
    Channel<MessageType::LoadReq> & loadReq,
    Channel<MessageType::LoadRes> & loadRes)
{
    // something cool
}

}
}