#include "../manager.h"
#include "../context.h"
#include "../channel.h"

namespace Manager {

void commander(
    Context const & ctx,
    Channel<MessageType::CommandReq> & cmdReq,
    Channel<MessageType::CommandRes> & cmdRes)
{
    auto const MAXROW = ctx.meta.info.count.row;
    //auto const MAXJOB = ((MAXROW) * (MAXROW + 1) * (MAXROW + 2)) / 6;

    for (size_t row = 0; row < MAXROW; row++) {
        for (size_t col = 0; col <= row; col++) {
            for (size_t i = col; i <= row; i++) {
                typename MessageType::CommandReq req;

                req.G[0] = {i, col};
                req.G[1] = {row, col};
                req.G[2] = {row, i};

                cmdReq << req;
            }
        }
    }

    Close(cmdReq);

    MessageType::CommandRes res;
    ForChan(res, cmdRes, [&]{
        if (res.success) {
            printf("Success (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                res.G[0].row,
                res.G[0].col,
                res.G[1].row,
                res.G[1].col,
                res.G[2].row,
                res.G[2].col,
                res.triangle,
                res.elapsed);
        } else {
            printf("Failed (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                res.G[0].row,
                res.G[0].col,
                res.G[1].row,
                res.G[1].col,
                res.G[2].row,
                res.G[2].col,
                res.triangle,
                res.elapsed);
        }
    });
}
} // namespace Manager