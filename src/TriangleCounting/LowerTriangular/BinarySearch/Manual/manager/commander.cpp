#include "../manager.h"
#include "../context.h"
#include <vector>
#include <memory>
#include <boost/fiber/all.hpp>

namespace Manager {

void commander(
    Context const & ctx,
    chanCmdReq & cmdReq,
    std::vector<std::shared_ptr<chanCmdRes>> & cmdRes)
{
    auto const MAXROW = ctx.meta.info.count.row;
    //auto const MAXJOB = ((MAXROW) * (MAXROW + 1) * (MAXROW + 2)) / 6;

    auto requester = boost::fibers::fiber([&MAXROW, &cmdReq]{
        for (size_t row = 0; row < MAXROW; row++) {
            for (size_t col = 0; col <= row; col++) {
                for (size_t i = col; i <= row; i++) {
                    MessageType::CommandReq req;

                    req.G[0] = {i, col};
                    req.G[1] = {row, col};
                    req.G[2] = {row, i};
                    cmdReq.push(req);
                }
            }
        }

        cmdReq.close();
    });

    auto responser = boost::fibers::fiber([&cmdRes]{
        for (auto & c : cmdRes) {
            for (auto & res : *c.get()) {
                if (res.success) {
                    fprintf(stdout, "SUCCESS: DEV%d (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                        res.deviceID,
                        res.G[0].row, res.G[0].col,
                        res.G[1].row, res.G[1].col,
                        res.G[2].row, res.G[2].col,
                        res.triangle,
                        res.elapsed);
                } else {
                    fprintf(stderr, "FAILED:  DEV%d (%ld,%ld)(%ld,%ld)(%ld,%ld)=%lld,%lf(sec)\n",
                        res.deviceID,
                        res.G[0].row, res.G[0].col,
                        res.G[1].row, res.G[1].col,
                        res.G[2].row, res.G[2].col,
                        res.triangle,
                        res.elapsed);
                }
            }
        }
    });

    requester.join();
    responser.join();
}

} // namespace Manager