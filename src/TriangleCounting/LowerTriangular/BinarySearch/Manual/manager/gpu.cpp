#include "../manager.h"
#include "../context.h"
#include <unistd.h>
//#include <cuda_runtime.h>

namespace Manager {
namespace Execute {
void GPU(
    Context const & ctx,
    chanCmdReq & cmdReq,
    chanCmdRes* cmdRes,
    chanLoadReq & loadReq,
    chanLoadRes* loadRes,
    int gpuID)
{
    //cudaSetDevice(gpuID);
    int * temp;

    for (auto & req : cmdReq) {
        auto timeStart = std::chrono::system_clock::now();
        //cudaMalloc(&temp, 1024L * 1024L * 1024L * sizeof(*temp));
        //cudaFree(temp);
        auto timeEnd = std::chrono::system_clock::now();

        std::chrono::duration<double> timeSecond = timeEnd - timeStart;

        MessageType::CommandRes res;

        res.G[0].row = req.G[0].row; res.G[0].col = req.G[0].col;
        res.G[1].row = req.G[1].row; res.G[1].col = req.G[1].col;
        res.G[2].row = req.G[2].row; res.G[2].col = req.G[2].col;

        res.elapsed = timeSecond.count();
        res.triangle = 0;
        res.deviceID = gpuID;
        res.success = true;

        cmdRes->push(res);
    }

    cmdRes->close();

    //cudaDeviceReset();
}

} // namespace Execute
} // namespace Manager