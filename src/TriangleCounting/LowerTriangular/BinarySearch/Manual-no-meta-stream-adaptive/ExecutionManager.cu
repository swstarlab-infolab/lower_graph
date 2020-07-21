#include "ExecutionManager.cuh"
#include "type.cuh"

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

static void Execution(Context &								ctx,
					  DeviceID								myID,
					  size_t								myStreamID,
					  std::shared_ptr<bchan<Command>>		in,
					  std::shared_ptr<bchan<CommandResult>> out)
{
	// printf("myDeviceID: %d, myStreamID: %ld\n", myID, myStreamID);
	using DataTxCallback = bchan<MemInfo<Vertex>>;

	size_t hitCount = 0, missCount = 0;

	for (auto & req : *in) {
		Grids								memInfo;
		std::array<std::array<fiber, 3>, 3> waitGroup;
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i][type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					Tx tx;
					tx.method = Method::Ready;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.dataManagerCtx[myID].chan->push(tx);

					for (auto & cbres : *callback) {
						memInfo[i][type] = cbres;
					}
				});
			}
		}

		// Must wait all memory info
		for (auto & row : waitGroup) {
			for (auto & w : row) {
				if (w.joinable()) {
					w.join();
				}
			}
		}

		for (auto & row : memInfo) {
			for (auto & i : row) {
				if (i.hit) {
					hitCount++;
				} else {
					missCount++;
				}
			}
		}

		Count myTriangle = 0;

		// PREPARE
		[&] {
			auto minBlock  = ctx.setting[1];
			auto maxThread = ctx.setting[2];

			for (size_t block = minBlock; block < minBlock * 32; block <<= 1) {
				size_t thread = maxThread;
				// for (size_t thread = maxThread / 32; thread < maxThread; thread <<= 1) {

				auto start = std::chrono::system_clock::now();
				// LAUNCH
				if (myID > -1) {
					myTriangle = launchKernelGPU(ctx, myID, myStreamID, memInfo, block, thread);
				}

				auto end  = std::chrono::system_clock::now();
				auto dura = std::chrono::duration<double>(end - start).count();

				printf("%d,%d,%d,%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%llu,%.6lf\n",
					   req.gidx[0][0],
					   req.gidx[0][1],
					   req.gidx[1][0],
					   req.gidx[1][1],
					   req.gidx[2][0],
					   req.gidx[2][1],
					   memInfo[0][0].byte,
					   memInfo[0][2].byte,
					   memInfo[1][0].byte,
					   memInfo[1][2].byte,
					   memInfo[2][0].byte,
					   memInfo[2][2].byte,
					   block,
					   thread,
					   myTriangle,
					   dura);
				//}
			}
		}();

		// RELEASE MEMORY
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i][type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					Tx tx;
					tx.method = Method::Done;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.dataManagerCtx[myID].chan->push(tx);

					for (auto & cbres : *callback) {
						memInfo[i][type] = cbres;
					}
				});
			}
		}

		for (auto & row : waitGroup) {
			for (auto & w : row) {
				if (w.joinable()) {
					w.join();
				}
			}
		}

		// CALLBACK RESPONSE
		CommandResult res;
		res.gidx	 = req.gidx;
		res.deviceID = myID;
		res.streamID = myStreamID;
		res.triangle = myTriangle;

		out->push(res);
	}
}

std::shared_ptr<bchan<CommandResult>>
ExecutionManager(Context & ctx, int myID, std::shared_ptr<bchan<Command>> in)
{
	auto out = std::make_shared<bchan<CommandResult>>(1 << 4);
	if (myID >= -1) {
		std::thread([=, &ctx] {
			std::vector<std::thread> ts(ctx.setting[0]);

			for (size_t streamID = 0; streamID < ctx.setting[0]; streamID++) {
				ts[streamID] = std::thread([=, &ctx] { Execution(ctx, myID, streamID, in, out); });
			}

			for (size_t streamID = 0; streamID < ctx.setting[0]; streamID++) {
				if (ts[streamID].joinable()) {
					ts[streamID].join();
				}
			}

			ctx.dataManagerCtx[myID].chan->close();
			out->close();
		}).detach();
	} else {
		out->close();
	}

	return out;
}