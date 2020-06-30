#include "ExecutionManager.cuh"
#include "type.cuh"

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

static void Execution(Context &								ctx,
					  DeviceID								myID,
					  std::shared_ptr<bchan<Command>>		in,
					  std::shared_ptr<bchan<CommandResult>> out)
{
	using DataTxCallback = bchan<MemInfo<Vertex>>;

	size_t hitCount = 0, missCount = 0;

	for (auto & req : *in) {
		// PREPARE
		auto start = std::chrono::system_clock::now();

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
		// LAUNCH
		if (myID > -1) {
			myTriangle = launchKernelGPU(ctx, myID, memInfo);
		} else {
			myTriangle = launchKernelCPU(ctx, myID, memInfo);
		}

		/*
				printf("Kernel End:\n"
					   "(%d,%d):[%s,%s,%s]\n"
					   "(%d,%d):[%s,%s,%s]\n"
					   "(%d,%d):[%s,%s,%s]\n",
					   req.gidx[0][0],
					   req.gidx[0][1],
					   memInfo[0][0].print().c_str(),
					   memInfo[0][1].print().c_str(),
					   memInfo[0][2].print().c_str(),
					   req.gidx[1][0],
					   req.gidx[1][1],
					   memInfo[1][0].print().c_str(),
					   memInfo[1][1].print().c_str(),
					   memInfo[1][2].print().c_str(),
					   req.gidx[2][0],
					   req.gidx[2][1],
					   memInfo[2][0].print().c_str(),
					   memInfo[2][1].print().c_str(),
					   memInfo[2][2].print().c_str());
					   */

		auto end = std::chrono::system_clock::now();

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
		res.gidx		= req.gidx;
		res.deviceID	= myID;
		res.triangle	= myTriangle;
		res.elapsedTime = std::chrono::duration<double>(end - start).count();

		out->push(res);
	}

	ctx.dataManagerCtx[myID].chan->close();
	// out->close();

	printf("HIT: %ld, MISS: %ld, HIT/TOTAL: %lf\n",
		   hitCount,
		   missCount,
		   double(hitCount) / double(hitCount + missCount));
}

std::shared_ptr<bchan<CommandResult>>
ExecutionManager(Context & ctx, int myID, std::shared_ptr<bchan<Command>> in)
{
	auto out = std::make_shared<bchan<CommandResult>>(1 << 4);
	if (myID >= -1) {
		std::thread([&, myID, in, out] { Execution(ctx, myID, in, out); }).detach();
	} else {
		out->close();
	}

	return out;
}