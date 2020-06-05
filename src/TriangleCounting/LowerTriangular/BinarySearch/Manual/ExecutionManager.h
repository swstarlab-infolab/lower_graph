#ifndef E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83
#define E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83

#include "make.h"
#include "type.h"

#include <BuddySystem/BuddySystem.h>
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <unistd.h>

static auto launchKernel(std::array<MemInfo, 9> & memInfo) { return 10L; }

void ComputationCPU(Context const &						  ctx,
					int									  myID,
					std::shared_ptr<bchan<Command>>		  in,
					std::shared_ptr<bchan<CommandResult>> out)
{
}

void ComputationGPU(Context &							  ctx,
					int									  myID,
					std::shared_ptr<bchan<Command>>		  in,
					std::shared_ptr<bchan<CommandResult>> out)
{
	using DataTx		 = Tx<DataMethod, MemInfo>;
	using DataTxCallback = bchan<MemInfo>;

	for (auto & req : *in.get()) {
		auto start = std::chrono::system_clock::now();

		// PREPARE
		std::array<MemInfo, 9> memInfo;
		std::array<fiber, 9>   waitGroup;
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i * 3 + type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					DataTx tx;
					tx.method = DataMethod::Ready;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.memChan[myID].get()->push(tx);

					printf("Device%d, Key=<(%d,%d),%d>, Wait Callback: %p\n",
						   myID,
						   tx.key.idx[0],
						   tx.key.idx[1],
						   tx.key.type,
						   tx.cb.get());

					for (auto & cbres : *callback.get()) {
						memInfo[i * 3 + type] = cbres;
					}
					// callback.get()->pop(memInfo[i * 3 + type]);
					printf("Device%d, Key=<(%d,%d),%d>, Got Callback\n",
						   myID,
						   tx.key.idx[0],
						   tx.key.idx[1],
						   tx.key.type);
				});
			}
		}

		for (auto & w : waitGroup) {
			if (w.joinable()) {
				w.join();
			}
		}

		// LAUNCH
		launchKernel(memInfo);

		auto end = std::chrono::system_clock::now();

		// RELEASE MEMORY
		for (uint32_t i = 0; i < 3; i++) {
			for (uint32_t type = 0; type < 3; type++) {
				waitGroup[i * 3 + type] = fiber([&, myID, i, type] {
					auto callback = std::make_shared<DataTxCallback>(2);

					DataTx tx;
					tx.method = DataMethod::Done;
					tx.key	  = {req.gidx[i], (DataType)(type)};
					tx.cb	  = callback;

					ctx.memChan[myID].get()->push(tx);

					callback.get()->pop(memInfo[i * 3 + type]);
				});
			}
		}

		for (auto & w : waitGroup) {
			if (w.joinable()) {
				w.join();
			}
		}

		// CALLBACK RESPONSE
		CommandResult res;
		res.gidxs		= req.gidx;
		res.deviceID	= myID;
		res.triangles	= 0;
		res.elapsedTime = std::chrono::duration<double>(end - start).count();

		out.get()->push(res);
	}

	ctx.memChan[myID].get()->close();
	out.get()->close();
}

auto Computation(Context & ctx, int myID, std::shared_ptr<bchan<Command>> in)
{
	// auto out = make<bchan<CommandResult>>(1 << 4);
	auto out = std::make_shared<bchan<CommandResult>>(1 << 4);
	// prepare channels
	if (myID > 0) {
		std::thread([&, myID, in, out] { ComputationGPU(ctx, myID, in, out); }).detach();
	} else {
		std::thread([&, myID, in, out] { ComputationCPU(ctx, myID, in, out); }).detach();
	}

	return out;
}

/*
void GPU(
	Context const& ctx,
	ChanCmdReq& cmdReq,
	ChanCmdRes* cmdRes,
	ChanLoadReq& loadReq,
	ChanLoadRes* loadRes,
	ChanLoadComp& loadComp,
	int gpuID)
{
	fprintf(stdout, "[GPU%d  ] START\n", gpuID);

	auto map = makeHashMap<DataType::GridIndex, DataType::CacheValue>(
		1024,
		[](DataType::GridIndex const& key) {
			using KeyType =
std::remove_const<std::remove_reference<decltype(key.front())>::type>::type;
			return std::hash<KeyType>()(key[0]) ^ (std::hash<KeyType>()(key[1])
<< 1);
		});

	auto searchInMyMap = [&map](
							 std::array<DataType::GridIndex, 3> const& grid,
							 std::array<DataType::Memory, 3>& location) {
		std::array<decltype(map.end()), 3> tmp;

		for (unsigned int i = 0; i < grid.size(); i++) {
			decltype(map)::accessor a;
			if (!map.find(a, grid[i])) {
				return false;
			}
			location[i] = a->second.arr[i];
		}

		return true;
	};

	cudaSetDevice(gpuID);
	DataType::MemoryShared myMemory;
	myMemory.second = 1024L * 1024L * 1024L;
	myMemory.first = allocCUDA<char>(myMemory.second);

	portable_buddy_system buddy;
	buddy.init(memrgn_t { (void*)myMemory.first.get(), myMemory.second }, 256,
1);

	fprintf(stdout, "[GPU%d  ] Init Complete\n", gpuID);

	for (auto& cReq : cmdReq) {
		MessageType::CommandRes cRes;
		cRes.deviceID = gpuID;

		auto timeStart = std::chrono::system_clock::now();

		// check my memory
		std::array<DataType::Memory, 3> location;

		if (searchInMyMap(cReq.G, location)) {
			fprintf(stdout, "[GPU%d  ] Find Memory:
(%ld,%ld)(%ld,%ld)(%ld,%ld)\n", gpuID, cReq.G[0][0], cReq.G[0][0], cReq.G[1][0],
cReq.G[1][0], cReq.G[2][0], cReq.G[2][0]); } else { fprintf(stdout, "[GPU%d  ]
Not Find Memory: (%ld,%ld)(%ld,%ld)(%ld,%ld), Request to Load Manager\n", gpuID,
				cReq.G[0][0], cReq.G[0][0],
				cReq.G[1][0], cReq.G[1][0],
				cReq.G[2][0], cReq.G[2][0]);

			for (unsigned int g = 0; g < cReq.G.size(); g++) {
				MessageType::LoadReq lReq;
				lReq.idx = cReq.G[g];
				lReq.deviceID = gpuID;

				loadReq.push(lReq);
			}

			fprintf(stdout, "[GPU%d  ] Not Find Memory:
(%ld,%ld)(%ld,%ld)(%ld,%ld), Waiting Load Manager Response\n", gpuID,
				cReq.G[0][0], cReq.G[0][0],
				cReq.G[1][0], cReq.G[1][0],
				cReq.G[2][0], cReq.G[2][0]);

			// wait...
			for (unsigned int g = 0; g < cReq.G.size(); g++) {
				MessageType::LoadRes lRes;

				loadRes[gpuID].pop(lRes);

				// malloc and copy
				std::array<void*, 3> ptr;
				decltype(map)::accessor a;
				for (unsigned int i = 0; i < ptr.size(); i++) {
					map.insert(a, lRes.idx);
					a->second.arr[i].second = lRes.arr[i].second;
					a->second.arr[i].first = buddy.allocate(lRes.arr[i].second);
					cudaMemcpy(a->second.arr[i].first, lRes.arr[i].first,
lRes.arr[i].second, cudaMemcpyDeviceToHost);

					// if full, evict
					if (full) {
						// select evictable memory
						buddy.deallocate()
					}
				}
				a.release();

				// notify memcpy completion!
				MessageType::LoadComp lComp;

				lComp.idx = lRes.idx;
				loadComp.push(lComp);
			}

			fprintf(stdout, "[GPU%d  ] Loaded: (%ld,%ld)(%ld,%ld)(%ld,%ld)\n",
gpuID, cReq.G[0][0], cReq.G[0][0], cReq.G[1][0], cReq.G[1][0], cReq.G[2][0],
cReq.G[2][0]);
		}

		// okay, memory is ready... kernel launch
		cRes.triangle = launchKernel();

		auto timeEnd = std::chrono::system_clock::now();

		std::chrono::duration<double> timeSecond = timeEnd - timeStart;

		for (int g = 0; g < 3; g++) {
			cRes.G[g] = cReq.G[g];
		}

		cRes.elapsed = timeSecond.count();
		cRes.success = true;

		cmdRes->push(cRes);
	}

	loadReq.close();
	cmdRes->close();

	//cudaDeviceReset();
}
					*/

#endif /* E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83 */
