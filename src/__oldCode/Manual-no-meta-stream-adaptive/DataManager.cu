#include "DataManager.cuh"
#include "make.cuh"

#include <BuddySystem/BuddySystem.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

static auto genPath(Context & ctx, Key const & k)
{
	auto baseString =
		std::string(ctx.folderPath) + std::to_string(k.idx[0]) + "-" + std::to_string(k.idx[1]);

	fs::path finalPath;
	switch (k.type) {
	case DataType::Row:
		finalPath = fs::path(baseString + ".row");
		break;
	case DataType::Ptr:
		finalPath = fs::path(baseString + ".ptr");
		break;
	case DataType::Col:
		finalPath = fs::path(baseString + ".col");
		break;
	}

	return finalPath;
}

static auto methodDone(Context & ctx, DeviceID myID)
{
	auto in = std::make_shared<bchan<Tx>>(16);
	std::thread([&, myID, in] {
		auto & myCtx = ctx.dataManagerCtx[myID];
		for (auto & tx : *in) {
			MemInfo<Vertex> myInfo = {
				0,
			};
			{
				std::lock_guard<std::mutex> lg(*myCtx.cacheMtx);
				myCtx.cache->at(tx.key).refCnt -= 1;
				// printf("[%2d] %s Done %d -> %d\n", myID, tx.key.print().c_str(),
				// myCtx.cache->at(tx.key).refCnt + 1, myCtx.cache->at(tx.key).refCnt);
			}
			myInfo.ok  = true;
			myInfo.hit = true;
			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();

	return in;
}

static void tryAllocate(Context &					   ctx,
						Key &						   key,
						DeviceID					   myID,
						MemInfo<Vertex> &			   myInfo,
						std::unique_lock<std::mutex> & ul,
						bool &						   iHaveLock)
{
	auto & myCtx = ctx.dataManagerCtx[myID];

	while (true) {
		myInfo.ptr = (Vertex *)myCtx.buddy->allocate(myInfo.byte);

		if (myInfo.ptr != nullptr) {
			myCtx.cache->insert({key, {myInfo, 1}});
			// printf("[%2d] %s Allc %d -> %d\n", myID, key.print().c_str(),
			// myCtx.cache->at(key).refCnt - 1, myCtx.cache->at(key).refCnt);

			myInfo.ok  = true;
			myInfo.hit = false;

			return;
		} else {
			// allocation failure
			if (myCtx.cache->size() > 0) {
				bool evictSuccess = false;
				while (!evictSuccess) {
					if (!iHaveLock) {
						ul.lock();
						iHaveLock = true;
					}

					for (auto it = myCtx.cache->begin(); it != myCtx.cache->end();) {
						if (it->second.refCnt == 0) {
							myCtx.buddy->deallocate(it->second.info.ptr);
							it			 = myCtx.cache->erase(it);
							evictSuccess = true;
							// printf("[%2d] %s Evict %s\n", myID, key.print().c_str(),
							// it->first.print().c_str());
							break;
						} else {
							++it;
						}
					}

					if (iHaveLock) {
						ul.unlock();
						iHaveLock = false;
					}
				}
			} else {
				throw "Strange Error";
			}
		}
	}
}

static MemInfo<Vertex> requestToReady(Context & ctx, Key & key, DeviceID targetID)
{
	Tx tx;

	tx.key	  = key;
	tx.method = Method::Ready;
	tx.cb	  = std::make_shared<bchan<MemInfo<Vertex>>>(2);

	ctx.dataManagerCtx[targetID].chan->push(tx);

	MemInfo<Vertex> otherInfo;
	for (auto & info : *tx.cb) { // Code hangs on this line
		otherInfo = info;
	}

	return otherInfo;
}

static auto methodReady(Context & ctx, DeviceID myID)
{
	auto in = std::make_shared<bchan<Tx>>(16);
	std::thread([&, myID, in] {
		for (auto & tx : *in) {
			auto & myCtx = ctx.dataManagerCtx[myID];

			MemInfo<Vertex> myInfo = {
				0,
			};

			std::unique_lock<std::mutex> ul(*myCtx.cacheMtx);

			bool iHaveLock = true;

			if (myCtx.cache->find(tx.key) != myCtx.cache->end()) {
				// printf("[%2d] %s Hit!\n", myID, tx.key.print().c_str());
				myInfo = myCtx.cache->at(tx.key).info;
				myCtx.cache->at(tx.key).refCnt += 1;
				// printf("[%2d] %s Hit  %d -> %d\n", myID, tx.key.print().c_str(),
				// myCtx.cache->at(tx.key).refCnt - 1, myCtx.cache->at(tx.key).refCnt);

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				myInfo.hit = true;
			} else {
				// printf("[%2d] %s Miss!\n", myID, tx.key.print().c_str());

				myInfo.byte = fs::file_size(genPath(ctx, tx.key));

				tryAllocate(ctx, tx.key, myID, myInfo, ul, iHaveLock);

				// auto	targetID = asktoNeighbor(ctx, tx.key, myID, myInfo);
				DeviceID		targetID; // for debugging
				MemInfo<Vertex> otherInfo;
				// if (targetID != std::numeric_limits<DeviceID>::min()) {
				if (false) { // for debugging
					targetID  = -10;
					otherInfo = requestToReady(ctx, tx.key, targetID);
				} else {
					targetID = myCtx.conn->upstream;
					assert(myID != myCtx.conn->upstream);
					otherInfo = requestToReady(ctx, tx.key, targetID);
				}

				assert(myID >= -1);
				if (myID == -1) {
					// printf("start to read!\n");

					// CPU
					// std::ifstream f(otherInfo.path, std::ios::binary);
					// printf("[%2d] %s fread       SSD[%s]->Host[%p], %ld bytes)\n", myID,
					// tx.key.print().c_str(), otherInfo.path.c_str(), myInfo.ptr, otherInfo.byte);
					auto fp = open64(otherInfo.path.c_str(), O_RDONLY);

					constexpr uint64_t cDef		 = (1L << 30); // chunk Default
					uint64_t		   chunkByte = (myInfo.byte < cDef) ? myInfo.byte : cDef;
					uint64_t		   bytePos	 = 0;
					while (bytePos < myInfo.byte) {
						chunkByte =
							(myInfo.byte - bytePos > chunkByte) ? chunkByte : myInfo.byte - bytePos;
						auto loaded = read(fp, &(((uint8_t *)myInfo.ptr)[bytePos]), chunkByte);
						bytePos += loaded;
					}

					close(fp);
				} else {
					// GPU
					// printf("[%2d] %s cudaMemcpy Host[%p]-> GPU[%p], %ld bytes)\n", myID,
					// tx.key.print().c_str(), otherInfo.ptr, myInfo.ptr, otherInfo.byte);
					cudaSetDevice(myID);
					cudaMemcpyAsync(myInfo.ptr,
									otherInfo.ptr,
									otherInfo.byte,
									cudaMemcpyHostToDevice,
									myCtx.stream);
					cudaStreamSynchronize(myCtx.stream);
				}

				// printf("[%2d] %s Memcpy/Read complete\n", myID, tx.key.print().c_str());

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				// Done
				Tx compTx;

				compTx.key	  = tx.key;
				compTx.method = Method::Done;
				compTx.cb	  = std::make_shared<bchan<MemInfo<Vertex>>>(2);

				ctx.dataManagerCtx[targetID].chan->push(compTx);

				for (auto & res : *compTx.cb) {
				}

				myInfo.hit = false;
			}

			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();
	return in;
}

void DataManager(Context & ctx, DeviceID myID)
{
	if (myID < -1) {
		// Storage
		std::thread([&, myID] {
			for (auto & tx : *ctx.dataManagerCtx[myID].chan) {
				switch (tx.method) {
				case Method::Ready:
					fiber([&, myID, tx] {
						MemInfo<Vertex> myInfo = {
							0,
						};
						myInfo.ptr	= nullptr;
						myInfo.path = genPath(ctx, tx.key);
						myInfo.byte = fs::file_size(myInfo.path);
						myInfo.ok	= true;
						myInfo.hit	= true;

						tx.cb->push(myInfo);
						tx.cb->close();
					}).detach();
					break;
				case Method::Done:
					fiber([&, myID, tx] {
						MemInfo<Vertex> myInfo = {
							0,
						};

						myInfo.ok = true;

						tx.cb->push(myInfo);
						tx.cb->close();
					}).detach();
					break;
				}
			}
		}).detach();
	} else {
		// Main/GPU Memory
		std::thread([&, myID] {
			auto ReadyChan = methodReady(ctx, myID);
			auto DoneChan  = methodDone(ctx, myID);

			for (auto & tx : *ctx.dataManagerCtx[myID].chan) {
				switch (tx.method) {
				case Method::Ready:
					ReadyChan->push(tx);
					break;
				case Method::Done:
					DoneChan->push(tx);
					break;
				}
			}

			ReadyChan->close();
			DoneChan->close();
		}).detach();
	}
}