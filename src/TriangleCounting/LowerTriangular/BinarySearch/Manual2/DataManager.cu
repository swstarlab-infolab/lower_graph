#include "DataManager.cuh"
#include "make.cuh"

#include <BuddySystem/BuddySystem.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

static auto genPath(Context & ctx, Key const & k)
{
	auto baseString = std::string(ctx.folderPath) + std::to_string(k.idx[0]) + "-" +
					  std::to_string(k.idx[1]) + ".";

	fs::path finalPath;
	switch (k.type) {
	case DataType::Row:
		finalPath = fs::path(baseString + ctx.meta.extension.row);
		break;
	case DataType::Ptr:
		finalPath = fs::path(baseString + ctx.meta.extension.ptr);
		break;
	case DataType::Col:
		finalPath = fs::path(baseString + ctx.meta.extension.col);
		break;
	}

	return finalPath;
}

static auto methodFind(Context & ctx, DeviceID myID)
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
				myInfo.ok = (myCtx.cache->find(tx.key) != myCtx.cache->end());
			}
			myInfo.hit = myInfo.ok;
			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();

	return in;
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
		printf("methodReady start at dev %d\n", myID);

		for (auto & tx : *in) {
			auto & myCtx = ctx.dataManagerCtx[myID];

			MemInfo<Vertex> myInfo = {
				0,
			};

			printf("allright dev %d got some request\n", myID);

			std::unique_lock<std::mutex> ul(*myCtx.cacheMtx);
			printf("allright dev %d got mutex\n", myID);

			bool iHaveLock = true;

			if (myCtx.cache->find(tx.key) != myCtx.cache->end()) {
				printf("[%2d] %s Hit!\n", myID, tx.key.print().c_str());
				myInfo = myCtx.cache->at(tx.key).info;
				myCtx.cache->at(tx.key).refCnt += 1;
				printf("[%2d] %s Hit  %d -> %d\n",
					   myID,
					   tx.key.print().c_str(),
					   myCtx.cache->at(tx.key).refCnt - 1,
					   myCtx.cache->at(tx.key).refCnt);

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				myInfo.hit = true;
			} else {
				printf("[%2d] %s Miss!\n", myID, tx.key.print().c_str());

				myInfo.byte = fs::file_size(genPath(ctx, tx.key));

				tryAllocate(ctx, tx.key, myID, myInfo, ul, iHaveLock);

				auto targetID  = myCtx.conn->upstream;
				auto otherInfo = requestToReady(ctx, tx.key, targetID);

				if (myID == -1) {
					printf("start to read!\n");
					printf("%p\n", myInfo.ptr);
					printf("otherInfo.path.c_str(): %s\n", otherInfo.path.c_str());
					auto fp = open64(otherInfo.path.c_str(), O_RDONLY);
					if (fp < 0) {
						printf("error = %s(%d)\n", strerror(errno), errno);
						exit(EXIT_FAILURE);
					}

					constexpr uint64_t cDef		 = (1L << 30); // chunk Default
					uint64_t		   chunkByte = (myInfo.byte < cDef) ? myInfo.byte : cDef;
					uint64_t		   bytePos	 = 0;
					while (bytePos < myInfo.byte) {
						chunkByte =
							(myInfo.byte - bytePos > chunkByte) ? chunkByte : myInfo.byte - bytePos;

						printf("cbyte bytePos/myInfo.byte = %ld %ld/%ld\n",
							   chunkByte,
							   bytePos,
							   myInfo.byte);
						auto loaded =
							pread(fp, &(((uint8_t *)myInfo.ptr)[bytePos]), chunkByte, bytePos);

						if (loaded < 0) {
							printf("error = %s(%d)\n", strerror(errno), errno);
							exit(EXIT_FAILURE);
						}

						bytePos += loaded;
					}

					close(fp);
				} else {
					// GPU
					// printf("[%2d] %s cudaMemcpy Host[%p]-> GPU[%p], %ld bytes)\n", myID,
					// tx.key.print().c_str(), otherInfo.ptr, myInfo.ptr, otherInfo.byte);
					cudaSetDevice(myID);
					cudaMemcpy(myInfo.ptr, otherInfo.ptr, otherInfo.byte, cudaMemcpyHostToDevice);
				}

				printf("[%2d] %s Memcpy/Read complete\n", myID, tx.key.print().c_str());

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
				printf("Storage: I got something!\n");
				switch (tx.method) {
				case Method::Find:
					fiber([&, myID, tx] {
						MemInfo<Vertex> myInfo = {
							0,
						};

						myInfo.hit = true;

						tx.cb->push(myInfo);
						tx.cb->close();
					}).detach();
					break;
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
			auto FindChan  = methodFind(ctx, myID);
			auto ReadyChan = methodReady(ctx, myID);
			auto DoneChan  = methodDone(ctx, myID);

			for (auto & tx : *ctx.dataManagerCtx[myID].chan) {
				switch (tx.method) {
				case Method::Find:
					FindChan->push(tx);
					break;
				case Method::Ready:
					ReadyChan->push(tx);
					break;
				case Method::Done:
					DoneChan->push(tx);
					break;
				}
			}

			FindChan->close();
			ReadyChan->close();
			DoneChan->close();
		}).detach();
	}
}