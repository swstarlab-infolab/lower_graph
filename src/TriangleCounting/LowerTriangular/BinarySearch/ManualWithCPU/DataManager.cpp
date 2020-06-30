#include "DataManager.h"

#include "context.h"
#include "type.h"
#include "util.h"

#include <cuda_runtime.h>
#include <iostream>

void DataManager::init(int const ID, sp<DataManager> upstream)
{
	this->ID	   = ID;
	this->upstream = upstream;

	this->chan.ready = makeSp<bchan<Tx>>(ctx.chanSz);
	this->chan.find	 = makeSp<bchan<Tx>>(ctx.chanSz);
	this->chan.done	 = makeSp<bchan<Tx>>(ctx.chanSz);

	if (this->ID > -1) {
		this->initGPU();
	} else if (this->ID == -1) {
		this->initCPU();
	} else {
		// this->initStorage();
	}
}

void DataManager::initCPU()
{
	std::cout << "DM: CPU init, Me: " << this << ", Upstream: " << this->upstream << std::endl;

	// size_t freeMem = (1L << 37); // 128GB
	size_t freeMem = (1L << 33); // 8GB

	this->mem.buf.byte = freeMem;
	cudaMallocHost((void **)&this->mem.buf.ptr, this->mem.buf.byte);
	this->mem.buddy.init(memrgn_t{this->mem.buf.ptr, this->mem.buf.byte}, 8, 1);

	this->mem.cache = std::make_shared<Cache>(1 << 24);
}

void DataManager::initGPU()
{
	std::cout << "DM: GPU " << this->ID << " init, Me: " << this << ", Upstream: " << this->upstream
			  << std::endl;

	size_t freeMem;
	cudaSetDevice(this->ID);
	cudaMemGetInfo(&freeMem, nullptr);
	freeMem -= (1L << 29);

	cudaSetDevice(this->ID);
	this->mem.buf.byte = freeMem;
	cudaMalloc((void **)&this->mem.buf.ptr, this->mem.buf.byte);
	this->mem.buddy.init(memrgn_t{this->mem.buf.ptr, this->mem.buf.byte}, 256, 1);

	this->mem.cache = std::make_shared<Cache>(1 << 24);
}

void * DataManager::manualAlloc(size_t const byte) { return this->mem.buddy.allocate(byte); }

void DataManager::run()
{
	if (this->ID < -1) {
		this->routineStorage();
	} else {
		this->methodFind();
		this->methodReady();
		this->methodDone();
	}
}

std::string DataManager::pathEncode(CacheKey const key)
{
	auto baseString = (ctx.folder / fs::path(filenameEncode(key.idx))).string();

	fs::path finalPath;

	switch (key.type) {
	case Type::row:
		finalPath = fs::path(baseString + ".row");
		break;
	case Type::ptr:
		finalPath = fs::path(baseString + ".ptr");
		break;
	case Type::col:
		finalPath = fs::path(baseString + ".col");
		break;
	}

	return finalPath.string();
}

void DataManager::routineStorage()
{
	std::thread([this] {
		for (auto & tx : *this->chan.find) {
			// fiber([this, tx] {
			printf("DM: Storage find\n");
			TxCb myInfo;
			myInfo.info.ptr	 = nullptr;
			myInfo.info.byte = 0;
			myInfo.path		 = "";
			myInfo.hit		 = true;
			myInfo.ok		 = true;

			printf("DM: Storage find, (%d,%d)\n", tx.key.idx[0], tx.key.idx[1]);
			tx.cb->push(myInfo);
			tx.cb->close();
			//}).detach();
		}
	}).detach();

	std::thread([this] {
		for (auto & tx : *this->chan.ready) {
			// fiber([this, tx] {
			printf("DM: Storage ready\n");
			TxCb myInfo;
			myInfo.info.ptr	 = nullptr;
			myInfo.path		 = this->pathEncode(tx.key);
			myInfo.info.byte = fs::file_size(myInfo.path);
			myInfo.ok		 = true;
			myInfo.hit		 = true;

			printf("DM: Storage ready, (%d,%d) %ld Bytes\n",
				   tx.key.idx[0],
				   tx.key.idx[1],
				   myInfo.info.byte);
			tx.cb->push(myInfo);
			tx.cb->close();
			//}).detach();
		}
	}).detach();

	std::thread([this] {
		for (auto & tx : *this->chan.done) {
			// fiber([this, tx] {
			printf("DM: Storage done\n");
			TxCb myInfo;
			myInfo.info.ptr	 = nullptr;
			myInfo.info.byte = 0;
			myInfo.path		 = "";
			myInfo.ok		 = true;
			myInfo.hit		 = true;

			printf("DM: Storage done, (%d,%d)\n", tx.key.idx[0], tx.key.idx[1]);
			tx.cb->push(myInfo);
			tx.cb->close();
			//}).detach();
			break;
		}
	}).detach();
}

void DataManager::methodFind()
{
	std::thread([this] {
		for (auto & tx : *this->chan.find) {
			printf("DM: dev%d find, (%d,%d)\n", this->ID, tx.key.idx[0], tx.key.idx[1]);
			TxCb myInfo;
			myInfo.info.ptr	 = nullptr;
			myInfo.info.byte = 0;
			myInfo.path		 = "";

			{
				std::lock_guard<std::mutex> lg(this->mem.cacheMtx);
				myInfo.ok = (this->mem.cache->find(tx.key) != this->mem.cache->end());
			}
			myInfo.hit = myInfo.ok;

			tx.cb->push(myInfo);
			tx.cb->close();
		}
	}).detach();
}

void DataManager::tryAllocate(CacheKey const				 key,
							  TxCb &						 txcb,
							  std::unique_lock<std::mutex> & ul,
							  bool &						 iHaveLock)
{
	while (true) {
		txcb.info.ptr = (Vertex32 *)(this->mem.buddy.allocate(txcb.info.byte));

		if (txcb.info.ptr != nullptr) {
			this->mem.cache->insert({key, {txcb.info, 1}});

			txcb.ok	 = true;
			txcb.hit = false;

			return;
		} else {
			// allocation failure
			if (this->mem.cache->size() > 0) {
				bool evicted = false;
				while (!evicted) {
					if (!iHaveLock) {
						ul.lock();
						iHaveLock = true;
					}

					for (auto it = this->mem.cache->begin(); it != this->mem.cache->end();) {
						if (it->second.refCnt == 0) {
							this->mem.buddy.deallocate(it->second.info.ptr);
							it		= this->mem.cache->erase(it);
							evicted = true;
							printf("DM: dev %d delete, (%d,%d), %ld Bytes\n",
								   this->ID,
								   it->first.idx[0],
								   it->first.idx[1],
								   it->second.info.byte);
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

void DataManager::methodReady()
{
	std::thread([this] {
		for (auto & tx : *this->chan.ready) {
			TxCb txcb;
			txcb.info.ptr  = nullptr;
			txcb.info.byte = 0;
			txcb.path	   = "";

			std::unique_lock<std::mutex> ul(this->mem.cacheMtx);

			bool iHaveLock = true;

			if (this->mem.cache->find(tx.key) != this->mem.cache->end()) {
				txcb.info = this->mem.cache->at(tx.key).info;

				this->mem.cache->at(tx.key).refCnt += 1;

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				printf("DM: dev %d ready hit, (%d,%d) %ld Bytes\n",
					   this->ID,
					   tx.key.idx[0],
					   tx.key.idx[1],
					   txcb.info.byte);
				txcb.hit = true;
			} else {
				txcb.info.byte = fs::file_size(pathEncode(tx.key));

				tryAllocate(tx.key, txcb, ul, iHaveLock);

				auto otherTxcb = this->upstream->reqReady(tx.key.idx, tx.key.type);

				if (this->ID == -1) {
					auto fp = open64(otherTxcb.path.c_str(), O_RDONLY);

					constexpr uint64_t cDef		 = (1L << 30); // chunk Default
					uint64_t		   chunkByte = (txcb.info.byte < cDef) ? txcb.info.byte : cDef;
					uint64_t		   bytePos	 = 0;
					while (bytePos < txcb.info.byte) {
						chunkByte = (txcb.info.byte - bytePos > chunkByte)
										? chunkByte
										: txcb.info.byte - bytePos;
						auto loaded = read(fp, &(((uint8_t *)txcb.info.ptr)[bytePos]), chunkByte);
						bytePos += loaded;
					}

					close(fp);
				} else {
					cudaSetDevice(this->ID);
					cudaMemcpy(txcb.info.ptr,
							   otherTxcb.info.ptr,
							   otherTxcb.info.byte,
							   cudaMemcpyHostToDevice);
				}

				if (iHaveLock) {
					ul.unlock();
					iHaveLock = false;
				}

				this->upstream->reqDone(tx.key.idx, tx.key.type);

				printf("DM: dev %d ready miss, (%d,%d) %ld Bytes\n",
					   this->ID,
					   tx.key.idx[0],
					   tx.key.idx[1],
					   txcb.info.byte);
				txcb.hit = false;
			}

			tx.cb->push(txcb);
			tx.cb->close();
		}
	}).detach();
}

void DataManager::methodDone()
{
	std::thread([this] {
		for (auto & tx : *this->chan.done) {
			TxCb txcb;

			{
				std::lock_guard<std::mutex> lg(this->mem.cacheMtx);
				this->mem.cache->at(tx.key).refCnt -= 1;
			}

			txcb.ok	 = true;
			txcb.hit = true;

			printf("DM: dev %d done, (%d,%d)\n", this->ID, tx.key.idx[0], tx.key.idx[1]);
			tx.cb->push(txcb);
			tx.cb->close();
		}
	}).detach();
}

DataManager::TxCb DataManager::reqFind(GridIndex32 const idx, Type const type)
{
	printf("DM: dev %d, reqFind: (%d,%d)\n", this->ID, idx[0], idx[1]);
	Tx tx;
	tx.key.idx	= idx;
	tx.key.type = type;
	tx.cb		= makeSp<bchan<TxCb>>(2);

	this->chan.find->push(tx);

	TxCb result;
	for (auto & res : *tx.cb) {
		result = res;
	}

	return result;
}

DataManager::TxCb DataManager::reqReady(GridIndex32 const idx, Type const type)
{
	printf("DM: dev %d, reqReady: (%d,%d)\n", this->ID, idx[0], idx[1]);
	Tx tx;
	tx.key.idx	= idx;
	tx.key.type = type;
	tx.cb		= makeSp<bchan<TxCb>>(2);

	this->chan.ready->push(tx);

	TxCb result;
	for (auto & res : *tx.cb) {
		result = res;
	}

	return result;
}

DataManager::TxCb DataManager::reqDone(GridIndex32 const idx, Type const type)
{
	printf("DM: dev %d, reqDone: (%d,%d)\n", this->ID, idx[0], idx[1]);

	Tx tx;
	tx.key.idx	= idx;
	tx.key.type = type;
	tx.cb		= makeSp<bchan<TxCb>>(2);

	this->chan.done->push(tx);

	TxCb result;
	for (auto & res : *tx.cb) {
		result = res;
	}

	return result;
}

void DataManager::closeAllChan()
{
	this->chan.find->close();
	this->chan.ready->close();
	this->chan.done->close();
}