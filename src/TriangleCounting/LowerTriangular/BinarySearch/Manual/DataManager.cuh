#ifndef B0343A5C_B8D6_4967_809A_3487B01AAA67
#define B0343A5C_B8D6_4967_809A_3487B01AAA67

#include "make.h"
#include "type.h"

#include <BuddySystem/BuddySystem.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

/*
static size_t getFileSize(fs::path const& path)
{
	std::ifstream f;
	f.open(path);
	f.seekg(0, std::ios::end);
	auto const fileSize = f.tellg();
	f.seekg(0, std::ios::beg);
	f.close();
	return fileSize;
}
*/

auto DataManagerMainMemory(Context & ctx, int myID)
{
	// Types for Cache
	auto _hash = [](Key const & k) {
		auto a = std::hash<uint64_t>{}(uint64_t(k.idx[0]) << (8 * sizeof(k.idx[0])));
		auto b = std::hash<uint64_t>{}(k.idx[1]);
		auto c = std::hash<uint64_t>{}(k.type);
		return a ^ b ^ c;
	};
	auto _equal = [](Key const & kl, Key const & kr) {
		return (kl.idx[0] == kr.idx[0] && kl.idx[1] == kr.idx[1] && kl.type == kr.type);
	};

	using Cache = std::unordered_map<Key, CacheValue, decltype(_hash), decltype(_equal)>;

	using DataTx		 = Tx<DataMethod, MemInfo>;
	using DataTxCallback = bchan<MemInfo>;
	// using DataTxChan = bchan<DataTx>;

	Cache cache(1L << 10, _hash, _equal);
	static std::mutex
		cacheMtx; // this is used in detached fibers; it must sustain until all thread is completed

	// Prepare Buddy System
	size_t _bufByte = 1L << 20;
	void * _buf;
	cudaHostAlloc(&_buf, _bufByte, cudaHostAllocPortable);
	portable_buddy_system buddy;
	buddy.init(memrgn_t{_buf, _bufByte}, 8, 1);

	for (auto & req : *(ctx.memChan[myID].get())) {
		switch (req.method) {
		case DataMethod::Find: {
			fiber([&, req] {
				MemInfo info = {
					0,
				};
				{
					std::lock_guard<std::mutex> lg(cacheMtx);
					info.ok = (cache.find(req.key) != cache.end());
				}
				// printf("DAT: %d (%d, %d) %d\n", myID, req.key.idx[0], req.key.idx[1],
				// req.key.type);
				req.cb.get()->push(info);
				// printf("DAT: %d (%d, %d) %d PUSHED, TRY CLOSE\n", myID, req.key.idx[0],
				// req.key.idx[1], req.key.type);
				req.cb.get()->close();
				// printf("DAT: %d (%d, %d) %d GOODBYE\n", myID, req.key.idx[0], req.key.idx[1],
				// req.key.type);
			}).detach();
		} break;
		case DataMethod::Ready: {
			fiber([&, req] {
				MemInfo myInfo = {
					0,
				};
				std::unique_lock<std::mutex> ul(cacheMtx);
				if (cache.find(req.key) != cache.end()) {
					// update reference count
					cache[req.key].refCnt += 1;
					printf("Device%d, Key=<(%d,%d),%d>, Cached, RefCnt=%d->%d\n",
						   myID,
						   req.key.idx[0],
						   req.key.idx[1],
						   req.key.type,
						   cache[req.key].refCnt - 1,
						   cache[req.key].refCnt);

					// unlock lock
					ul.unlock();

					myInfo.ok = true;
					req.cb.get()->push(myInfo);
					req.cb.get()->close();
					return;
				} else {
					// 0. grid size
					myInfo.byte = 1L << 17; // temporary value

					// 1. Try allocate
					while (true) {
						printf("Device%d, Key=<(%d,%d),%d>, Not-cached\n",
							   myID,
							   req.key.idx[0],
							   req.key.idx[1],
							   req.key.type);
						ul.lock myInfo.ptr = buddy.allocate(myInfo.byte);

						if (myInfo.ptr != nullptr) {
							cache[req.key] = {myInfo, 1};
							printf("Device%d, Key=<(%d,%d),%d>, New, RefCnt=%d->%d\n",
								   myID,
								   req.key.idx[0],
								   req.key.idx[1],
								   req.key.type,
								   cache[req.key].refCnt - 1,
								   cache[req.key].refCnt);

							printf("Device%d, Key=<(%d,%d),%d>, callbackptr:%p\n",
								   myID,
								   req.key.idx[0],
								   req.key.idx[1],
								   req.key.type,
								   req.cb.get());

							ul.unlock();
							myInfo.ok = true;

							req.cb.get()->push(myInfo);
							printf("Device%d, Key=<(%d,%d),%d>, Pushed\n",
								   myID,
								   req.key.idx[0],
								   req.key.idx[1],
								   req.key.type);
							req.cb.get()->close();
							printf("Device%d, Key=<(%d,%d),%d>, Closed\n",
								   myID,
								   req.key.idx[0],
								   req.key.idx[1],
								   req.key.type);
							return;
						} else {
							// if malloc failed
							if (cache.size() > 0) {
								// first of all, unlock lock so that any thread can modify
								// cache
								ul.unlock();
								bool evictSuccess = false;
								while (!evictSuccess) {
									// then lock again,
									ul.lock();
									// evict something
									printf("Device%d, Key=<(%d,%d),%d>, Try Evict Another\n",
										   myID,
										   req.key.idx[0],
										   req.key.idx[1],
										   req.key.type);
									for (auto it = cache.begin(); it != cache.end();) {
										if (it->second.refCnt == 0) {
											printf(
												"Device%d, Key=<(%d,%d),%d>, Evict=<(%d,%d),%d>\n",
												myID,
												req.key.idx[0],
												req.key.idx[1],
												req.key.type,
												it->first.idx[0],
												it->first.idx[1],
												it->first.type);
											buddy.deallocate(it->second.info.ptr);
											it			 = cache.erase(it);
											evictSuccess = true;
											break;
										} else {
											++it;
										}
									}
									// unlock lock
									ul.unlock();
								}
							} else {
								// malloc failed and cache size is zero? somethings wronk.
								printf("Device%d, Key=<(%d,%d),%d>, Strange Error\n",
									   myID,
									   req.key.idx[0],
									   req.key.idx[1],
									   req.key.type);
								ul.unlock();
								myInfo.ok = false;
								req.cb.get()->push(myInfo);
								req.cb.get()->close();
								return;
							}
						}
					}

					/*
					// 2. Ask to neikhborhood
					bool neighborHave = false;
					int neighborID;
					{
						auto& nList = ctx.conn[myID].neighbor;
						if (nList.size() > 0) {
							std::vector<bool> nSuccess(nList.size());
							std::vector<fiber> waitGroup(nList.size());
							for (size_t i = 0; i < nList.size(); i++) {
								waitGroup[i] = fiber([&, i] {
									auto callback = std::make_shared<DataTxCallback>();
									DataTx tx;
									tx.key = req.key;
									tx.method = DataMethod::Find;
									tx.cb = callback;
									ctx.memChan[nList[i]].get()->push(tx);

									MemInfo otherInfo;
									tx.cb.get()->pop(otherInfo);
									nSuccess[i] = otherInfo.ok;
								});
							}

							for (auto& w : waitGroup) {
								if (w.joinable()) {
									w.join();
								}
							}

							for (size_t i = 0; i < nList.size(); i++) {
								if (nSuccess[i]) {
									neighborHave = true;
									neighborID = nList[i];
									break;
								}
							}
						}
					}

					// 3. Request Memory
					{
						auto& upstreamID = ctx.conn[myID].upstream;
						auto reqID = (neighborHave) ? neighborID : upstreamID;

						if (reqID != myID) {
							auto callback = std::make_shared<DataTxCallback>();
							DataTx tx;
							tx.key = req.key;
							tx.method = DataMethod::Ready;
							tx.cb = callback;
							ctx.memChan[reqID].get()->push(tx);

							MemInfo otherInfo;
							tx.cb.get()->pop(otherInfo);
							memcpy(myInfo.ptr, otherInfo.ptr, otherInfo.byte);

							tx.method = DataMethod::Done;
							ctx.memChan[reqID].get()->push(tx);

							myInfo.ok = true;
							req.cb.get()->push(myInfo);
							req.cb.get()->close();

							tx.cb->pop(otherInfo);

							return;
						} else {
							// nothing to do...
							myInfo.ok = false;
							req.cb.get()->push(myInfo);
							req.cb.get()->close();
							return;
						}
					}
					*/
				}
			}).detach();
		} break;
		case DataMethod::Done: {
			fiber([&, req] {
				MemInfo info = {
					0,
				};
				{
					std::lock_guard<std::mutex> lg(cacheMtx);
					cache[req.key].refCnt -= 1;
					printf("Device%d, Key=<(%d,%d),%d>, RefCnt=%d->%d\n",
						   myID,
						   req.key.idx[0],
						   req.key.idx[1],
						   req.key.type,
						   cache[req.key].refCnt + 1,
						   cache[req.key].refCnt);
					/* for testing */
					/*
					if (cache[req.key].refCnt <= 0) {
						buddy.deallocate(cache[req.key].info.ptr);
						cache.erase(req.key);
					}
					*/
					/* for testing */
				}
				info.ok = true;
				req.cb.get()->push(info);
				req.cb.get()->close();
			}).detach();
		} break;
		}
	}
}

/*
auto DataManagerStorage(Context& ctx, int myID)
{
	std::unordered_map<GridIndex, FileInfo> memmap;
	std::mutex cacheMtx;

	for (auto& req : *ctx.fileChan[myID].get()) {
		switch (req.method) {
		case Method::QUERY: {
			bchan<bool> callback(1);
			CacheTx<bool> tx;
			tx.method = CacheTxMethod::FIND;
			tx.gidx = req.gidx;
			tx.info = callbacg;

			bool result;
			callbacg.pop(result);

			if (result) {
				auto const baseString = std::string(ctx.folderPath) +
std::to_string(g[0]) + "-" + std::to_string(g[1]) + "."; FileInfo info;
				info.path.row = std::ifstream(fs::path(baseString +
ctx.meta.extension.row)); info.path.ptr = std::ifstream(fs::path(baseString +
ctx.meta.extension.ptr)); info.path.col = std::ifstream(fs::path(baseString +
ctx.meta.extension.col)); info.ok = true; req.callback.push(info)
			}
		} break;
		case Method::READY: {
			std::unique_lock<std::mutex> ul(cacheMtx);
		} break;
		case Method::DONE: {
			std::unique_lock<std::mutex> ul(cacheMtx);

		} break;
		case Method::DESTROY: {
			std::unique_lock<std::mutex> ul(cacheMtx);

		} break;
		}
	}
}
	*/

auto DataManagerDeviceMemory(Context & ctx, int myID) { return false; }

void DataManager(Context & ctx, int myID)
{
	if (myID < 0) {
		// fiber([&, myID] { DataManagerStorage(ctx, myID); }).detach();
	} else if (myID == 0) {
		fiber([&, myID] { DataManagerMainMemory(ctx, myID); }).detach();
	} else {
		fiber([&, myID] { DataManagerMainMemory(ctx, myID); }).detach(); // for testing
	}
}

/*

static void loadFile(fs::path const& path, void* ptr, size_t byte)
{
	std::ifstream f;
	f.open(path, std::ios::binary);
	f.read((char*)ptr, byte);
	f.close();
}

static auto tryLoad(
	Context const& ctx,
	portable_buddy_system& buddy,
	DataType::GridIndex const& g,
	DataType::CacheValue& v)
{

	// get file size
	std::array<DataType::Memory, 3> mem;
	for (unsigned int i = 0; i < mem.size(); i++) {
		mem[i].second = getFileSize(path[i]);
		mem[i].first = buddy.allocate(mem[i].second);
		if (mem[i].first == nullptr) {
			return false;
		}
		loadFile(path[i], mem[i].first, mem[i].second);
	}

	// success
	v.arr = mem;

	return true;
}

void loader(
	Context const& ctx,
	ChanLoadReq& loadReq,
	std::vector<std::shared_ptr<ChanLoadRes>>& loadRes,
	ChanLoadComp& loadComp)
{
	fprintf(stdout, "[LOADER] START\n");

	// memory allocation
	auto map = makeHashMap<DataType::GridIndex, DataType::CacheValue>(
		1024,
		[](DataType::GridIndex const& key) {
			using KeyType =
std::remove_const<std::remove_reference<decltype(key.front())>::type>::type;
			return std::hash<KeyType>()(key[0]) ^ (std::hash<KeyType>()(key[1])
<< 1);
		});

	DataType::MemoryShared myMemory;
	myMemory.second = 1024L * 1024L * 1024L;
	myMemory.first = allocCUDA<char>(myMemory.second);

	portable_buddy_system buddy;
	buddy.init(memrgn_t { (void*)myMemory.first.ket(), myMemory.second }, 8, 1);

	fprintf(stdout, "[LOADER] Init Complete\n");

	auto karbakeCollector = boost::fibers::fiber([&] {
		for (auto& comp : loadComp) {
			decltype(map)::accessor a;
			map.find(a, comp.idx);
			a->second.counter--;
		}
	});

	auto cacheman = boost::fibers::fiber([&] {
		for (auto& req : loadReq) {
			fprintf(stdout, "[LOADER] Got Request!: (%d,%d)\n", req.idx[0],
req.idx[1]);

			DataType::CacheValue v;
			decltype(map)::accessor a;

			if (map.find(a, req.idx)) {
				fprintf(stdout, "[LOADER] Present: (%d,%d), %p,%ld\n",
req.idx[0], req.idx[1], v.arr[0].first, v.arr[0].second);

				v.arr = a->second.arr;
				a->second.counter++;
				a.release();
			} else {
				// not present
				if (tryLoad(ctx, buddy, req.idx, v)) {
					fprintf(stdout, "[LOADER] Try Load Success: (%d,%d),
%p,%ld\n", req.idx[0], req.idx[1], v.arr[0].first, v.arr[0].second); v.counter =
1; map.insert(a, req.idx); a->second.arr = v.arr; a->second.counter = v.counter;
					a.release();
				} else {
					// find evictable
					a.release();
				}
			}

			MessakeType::LoadRes res;
			// enqueue;
			res.idx = req.idx;
			res.arr = v.arr;

			loadRes[req.deviceID].ket()->push(res);
		}

		loadComp.close();
	});

	karbakeCollector.join();
	cacheman.join();

	for (auto& c : loadRes) {
		c->close();
	}
}
*/
#endif /* B0343A5C_B8D6_4967_809A_3487B01AAA67 */
