#ifndef F2E64CCA_9380_48B5_8695_08A9A2A38B40
#define F2E64CCA_9380_48B5_8695_08A9A2A38B40

#include "type.h"

#include <BuddySystem/BuddySystem.h>
#include <memory>
#include <mutex>
#include <unordered_map>

class DataManager
{
public:
	enum class Type : uint8_t { row, ptr, col };

	enum class Method : uint8_t { find, ready, done };

	struct CacheKey {
		GridIndex32 idx;
		Type		type;
		std::string print() const
		{
			std::string result =
				"<(" + std::to_string(this->idx[0]) + "," + std::to_string(this->idx[1]) + "),";
			switch (this->type) {
			case Type::row:
				result += "Row";
				break;
			case Type::ptr:
				result += "Ptr";
				break;
			case Type::col:
				result += "Col";
				break;
			}
			result += ">";
			return result;
		}
	};

	struct TxCb {
		MemInfo<Vertex32> info;
		std::string		  path;
		bool			  hit, ok;
	};

	struct Tx {
		CacheKey		key;
		sp<bchan<TxCb>> cb;
	};

private:
	struct CacheVal {
		MemInfo<Vertex32> info;
		int				  refCnt;
	};

	struct CacheKeyHash {
		std::size_t operator()(CacheKey const & k) const
		{
			auto a = std::hash<uint64_t>{}(uint64_t(k.idx[0]) << (8 * sizeof(k.idx[0])));
			auto b = std::hash<uint64_t>{}(k.idx[1]);
			auto c = std::hash<uint64_t>{}(uint64_t(k.type));
			return a ^ b ^ c;
		}
	};

	struct CacheKeyEqual {
		bool operator()(CacheKey const & kl, CacheKey const & kr) const
		{
			return (kl.idx[0] == kr.idx[0] && kl.idx[1] == kr.idx[1] && kl.type == kr.type);
		}
	};

private:
	using Cache = std::unordered_map<CacheKey, CacheVal, CacheKeyHash, CacheKeyEqual>;

	struct {
		MemInfo<void>		  buf;
		portable_buddy_system buddy;
		sp<Cache>			  cache;	// Cache
		std::mutex			  cacheMtx; // Cache Mutex
	} mem;

	int ID;

	sp<DataManager> upstream = nullptr;

	struct {
		sp<bchan<Tx>> ready, find, done;
	} chan;

	std::string pathEncode(CacheKey const key);

	void tryAllocate(CacheKey const					key,
					 TxCb &							txcb,
					 std::unique_lock<std::mutex> & ul,
					 bool &							iHaveLock);

	void initCPU();
	void initGPU();
	// void initStorage();

	void methodReady(); // Async Function
	void methodFind();	// Async Function
	void methodDone();	// Async Function

	void routine();		   // Async Function
	void routineStorage(); // Async Function

public:
	void init(int const ID, sp<DataManager> upstream);
	void run(); // Async Function

	void * manualAlloc(size_t const byte);
	// void   req(Tx & in) { this->chan.tx->push(in); }

	TxCb reqFind(GridIndex32 const key, Type const type);
	TxCb reqReady(GridIndex32 const key, Type const type);
	TxCb reqDone(GridIndex32 const key, Type const type);

	void closeAllChan();
};
#endif /* F2E64CCA_9380_48B5_8695_08A9A2A38B40 */
