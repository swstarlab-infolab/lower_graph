#ifndef C71CB3A3_88A0_421C_A9B3_D2A9852B151D
#define C71CB3A3_88A0_421C_A9B3_D2A9852B151D

#include "context.h"
#include "type.h"

#include <BuddySystem/BuddySystem.h>
#include <boost/fiber/all.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Data
{

class Manager;

extern std::unordered_map<int, sp<Manager>> managerSpace;

struct MemInfo {
	uint32_t * ptr;
	fs::path   path;
	size_t	   byte;
	bool	   ok;
	bool	   hit;
};

using Key = std::string;
struct Value {
	MemInfo info;
	int		refCnt;
};

struct Tx {
	Key				   key;
	sp<bchan<MemInfo>> cb;
};

class Manager
{
private:
	cudaStream_t myStream;
	int			 deviceID, upstreamDeviceID;

	std::unordered_map<Key, Value> cache;
	std::mutex					   cacheMtx;

	sp<bchan<Tx>> readyChan, doneChan;

	sp<portable_buddy_system> allocator;

	sp<bchan<Tx>> readyInternal(sp<Context> ctx);
	sp<bchan<Tx>> doneInternal();

	sp<void> myBuffer;

public:
	void init(sp<Context> ctx, int const deviceID, int const upstreamDeviceID);
	void run(sp<Context> ctx);

	void * malloc(size_t byte);
	void   free(void * const byte);

	MemInfo ready(Key const & key);
	MemInfo done(Key const & key);

	~Manager();
};

void init(sp<Context> ctx);
void run(sp<Context> ctx);

} // namespace Data

#endif /* C71CB3A3_88A0_421C_A9B3_D2A9852B151D */
