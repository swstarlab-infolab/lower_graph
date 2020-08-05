#ifndef E58CCAAE_6910_4770_B569_4A24633C4BB5
#define E58CCAAE_6910_4770_B569_4A24633C4BB5

#include "type.h"

#include <BuddySystem/BuddySystem.h>
#include <boost/fiber/buffered_channel.hpp>
#include <cuda.h>
#include <gdrapi.h>
#include <memory>
#include <unordered_map>

namespace Data
{

struct MemInfo {
	void * ptr;
	size_t byte;
};

using Key = std::string;
struct Value {
	MemInfo info;
	int		refCnt;
};

struct Req {
	Key key;

	std::shared_ptr<boost::fibers::buffered_channel<MemInfo>> cb;
};

class Manager
{
private:
	int deviceID;
	struct {
		CUdeviceptr devPtr;
		void *		mapPtr;
		size_t		byte;
		struct {
			gdr_t	 g;
			gdr_mh_t mh;
		} gdr;

		std::shared_ptr<portable_buddy_system> buddy;
		std::unordered_map<Key, Value>		   cache;
		std::mutex							   cacheMtx;
	} mem;

	std::shared_ptr<boost::fibers::buffered_channel<Req>> doneQ;
	std::shared_ptr<boost::fibers::buffered_channel<Req>> reqQ;
	std::shared_ptr<boost::fibers::buffered_channel<Req>> reqMustAlloc;

	fs::path folderPath;

public:
	Manager(int const deviceID, fs::path const & folderPath);
	~Manager();

	void run();

	void *	alloc(size_t const size);
	MemInfo load(Key const & key);
	void	done(Key const & key);
};

} // namespace Data

#endif /* E58CCAAE_6910_4770_B569_4A24633C4BB5 */