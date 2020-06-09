#ifndef C26BEB06_5F3E_48D4_AB98_5DE67AD09131
#define C26BEB06_5F3E_48D4_AB98_5DE67AD09131
#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <BuddySystem/BuddySystem.h>
#include <GridCSR/GridCSR.h>
#include <array>
#include <boost/fiber/all.hpp>
#include <memory>
#include <unordered_map>

struct MemInfo;

using Vertex	 = uint32_t;
using Lookup	 = uint32_t;
using Count		 = unsigned long long;
using GridIndex	 = std::array<uint32_t, 2>;
using ThreeGrids = std::array<GridIndex, 3>;
using DeviceID	 = int32_t;
using Grid		 = std::array<MemInfo, 3>;
using Grids		 = std::array<Grid, 3>;

#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

// Memory Information
struct MemInfo {
	void *		ptr;
	std::string path;
	size_t		byte;
	bool		ok;
	bool		hit;

	std::string print()
	{
		char a[24];
		sprintf(a, "%p", ptr);
		std::string b(a);
		return "<" + b + "," + path + "," + std::to_string(byte) + "," + std::to_string(ok) + "," +
			   std::to_string(hit) + ">";
	}
};

template <typename Type>
struct Memory {
	std::shared_ptr<Type> ptr;
	size_t				  byte;

	size_t count() { return this->byte / sizeof(Type); }
};

enum DataType : uint32_t { Row, Ptr, Col };

struct Key {
	GridIndex idx;
	DataType  type;

	std::string print()
	{
		std::string result =
			"<(" + std::to_string(this->idx[0]) + "," + std::to_string(this->idx[1]) + "),";
		switch (this->type) {
		case DataType::Row:
			result += "Row";
			break;
		case DataType::Ptr:
			result += "Ptr";
			break;
		case DataType::Col:
			result += "Col";
			break;
		}
		result += ">";
		return result;
	}
};

struct CacheValue {
	MemInfo info;
	int		refCnt;
};

enum Method : uint32_t { Find, Ready, Done };

// Transaction
struct Tx {
	Key								key;
	Method							method;
	std::shared_ptr<bchan<MemInfo>> cb;
};

struct CommandResult {
	ThreeGrids gidx;
	Count	   triangle;
	double	   elapsedTime;
	int		   deviceID;
};

struct Command {
	ThreeGrids gidx;
};

// Types for Cache
struct KeyHash {
	std::size_t operator()(Key const & k) const
	{
		auto a = std::hash<uint64_t>{}(uint64_t(k.idx[0]) << (8 * sizeof(k.idx[0])));
		auto b = std::hash<uint64_t>{}(k.idx[1]);
		auto c = std::hash<uint64_t>{}(k.type);
		return a ^ b ^ c;
	}
};

struct KeyEqual {
	bool operator()(Key const & kl, Key const & kr) const
	{
		return (kl.idx[0] == kr.idx[0] && kl.idx[1] == kr.idx[1] && kl.type == kr.type);
	}
};

struct DataManagerContext {

	using Cache = std::unordered_map<Key, CacheValue, KeyHash, KeyEqual>;

	struct Connections {
		DeviceID			  upstream;
		std::vector<DeviceID> neighbor;
	};

	std::shared_ptr<Cache>		cache;	  // Cache
	std::shared_ptr<std::mutex> cacheMtx; // Cache Mutex

	std::shared_ptr<void>				   buf;	  // Pre-allocated buffer location
	std::shared_ptr<portable_buddy_system> buddy; // Pre-allocated buffer allocator

	std::shared_ptr<Connections> conn; // Connection to other memory
	std::shared_ptr<bchan<Tx>>	 chan; // Transaction input channel
};

struct ExecutionManagerContext {
	struct {
		Memory<Lookup> G0, G2, temp;
	} lookup;

	Memory<void>  cub;
	Memory<Count> count;
};

struct Context {
	fs::path			  folderPath;		// folder path
	int					  deviceCount = -1; // device count
	std::array<size_t, 3> setting;			// setting (cudaStreams, cudaBlocks, cudaThreads)
	GridCSR::MetaData	  meta;				// Grid metadata

	std::unordered_map<DeviceID, DataManagerContext>	  dataManagerCtx;
	std::unordered_map<DeviceID, ExecutionManagerContext> executionManagerCtx;
};
#endif /* C26BEB06_5F3E_48D4_AB98_5DE67AD09131 */
