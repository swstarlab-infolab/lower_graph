#ifndef C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7
#define C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "wg.h"

#include <array>
#include <boost/fiber/all.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// shorten long name
#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

#define WORDSZ 6 // do not adjust!!!

#define GWIDTH (1 << 24)

// Tweaking
#define CHANSZ		   16
#define UNORDEREDMAPSZ 1024
#define WORKERSZ	   1
#define MAPPERSZ	   1
#define TEMPFILEEXT	   ".el32"

// Primitive Types
using FileList		  = std::vector<fs::path>;
using RawData		  = std::vector<uint8_t>;
using Vertex32		  = uint32_t;
using Vertex64		  = uint64_t;
using Edge32		  = std::array<Vertex32, 2>;
using Edge64		  = std::array<Vertex64, 2>;
using GridIndex32	  = std::array<uint32_t, 2>;
using GridAndEdge	  = std::pair<GridIndex32, Edge32>;
using GridAndEdgeList = std::vector<GridAndEdge>;
using EdgeList32	  = std::vector<Edge32>;

struct KeyHash {
	std::size_t operator()(GridIndex32 const & k) const
	{
		auto a = std::hash<uint64_t>{}(uint64_t(k[0]) << 32);
		auto b = std::hash<uint64_t>{}(uint64_t(k[1]));
		return a ^ b;
	}
};

struct KeyEqual {
	bool operator()(GridIndex32 const & kl, GridIndex32 const & kr) const
	{
		return (kl[0] == kr[0] && kl[1] == kr[1]);
	}
};

using WriterEntry = std::unordered_map<GridIndex32,
									   std::shared_ptr<bchan<std::shared_ptr<EdgeList32>>>,
									   KeyHash,
									   KeyEqual>;

struct Context {
	fs::path	inFolder;
	fs::path	outFolder;
	std::string outName;
};

struct SplittedRawData {
	Vertex64  src;
	size_t	  cnt;
	uint8_t * dst;
};

#endif /* C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7 */
