#ifndef C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7
#define C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <array>
#include <boost/fiber/all.hpp>
#include <fcntl.h>
//#include <fstream>
//#include <memory>
#include <iomanip>
#include <string>
#include <tuple>
//#include <unordered_map>
#include <vector>

// shorten long name
#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

#define __WordByteLength 6 // do not adjust!!!

#define __GridWidth (1 << 24)

// Tweaking
#define __ChannelSize		128
#define __UnorderedMapSize	1024
#define __WorkerCount		2
#define __MapperCount		64
#define __FilenameDelimiter "-"
#define __TempFileExt		".el32"
constexpr char const * __OutFileExts[] = {".row", ".ptr", ".col"};

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

/*
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
*/

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

template <typename T>
auto load(fs::path inFile)
{
	auto fp	   = open64(inFile.string().c_str(), O_RDONLY);
	auto fbyte = fs::file_size(inFile);
	auto out   = std::make_shared<std::vector<T>>(fbyte / sizeof(T));

	constexpr uint64_t cDef		 = (1L << 30); // chunk Default
	uint64_t		   chunkSize = (fbyte < cDef) ? fbyte : cDef;
	uint64_t		   pos		 = 0;

	while (pos < fbyte) {
		chunkSize = (fbyte - pos > chunkSize) ? chunkSize : fbyte - pos;
		auto b	  = read(fp, &(((uint8_t *)(out->data()))[pos]), chunkSize);
		pos += b;
	}

	close(fp);

	return out;
}

auto filenameEncode(GridIndex32 in)
{
	return std::to_string(in[0]) + __FilenameDelimiter + std::to_string(in[1]);
}

auto filenameDecode(std::string const & in)
{
	GridIndex32 gidx32 = {0, 0};

	auto delimPos = in.find(__FilenameDelimiter);
	gidx32[0]	  = atoi(in.substr(0, delimPos).c_str());
	gidx32[1]	  = atoi(in.substr(delimPos + 1, in.size()).c_str());

	return gidx32;
}

auto walk(fs::path const & inFolder, std::string const & ext)
{
	auto out = std::make_shared<FileList>();
	for (fs::recursive_directory_iterator iter(inFolder), end; iter != end; iter++) {
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
			if (ext != "" && iter->path().extension() != ext) {
				continue;
			}
			out->push_back(iter->path());
		}
	}
	return out;
}

void log(std::string const & s)
{
	auto currentTimeAndDate = [] {
		auto			  now		= std::chrono::system_clock::now();
		auto			  in_time_t = std::chrono::system_clock::to_time_t(now);
		std::stringstream ss;
		ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
		return ss.str();
	};

	printf("%s %s\n", currentTimeAndDate().c_str(), s.c_str());
}

#endif /* C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7 */
