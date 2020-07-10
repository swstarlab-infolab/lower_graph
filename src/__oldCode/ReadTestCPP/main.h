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
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// shorten long name
#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

#define __WordByteLength 6 // do not adjust!!!

#define __GridWidth (1 << 24)

// Tweaking
#define __ChannelSize 128
#define __WorkerCount 1
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

	uint64_t chunkSize = (1L << 30);
	uint64_t pos	   = 0;
	while (pos < fbyte) {
		chunkSize	= (fbyte - pos > chunkSize) ? chunkSize : fbyte - pos;
		auto loaded = read(fp, &(out->at(pos)), chunkSize);
		pos += loaded;
	}

	close(fp);

	return out;
}

/*

	f.seekg(0, std::ios::beg);
	for (uint64_t pos = 0; pos < fbyte;) {
		auto b = f.readsome((char *)(&(out->at(pos))), fbyte - pos);
		std::cout << "b=" << b << ",pos=" << pos << ",left=" << fbyte - pos << std::endl;
		// under construction
		pos += b;
		f.seekg(pos, std::ios::beg);
		std::cin.ignore();
	}
	f.close();

	*/

std::shared_ptr<FileList> walk(fs::path const & inFolder, std::string const & ext);
void					  log(std::string const & str);

void phase1(Context const & ctx);

#endif /* C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7 */
