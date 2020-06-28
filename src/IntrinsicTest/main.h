#ifndef C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7
#define C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <fcntl.h>
#include <unistd.h>

using Vertex32 = uint32_t;

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

#endif /* C095FE4C_6D1F_4B64_B717_F7FDBDAF95F7 */
