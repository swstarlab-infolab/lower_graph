#ifndef D97C37A5_F5EF_4571_BF60_ACE120B68E58
#define D97C37A5_F5EF_4571_BF60_ACE120B68E58
#include <fcntl.h>
#include <unistd.h>
//#include <functional>
#include <string>

#define __CDEF (1L << 27) // 128MB

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

template <typename T>
auto fileSave(fs::path const & path, T * data, size_t byte)
{
	auto fp = open64(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);

	uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
	uint64_t pos	   = 0;

	while (pos < byte) {
		chunkSize = (byte - pos > chunkSize) ? chunkSize : byte - pos;
		auto b	  = write(fp, &(((uint8_t *)data)[pos]), chunkSize);
		pos += b;
	}

	close(fp);
}

template <typename T>
auto fileSaveAppend(fs::path const & path, T * data, size_t byte)
{
	auto fp = open64(path.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);

	uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
	uint64_t pos	   = 0;

	while (pos < byte) {
		chunkSize = (byte - pos > chunkSize) ? chunkSize : byte - pos;
		auto b	  = write(fp, &(((uint8_t *)data)[pos]), chunkSize);
		pos += b;
	}

	close(fp);
}

template <typename T>
auto fileLoad(fs::path const & path)
{
	auto fp	   = open64(path.c_str(), O_RDONLY);
	auto fbyte = fs::file_size(path);
	auto out   = std::make_shared<std::vector<T>>(fbyte / sizeof(T));

	uint64_t chunkSize = (fbyte < __CDEF) ? fbyte : __CDEF;
	uint64_t pos	   = 0;

	while (pos < fbyte) {
		chunkSize = (fbyte - pos > chunkSize) ? chunkSize : fbyte - pos;
		auto b	  = read(fp, &(((uint8_t *)(out->data()))[pos]), chunkSize);
		pos += b;
	}

	close(fp);

	return out;
}

#endif /* D97C37A5_F5EF_4571_BF60_ACE120B68E58 */
