#ifndef D5D78C16_48D8_4581_8C8D_E4336F18BDC9
#define D5D78C16_48D8_4581_8C8D_E4336F18BDC9

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <array>
#include <string>

#define GRIDWIDTH	(1UL << 24)
#define EXP_BITMAP0 (12UL)
#define EXP_BITMAP1 (5UL)
//#define CPUOFF

template <typename T>
struct DataInfo;

template <>
struct DataInfo<void> {
	void *	 addr;
	size_t	 byte;
	fs::path path;
};

template <typename T>
struct DataInfo {
	T *		  addr;
	size_t	  byte;
	fs::path  path;
	T &		  operator[](size_t const i) { return this->addr[i]; }
	T const & operator[](size_t const i) const { return this->addr[i]; }
	size_t	  count() const { return byte / sizeof(T); }
};

// template <typename T>
/*
struct DataInfoContainer {
	void *	 addr;
	size_t	 byte;
	fs::path path;

	DataInfoContainer(DataInfoContainer const & other)
	{
		this->addr = other.addr;
		this->byte = other.byte;
		this->path = other.path;
	}
};

template <typename T>
struct DataInfo : DataInfoContainer {
	T &		  operator[](size_t const i) { return this->addr[i]; }
	T const & operator[](size_t const i) const { return this->addr[i]; }
	size_t	  count() const { return this->byte / sizeof(T); }
};

template <>
struct DataInfo<void> : DataInfoContainer {
	template <typename To>
	operator DataInfo<To>()
	{
		return *this;
	}
};
*/

using Count					   = unsigned long long int;
char const * const EXTENSION[] = {".row", ".ptr", ".col"};

#endif /* D5D78C16_48D8_4581_8C8D_E4336F18BDC9 */
