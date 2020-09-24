#ifndef F51F806D_BC4D_4019_AA22_CA25D217A0B0
#define F51F806D_BC4D_4019_AA22_CA25D217A0B0

#include "my_mysql.h"

#include <functional>
#include <memory>
#include <string>

inline size_t s2l(std::string const & in) { return strtol(in.c_str(), nullptr, 10); }
inline size_t s2l(mysqlpp::String const & in) { return strtol(in.c_str(), nullptr, 10); }

size_t ceil(size_t const x, size_t const y);
void   parallelDo(size_t const workers, std::function<void(size_t const)> func);

template <typename... Args>
std::string sprn(const std::string & format, Args... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size <= 0) {
		throw std::runtime_error("Error during formatting.");
	}
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

#endif /* F51F806D_BC4D_4019_AA22_CA25D217A0B0 */
