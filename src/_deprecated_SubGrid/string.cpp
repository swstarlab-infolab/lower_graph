#include "string.h"

std::shared_ptr<std::vector<std::string>> split(std::string const & in, std::string const & delim)
{
	auto out = std::make_shared<std::vector<std::string>>();
	for (size_t pos = 0; pos < in.length();) {
		size_t found = in.find(delim, pos);
		if (found == std::string::npos) {
			found = in.length();
		}
		out->push_back(in.substr(pos, found - pos));
		pos = found + delim.length();
	}
	return out;
}

std::string unit(size_t const byte)
{
	std::stringstream ss;
	ss << std::fixed << std::showpoint << std::setprecision(3);
	if (byte < (1L << 10)) {
		ss << std::to_string(byte) << " Byte";
		return ss.str();
	} else if (byte < (1L << 20)) {
		ss << double(byte) / double(1L << 10) << " KiB";
	} else if (byte < (1L << 30)) {
		ss << double(byte) / double(1L << 20) << " MiB";
	} else if (byte < (1L << 40)) {
		ss << double(byte) / double(1L << 30) << " GiB";
	} else if (byte < (1L << 50)) {
		ss << double(byte) / double(1L << 40) << " TiB";
	} else {
		ss << double(byte) / double(1L << 50) << " PiB";
	}
	return ss.str();
}