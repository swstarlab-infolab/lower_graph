#ifndef A48898B6_6582_420F_A104_E7122EEC2718
#define A48898B6_6582_420F_A104_E7122EEC2718

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "type.h"

#include <string>

static auto filenameEncode(GridIndex32 in)
{
	return std::to_string(in[0]) + "-" + std::to_string(in[1]);
}

static auto filenameDecode(std::string const & in)
{
	GridIndex32 gidx32 = {0, 0};

	auto delimPos = in.find("-");
	gidx32[0]	  = atoi(in.substr(0, delimPos).c_str());
	gidx32[1]	  = atoi(in.substr(delimPos + 1, in.size()).c_str());

	return gidx32;
}
#endif /* A48898B6_6582_420F_A104_E7122EEC2718 */
