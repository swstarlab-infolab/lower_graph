#include "util.cuh"

fs::path csrPath(fs::path const & folder, GridIndex const & gidx, DataType const dataType)
{
	if (gidx.shard == -1) {
		return fs::path(std::string(folder) + std::to_string(gidx.xy[0]) + "-" +
						std::to_string(gidx.xy[1]) + extension[dataType]);
	} else {
		return fs::path(std::string(folder) + std::to_string(gidx.xy[0]) + "-" +
						std::to_string(gidx.xy[1]) + "," + std::to_string(gidx.shard) +
						extension[dataType]);
	}
}

/*
void parseFilename(fs::path const & stem)
{
	auto & in		= stem.string();
	auto   commaPos = in.find(",");
	in.substr(0L, commaPos);

	strtol(in.substr(commaPos, in.length()), nullptr, 10);
}
*/

bool csrExist(fs::path const & folder, fs::path const & stem)
{
	for (int i = 0; i < 3; i++) {
		if (!fs::exists(fs::path((folder / stem).string() + extension[i]))) {
			return false;
		}
	}
	return true;
}

#include <iostream>

size_t csrShardCount(fs::path const & folder, std::string const & target)
{
	size_t	   count = 0;
	std::regex rgx("^" + target + "(,\\d*)?\\" + extension[0]);

	for (fs::recursive_directory_iterator iter{folder}, end; iter != end; ++iter) {
		const std::string candidate =
			iter->path().stem().string() + iter->path().extension().string();
		if (fs::is_regular_file(*iter) && std::regex_search(candidate, rgx)) {
			count++;
		}
	}
	return count;
}
