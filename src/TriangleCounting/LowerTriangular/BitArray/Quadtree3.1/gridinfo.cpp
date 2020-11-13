#include "gridinfo.h"

#include <algorithm>

void GridInfo::init(fs::path const & folderPath)
{
	uint32_t SIZEMAX = 0;
	for (fs::recursive_directory_iterator curr(folderPath), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0 &&
			curr->path().extension() == EXTENSION[0]) {
			auto stem = curr->path().stem().string();

			ShardIndex sIdx;
			sIdx.parse(stem);
			SIZEMAX = std::max(std::max(sIdx.grid[0], sIdx.grid[1]), SIZEMAX);
		}
	}
	SIZEMAX++;

	this->matrix.resize(SIZEMAX);
	for (auto & row : this->matrix) {
		row.resize(SIZEMAX);
	}

	uint32_t gridID = 0;

	for (fs::recursive_directory_iterator curr(folderPath), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0 &&
			curr->path().extension() == EXTENSION[0]) {
			auto stem = curr->path().stem().string();

			ShardIndex sIdx;
			sIdx.parse(stem);

			ShardRange sRange;
			sRange.conv(sIdx);
			sRange.increase(24);

			GridInfoValue value;
			value.id	= gridID;
			value.grid	= sIdx.grid;
			value.depth = sIdx.depth;
			value.shard = sIdx.shard;
			value.range = sRange.range;

			for (int i = 0; i < 3; i++) {
				value.path[i] = std::string(folderPath / fs::path(stem + EXTENSION[i]));
				value.byte[i] = fs::file_size(value.path[i]);
			}

			this->matrix[value.grid[0]][value.grid[1]].push_back(value);

			gridID++;
		}
	}

	for (auto & row : this->matrix) {
		for (auto & eachlist : row) {
			for (auto & each : eachlist) {
				this->hashmap[each.id] = &each;
			}
		}
	}
}