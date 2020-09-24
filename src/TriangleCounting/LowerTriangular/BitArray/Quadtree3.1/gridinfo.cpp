#include "gridinfo.h"

void GridInfo::init(fs::path const & folderPath)
{
	this->matrix.resize(1);
	this->matrix.front().resize(1);
	this->hashmap.reserve(1L << 12);

	// Write all informations of grids in the folder
	int gridID = 0;

	for (fs::recursive_directory_iterator curr(folderPath), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0 &&
			curr->path().extension() == EXTENSION[0]) {
			auto stem = curr->path().stem().string();

			std::array<fs::path, 3> path;
			std::array<size_t, 3>	byte;
			for (int i = 0; i < 3; i++) {
				path[i] = folderPath / fs::path(stem + EXTENSION[i]);
				byte[i] = fs::file_size(path[i]);
			}

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
			value.byte	= byte;
			value.path	= path;

			if (this->matrix.size() <= value.grid[0] ||
				this->matrix.front().size() <= value.grid[1]) {

				this->matrix.resize(value.grid[0] + 1);
				for (auto & row : this->matrix) {
					row.resize(value.grid[1] + 1);
				}
			}

			this->matrix[value.grid[0]][value.grid[1]].push_back(value);
			this->hashmap[value.id] = &(this->matrix[value.grid[0]][value.grid[1]].back());

			gridID++;
		}
	}
}