#ifndef D0A9018F_9699_40C9_A354_DBD2CFAD848A
#define D0A9018F_9699_40C9_A354_DBD2CFAD848A

#include "base/shard.h"
#include "base/type.h"

#include <array>
#include <string.h>
#include <unordered_map>
#include <vector>

struct GridInfoValue {
	uint32_t							 id;
	std::array<uint32_t, 2>				 grid;
	uint32_t							 depth;
	std::array<uint32_t, 2>				 shard;
	std::array<std::array<size_t, 2>, 2> range;
	std::array<size_t, 3>				 byte;
	std::array<std::string, 3>			 path;

	GridInfoValue() { memset((void *)this, 0x00, sizeof(*this)); }

	GridInfoValue(GridInfoValue const & copy)
	{
		this->id	= copy.id;
		this->grid	= copy.grid;
		this->depth = copy.depth;
		this->shard = copy.shard;
		this->range = copy.range;
		this->byte	= copy.byte;
		this->path	= copy.path;
	}
};

// using GridInfo = std::vector<GridInfoValue>;

struct GridInfo {
	std::vector<std::vector<std::vector<GridInfoValue>>> matrix;
	std::unordered_map<uint32_t, GridInfoValue *>		 hashmap;

	std::vector<GridInfoValue> & xy(uint32_t const row, uint32_t const col)
	{
		return this->matrix[row][col];
	}

	std::vector<GridInfoValue> const & xy(uint32_t const row, uint32_t const col) const
	{
		return this->matrix[row][col];
	}

	std::vector<GridInfoValue> & xy(std::array<uint32_t, 2> const & coord)
	{
		return this->matrix[coord[0]][coord[1]];
	}

	std::vector<GridInfoValue> const & xy(std::array<uint32_t, 2> const & coord) const
	{
		return this->matrix[coord[0]][coord[1]];
	}

	GridInfoValue &		  id(uint32_t const id) { return *this->hashmap[id]; }
	GridInfoValue const & id(uint32_t const id) const { return *(this->hashmap.at(id)); }

	void init(fs::path const & folderPath);
};

#endif /* D0A9018F_9699_40C9_A354_DBD2CFAD848A */
