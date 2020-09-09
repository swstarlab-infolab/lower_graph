#include "shard.h"

#include <regex>

std::string ShardIndex::string() const
{
	if (this->depth > 0) {
		return std::to_string(this->grid[0]) + "-" + std::to_string(this->grid[1]) + "," +
			   std::to_string(this->depth) + "," + std::to_string(this->shard[0]) + "-" +
			   std::to_string(this->shard[1]);
	} else {
		return std::to_string(this->grid[0]) + "-" + std::to_string(this->grid[1]);
	}
} // namespace std::stringShardIndex::string()const

bool ShardIndex::parse(std::string const & in)
{
	*this = {
		0,
	};

	std::regex	regex("^(\\d*)-(\\d*)(?:,(\\d*),(\\d*)-(\\d*))?$");
	std::smatch m;

	if (std::regex_match(in, m, regex)) {
		if (m[1].length() > 0) {
			this->grid[0] = strtol(std::string(m[1]).c_str(), nullptr, 10);
		}

		if (m[2].length() > 0) {
			this->grid[1] = strtol(std::string(m[2]).c_str(), nullptr, 10);
		}

		if (m[3].length() > 0) {
			this->depth = strtol(std::string(m[3]).c_str(), nullptr, 10);
		} else {
			return true;
		}

		if (m[4].length() > 0) {
			this->shard[0] = strtol(std::string(m[4]).c_str(), nullptr, 10);
		}

		if (m[5].length() > 0) {
			this->shard[1] = strtol(std::string(m[5]).c_str(), nullptr, 10);
		}

		return true;
	} else {
		return false;
	}
}

void ShardRange::conv(ShardIndex const & in)
{
	// mnemonics for easy coding
	constexpr auto s = 0, t = 1; // start, terminate

	this->depth = in.depth;
	for (size_t i = 0; i < this->range.size(); i++) {
		this->range[i][s] = in.grid[i] * (1 << in.depth) + in.shard[i];
		this->range[i][t] = this->range[i][s] + 1;
	}
}

bool ShardRange::increase(uint32_t const depth)
{
	auto diff = depth - this->depth;

	if (diff > 0) {
		for (auto & i : this->range) {
			for (auto & elem : i) {
				elem *= (1 << diff);
			}
		}
		this->depth = depth;
		return true;
	} else if (diff == 0) {
		return true;
	} else {
		return false;
	}
}