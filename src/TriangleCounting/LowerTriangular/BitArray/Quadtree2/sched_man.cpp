#include "sched_man.h"

#include <regex>
#include <thread>

namespace Sched
{
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

bool Manager::checkAvail(JobType const & in)
{
	uint32_t maxDepth = 0;
	for (size_t i = 0; i < in.size(); ++i) {
		maxDepth = std::max(in[i].depth, maxDepth);
	}

	std::array<ShardRange, 3> range;
	for (size_t i = 0; i < range.size(); ++i) {
		range[i].conv(in[i]);
		range[i].increase(maxDepth);
	}

	// range check
	auto IS = [](XY const & A, XY const & B) {
		constexpr auto s = 0, t = 1; // start, terminate
		return std::max(A[s], B[s]) < std::min(A[t], B[t]);
	};

	constexpr auto x = 0, y = 1; // start, terminate

	auto & A = range[0].range;
	auto & B = range[1].range;
	auto & C = range[2].range;

	return IS(A[y], B[y]) && IS(B[x], C[x]) && IS(A[x], C[y]);
}

void Manager::run()
{
	std::thread([=] {
		uint32_t MAXROW = 0;
		for (auto & i : this->entry) {
			MAXROW = std::max(i.first[0], MAXROW);
			MAXROW = std::max(i.first[1], MAXROW);
		}
		MAXROW++;

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					XY3 gidx = {{{i, col}, {row, col}, {row, i}}};

					if (this->entry.find(gidx[0]) == this->entry.end()) {
						continue;
					}
					if (this->entry.find(gidx[1]) == this->entry.end()) {
						continue;
					}
					if (this->entry.find(gidx[2]) == this->entry.end()) {
						continue;
					}

					for (auto & A : this->entry[gidx[0]]) {
						for (auto & B : this->entry[gidx[1]]) {
							for (auto & C : this->entry[gidx[2]]) {
								if (checkAvail(JobType{A, B, C})) {
									// printf("Sched::Manager: %s / %s / %s\n", A.string().c_str(),
									// B.string().c_str(), C.string().c_str());
									this->out->push({A, B, C});
								}
							}
						}
					}
				}
			}
		}

		this->out->close();
	}).detach();
}

Manager::Manager(fs::path const & folder) : folderPath(folder)
{

	for (fs::recursive_directory_iterator curr(this->folderPath), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0 &&
			curr->path().extension() == ".row") {
			auto stem = curr->path().stem().string();

			ShardIndex sIdx;
			sIdx.parse(stem);

			if (this->entry.find(sIdx.grid) != this->entry.end()) {
				this->entry.at(sIdx.grid).push_back(sIdx);
			} else {
				this->entry.insert({sIdx.grid, std::vector<ShardIndex>{{sIdx}}});
			}
		}
	}

	this->out = std::make_shared<boost::fibers::buffered_channel<JobType>>(1 << 8);
	printf("Constructor: Sched::Manager, Init Complete\n");
}

Manager::~Manager() { printf("Destructor: Sched::Manager, No error\n"); }
} // namespace Sched