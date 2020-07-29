#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"

#include <memory>
#include <regex>
#include <thread>
#include <vector>

std::string ShardIndex::string() const
{
	if (this->depth > 0) {
		return std::to_string(this->grid[0]) + "-" + std::to_string(this->grid[1]) + "," +
			   std::to_string(this->depth) + "," + std::to_string(this->shard[0]) + "-" +
			   std::to_string(this->shard[1]);
	} else {
		return std::to_string(this->grid[0]) + "-" + std::to_string(this->grid[1]);
	}
}

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
		} else {
		}

		if (m[5].length() > 0) {
			this->shard[1] = strtol(std::string(m[5]).c_str(), nullptr, 10);
		} else {
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

static bool checkAvail(std::array<ShardIndex, 3> const & in)
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

// ls | awk -F . '{print $1}' | awk -F , '{print $1}' | uniq | wc -l 를 하면 됨
static auto getGridSize(fs::path const & directory, fs::path const & extension)
{
	uint32_t   out = 0;
	std::regex regex("^(\\d*)-(\\d*)(?:,(\\d*),(\\d*)-(\\d*))?$");

	for (fs::recursive_directory_iterator curr(directory), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0) {
			// printf("%s", curr->path().c_str());
			ShardIndex sIdx;
			if (sIdx.parse(curr->path().stem().string()) && curr->path().extension() == extension) {
				// printf(" matched: (%d,%d),%d,(%d,%d)", sIdx.grid[0], sIdx.grid[1], sIdx.depth,
				// sIdx.shard[0], sIdx.shard[1]);
			}
			// printf("\n");

			out = std::max(sIdx.grid[0], out);
			out = std::max(sIdx.grid[1], out);
		}
		// printf("\n");
	}

	return out + 1;
}

std::shared_ptr<bchan<Command>> ScheduleManager(Context const & ctx)
{
	auto out = makeSp<bchan<Command>>(16);

	std::thread([=] {
		auto const MAXROW = getGridSize(ctx.folderPath, ".row");
		// printf("MAXROW: %d\n", MAXROW);

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					XY3 gidx = {{{i, col}, {row, col}, {row, i}}};
					// printf("(%d,%d),(%d,%d),(%d,%d)\n", gidx[0][0], gidx[0][1], gidx[1][0],
					// gidx[1][1], gidx[2][0], gidx[2][1]);

					auto findShards = [&ctx](XY const & idx) {
						std::vector<fs::path> out;

						std::regex regex("^" + std::to_string(idx[0]) + "-" +
										 std::to_string(idx[1]) + "(?:,\\d*,\\d*-\\d*)?$");

						for (fs::recursive_directory_iterator curr(ctx.folderPath), end;
							 curr != end;
							 ++curr) {
							// printf("FINDSHARD:%s\n", curr->path().c_str());
							if (fs::is_regular_file(curr->path()) &&
								fs::file_size(curr->path()) > 0 &&
								curr->path().extension() == ".row") {
								if (std::regex_search(curr->path().stem().string(), regex)) {
									out.push_back(curr->path());
								}
							}
						}

						return out;
					};

					for (auto & APath : findShards(gidx[0])) {
						ShardIndex A;
						A.parse(APath.stem().string());
						// printf("SHARD: %s\n", APath.c_str());
						for (auto & BPath : findShards(gidx[1])) {
							ShardIndex B;
							B.parse(BPath.stem().string());
							// printf("SHARD: %s, %s\n", APath.c_str(), BPath.c_str());
							for (auto & CPath : findShards(gidx[2])) {
								ShardIndex C;
								C.parse(CPath.stem().string());
								// printf("SHARD: %s, %s, %s\n", APath.c_str(), BPath.c_str(),
								// CPath.c_str());

								if (checkAvail(std::array<ShardIndex, 3>{A, B, C})) {
									/*
									printf("(%d,%d),%d,(%d,%d)",
										   A.grid[0],
										   A.grid[1],
										   A.depth,
										   A.shard[0],
										   A.shard[1]);
									printf("-(%d,%d),%d,(%d,%d)",
										   B.grid[0],
										   B.grid[1],
										   B.depth,
										   B.shard[0],
										   B.shard[1]);
									printf("-(%d,%d),%d,(%d,%d)\n",
										   C.grid[0],
										   C.grid[1],
										   C.depth,
										   C.shard[0],
										   C.shard[1]);

									auto ARange = convShardIndexToRange(A);
									auto BRange = convShardIndexToRange(B);
									auto CRange = convShardIndexToRange(C);
									increaseRangeDepth(ARange, 2);
									increaseRangeDepth(BRange, 2);
									increaseRangeDepth(CRange, 2);

									printf("[%d,%d),[%d,%d)",
										   ARange.range[0][0],
										   ARange.range[0][1],
										   ARange.range[1][0],
										   ARange.range[1][1]);
									printf(" - [%d,%d),[%d,%d)",
										   BRange.range[0][0],
										   BRange.range[0][1],
										   BRange.range[1][0],
										   BRange.range[1][1]);
									printf(" - [%d,%d),[%d,%d)\n",
										   CRange.range[0][0],
										   CRange.range[0][1],
										   CRange.range[1][0],
										   CRange.range[1][1]);
									*/
									// printf("%s %s %s\n", A.string().c_str(), B.string().c_str(),
									// C.string().c_str());
									out->push(Command{A.string(), B.string(), C.string()});
								}
							}
						}
					}
				}
			}
		}
		out->close();
	}).detach();

	return out;
}

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes)
{
	Count totalTriangles = 0;

	for (auto & res : *executionRes.get()) {
		fprintf(stdout,
				"%d;%s;%s;%s;%lld\n",
				res.deviceID,
				res.gidx[0].c_str(),
				res.gidx[1].c_str(),
				res.gidx[2].c_str(),
				res.triangle);
		totalTriangles += res.triangle;
	}

	fprintf(stdout, "total triangles: %lld\n", totalTriangles);
}