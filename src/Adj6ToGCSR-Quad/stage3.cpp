#include "stage.h"
#include "type.h"
#include "util.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <thread>

struct ShardIndex {
	std::array<V32, 2> grid, shard;
	uint32_t		   depth;
	std::string		   string() const;
	bool			   parse(std::string const & in);
};

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

static auto quad(sp<std::vector<E32>> in, uint32_t const gridWidth, size_t const currentDepth)
{
	// init
	// auto out = makeSp<std::vector<std::vector<E32>>>(in->size());

	auto out = makeSp<std::array<std::array<std::vector<E32>, 2>, 2>>();

	// sort only src of edge
	tbb::parallel_sort(
		in->begin(), in->end(), [&](E32 const & l, E32 const & r) { return (l[0] < r[0]); });

	auto currWidth = (gridWidth >> currentDepth);
	auto nextWidth = (gridWidth >> (currentDepth + 1));

	// printf("%d, %d\n", currWidth, nextWidth);

	size_t rowCut = in->size();

	// printf("QUAD A\n");
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in->size() - 1),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto curr = grain + offset;
					auto next = curr + 1;

					if ((in->at(curr)[0] / nextWidth) != (in->at(next)[0] / nextWidth)) {
						rowCut = next;
						// printf("%d / %d != %d / %d, rowCut: %d\n", in->at(curr)[0], nextWidth,
						// in->at(next)[0], nextWidth, rowCut);
						tbb::task::self().cancel_group_execution();
						return;
					}

					if (tbb::task::self().is_cancelled()) {
						return;
					}
				}
			}
		},
		tbb::auto_partitioner());

	// printf("rowCut = %d\n", rowCut);
	size_t colCut[2] = {rowCut, in->size()};
	// printf("QUAD B\n");

	parallelDo(2, [&](size_t const idx) {
		// sort only dst of edge

		auto range = tbb::blocked_range<size_t>(0, 1);
		switch (idx) {
		case 0:
			range = tbb::blocked_range<size_t>(0, rowCut - 1);
			tbb::parallel_sort(in->begin(),
							   in->begin() + rowCut,
							   [&](E32 const & l, E32 const & r) { return (l[1] < r[1]); });
			break;
		case 1:
			range = tbb::blocked_range<size_t>(rowCut, in->size() - 1);
			tbb::parallel_sort(in->begin() + rowCut, in->end(), [&](E32 const & l, E32 const & r) {
				return (l[1] < r[1]);
			});
			break;
		}

		tbb::parallel_for(
			range,
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto curr = grain + offset;
						auto next = curr + 1;

						if (((*in)[curr][1] / nextWidth) != ((*in)[next][1] / nextWidth)) {
							colCut[idx] = next;
							tbb::task::self().cancel_group_execution();
							return;
						}

						if (tbb::task::self().is_cancelled()) {
							return;
						}
					}
				}
			},
			tbb::auto_partitioner());
	});
	// printf("QUAD C\n");
	// printf("CUT: %ld, %ld, %ld, %ld, %ld\n", 0UL, colCut[0], rowCut, colCut[1], in->size());

	// sort
	parallelDo(4, [&](size_t const i) {
		switch (i) {
		case 0:
			(*out)[0][0].resize(colCut[0]);
			memcpy((*out)[0][0].data(), &((*in)[0]), (*out)[0][0].size() * sizeof(E32));
			break;
		case 1:
			(*out)[0][1].resize(rowCut - colCut[0]);
			memcpy((*out)[0][1].data(), &((*in)[colCut[0]]), (*out)[0][1].size() * sizeof(E32));
			break;
		case 2:
			(*out)[1][0].resize(colCut[1] - rowCut);
			memcpy((*out)[1][0].data(), &((*in)[rowCut]), (*out)[1][0].size() * sizeof(E32));
			break;
		case 3:
			(*out)[1][1].resize(in->size() - colCut[1]);
			memcpy((*out)[1][1].data(), &((*in)[colCut[1]]), (*out)[1][1].size() * sizeof(E32));
			break;
		}
		tbb::parallel_sort((*out)[i / 2][i % 2].begin(),
						   (*out)[i / 2][i % 2].end(),
						   [&](E32 const & l, E32 const & r) {
							   return (l[0] < r[0]) || ((l[0] == r[0]) && (l[1] < r[1]));
						   });
	});

	// printf("size\n" "%ld, %ld\n" "%ld, %ld\n", out->at(0).at(0).size(), out->at(0).at(1).size(),
	// out->at(1).at(0).size(), out->at(1).at(1).size());

	return out;
}

void stage3(fs::path const & outFolder, uint32_t const gridWidth, size_t const limitByte)
{
	bool exist = true;
	while (exist) {
		// consider that exists
		exist = false;

		auto jobs = [&] {
			auto out = makeSp<bchan<fs::path>>(128);
			std::thread([=, &exist] {
				auto fListChan = fileListOver(outFolder, ".el32", limitByte * 2);
				for (auto & f : *fListChan) {
					out->push(f);
					exist = true;
				}
				out->close();
			}).detach();
			return out;
		}();

		parallelDo(4, [&](size_t const i) {
			for (auto & fPath : *jobs) {
				stopwatch("Stage3, " + std::string(fPath), [&] {
					if (fs::file_size(fPath) == 0) {
						return;
					}
					// regex parse required
					auto	   rawData = fileLoad<E32>(fPath);
					ShardIndex sidx;
					sidx.parse(fPath.stem());
					// printf("sidx: (%d,%d),%d,(%d,%d)\n", sidx.grid[0], sidx.grid[1], sidx.depth,
					// sidx.shard[0], sidx.shard[1]);

					auto quaded = quad(rawData, gridWidth, sidx.depth);

					// printf("QUAD D\n");
					std::array<std::array<ShardIndex, 2>, 2> sidxNew;
					parallelDo(4, [&](size_t const idx) {
						auto r = idx / 2, c = idx % 2;
						sidxNew[r][c].grid	   = sidx.grid;
						sidxNew[r][c].depth	   = sidx.depth + 1;
						sidxNew[r][c].shard[0] = sidx.shard[0] * 2 + r;
						sidxNew[r][c].shard[1] = sidx.shard[1] * 2 + c;
					});
					// printf("QUAD E\n");

					parallelDo(4, [&](size_t const idx) {
						auto r = idx / 2, c = idx % 2;
						auto target =
							fPath.parent_path() / fs::path(sidxNew[r][c].string() + ".el32");
						if ((*quaded)[r][c].size() > 0) {
							fileSave(target,
									 (*quaded)[r][c].data(),
									 (*quaded)[r][c].size() * sizeof(E32));
						}
					});
					// printf("QUAD F\n");

					fs::remove(fPath);
				});
			}
		});
	}
}