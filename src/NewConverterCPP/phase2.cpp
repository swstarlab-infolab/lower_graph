#include "main.h"

#include <atomic>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <thread>

static auto dedup(std::shared_ptr<EdgeList32> in)
{
	// init
	auto out = std::make_shared<EdgeList32>(in->size());

	std::vector<std::atomic<uint32_t>> bitvec(size_t(ceil(double(in->size()) / 32.0)));

	auto getBit = [&bitvec](size_t const i) {
		return bool(bitvec[i / 32].load() & (1 << (i % 32)));
	};
	auto setBit = [&bitvec](size_t const i) { bitvec[i / 32].fetch_or(1 << (i % 32)); };

	std::vector<uint64_t> pSumRes(in->size() + 1);

	// prepare bit array
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, bitvec.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					bitvec[i] = 0;
				}
			}
		},
		tbb::auto_partitioner());

	// prepare exclusive sum array
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, pSumRes.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					pSumRes[i] = 0;
				}
			}
		},
		tbb::auto_partitioner());

	// sort
	tbb::parallel_sort(in->begin(), in->end(), [&](Edge32 const & l, Edge32 const & r) {
		return (l[0] < r[0]) || ((l[0] == r[0]) && (l[1] < r[1]));
	});

	// set bit 1 which is left != right (not the case: left == right)
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in->size() - 1),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto curr = grain + offset;
					auto next = grain + offset + 1;
					if (in->at(curr) != in->at(next)) {
						setBit(curr);
					}
				}
			}
		},
		tbb::auto_partitioner());
	setBit(in->size() - 1);

	// exclusive sum
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, in->size()),
		0,
		[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += (getBit(i) ? 1 : 0);
					if (isFinalScan) {
						pSumRes[i + 1] = temp;
					}
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	// count bit using parallel reduce
	size_t ones = tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, 32 * bitvec.size()),
		0,
		[&](tbb::blocked_range<size_t> const & r, size_t sum) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += (getBit(i) ? 1 : 0);
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	out->resize(ones);

	// reduce out vector
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in->size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					if (getBit(i)) {
						out->at(pSumRes[i]) = in->at(i);
					}
				}
			}
		},
		tbb::auto_partitioner());

	return out;
}

static void writeCSR(Context const & ctx, fs::path tempFilePath, std::shared_ptr<EdgeList32> in)
{
	std::vector<std::vector<Vertex32>> out(3);

	auto fillCol = std::thread([&, in] {
		// fill col
		out[2].resize(in->size());

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						out[2][i] = in->at(i)[1];
					}
				}
			},
			tbb::auto_partitioner());
	});

	auto fillRowAndPtr = std::thread([&, in] {
		// fill
		std::vector<std::atomic<uint32_t>> bitvec(size_t(ceil(double(in->size()) / 32.0)));

		auto getBit = [&bitvec](size_t const i) {
			return bool(bitvec[i / 32].load() & (1 << (i % 32)));
		};
		auto setBit = [&bitvec](size_t const i) { bitvec[i / 32].fetch_or(1 << (i % 32)); };

		std::vector<uint64_t> pSumRes(in->size() + 1);

		// prepare bit array
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, bitvec.size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						bitvec[i] = 0;
					}
				}
			},
			tbb::auto_partitioner());

		// prepare exclusive sum array
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, pSumRes.size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						pSumRes[i] = 0;
					}
				}
			},
			tbb::auto_partitioner());

		// set bit 1 which is left != right (not the case: left == right)
		tbb::parallel_for(
			tbb::blocked_range<size_t>(1, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto prev = grain + offset - 1;
						auto curr = grain + offset;
						if (in->at(prev)[0] != in->at(curr)[0]) {
							setBit(curr);
						}
					}
				}
			},
			tbb::auto_partitioner());
		setBit(0);

		// exclusive sum
		tbb::parallel_scan(
			tbb::blocked_range<size_t>(0, in->size()),
			0,
			[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
				auto temp = sum;
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;
						temp += (getBit(i) ? 1 : 0);
						if (isFinalScan) {
							pSumRes[i + 1] = temp;
						}
					}
				}
				return temp;
			},
			[&](size_t const & l, size_t const & r) { return l + r; },
			tbb::auto_partitioner());

		// count bit using parallel reduce
		size_t ones = tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, 32 * bitvec.size()),
			0,
			[&](tbb::blocked_range<size_t> const & r, size_t sum) {
				auto temp = sum;
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;
						temp += (getBit(i) ? 1 : 0);
					}
				}
				return temp;
			},
			[&](size_t const & l, size_t const & r) { return l + r; },
			tbb::auto_partitioner());

		out[0].resize(ones);
		out[1].resize(ones + 1);

		// fill row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						if (getBit(i)) {
							out[0][pSumRes[i]] = in->at(i)[0];
						}
					}
				}
			},
			tbb::auto_partitioner());

		// fill ptr
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						if (getBit(i)) {
							out[1][pSumRes[i]] = i;
						}
					}
				}
			},
			tbb::auto_partitioner());
		out[1].back() = in->size();
	});

	if (fillCol.joinable()) {
		fillCol.join();
	}

	if (fillRowAndPtr.joinable()) {
		fillRowAndPtr.join();
	}

	std::vector<std::thread> writeThreads(3);
	for (size_t i = 0; i < writeThreads.size(); i++) {
		writeThreads[i] = std::thread([&, i] {
			auto outFile = (ctx.outFolder / ctx.outName) /
						   fs::path(tempFilePath.stem().string() + __OutFileExts[i]);

			std::ofstream f(outFile, std::ios::binary | std::ios::out);

			f.write((char *)(out[i].data()), out[i].size() * sizeof(Vertex32));
			f.close();
		});
	}

	for (auto & t : writeThreads) {
		if (t.joinable()) {
			t.join();
		}
	}

	// fs::remove(tempFilePath);
}

void phase2(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		auto rawData = load<Edge32>(fpath);
		auto deduped = dedup(rawData);
		writeCSR(ctx, fpath, deduped);
		log("Phase 2 (EdgeList->CSR) " + fpath.string() + " Converted");
	};

	auto jobs = [&] {
		auto out = std::make_shared<bchan<fs::path>>(__ChannelSize);
		std::thread([&ctx, out] {
			auto fileList = walk(ctx.outFolder, __TempFileExt);
			for (auto & f : *fileList) {
				out->push(f);
			}
			out->close();
		}).detach();
		return out;
	}();

	std::vector<std::thread> threads(__WorkerCount);

	for (auto & t : threads) {
		t = std::thread([&ctx, jobs, fn] {
			for (auto & path : *jobs) {
				fn(path);
			}
		});
	}

	for (auto & t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
}