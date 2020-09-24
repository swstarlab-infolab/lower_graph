#ifndef AFEC62DB_FAEC_416F_8494_CDCA01D89CC1
#define AFEC62DB_FAEC_416F_8494_CDCA01D89CC1

#include "util.h"

#include <atomic>
#include <bitset>
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <vector>

template <typename T>
void fastStreamCompactionTBB(std::vector<T> const &			in,
							 std::vector<T> &				out,
							 std::function<bool(T const &)> condition_func)
{
	std::vector<std::atomic<uint32_t>> bv(ceil(in.size(), 32));
	std::vector<uint32_t>			   prefix(bv.size() + 1);

	auto getBit = [&bv](size_t const i) {
		return bool(bv[i >> 5].load() & (1UL << (i & (32 - 1))));
	};
	auto setBit = [&bv](size_t const i) { bv[i >> 5].fetch_or(1 << (i & (32 - 1))); };

	// initialize
	tbb::parallel_for_each(bv.begin(), bv.end(), [](std::atomic<uint32_t> & b) { b.store(0L); });

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, prefix.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					prefix[pos + off] = 0L;
				}
			}
		},
		tbb::auto_partitioner());

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					auto now = pos + off;
					if (condition_func(in[now])) {
						setBit(now);
					}
				}
			}
		},
		tbb::auto_partitioner());

	for (int i = 0; i < bv.size(); i++) {
		for (int j = 32 - 1; j >= 0; j--) {
			printf("%d", getBit(32 * i + j));
		}
		printf(" ");
	}
	printf("\n");

	// saving
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, bv.size()),
		0L,
		[&](tbb::blocked_range<size_t> const & r, uint32_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0UL; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += std::bitset<32>(bv[i]).count();
					if (isFinalScan) {
						prefix[i + 1] = temp;
					}
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	out.resize(prefix.back());

	if (out.size() == 0) {
		return;
	}

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, bv.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					auto now   = pos + off;
					auto bvnow = bv[now].load();
					for (int i = 0, j = 0; i < 32; i++) {
						if (bvnow & (1UL << i)) {
							out[prefix[now] + j] = in[32 * now + i];
							j++;
						}
					}
				}
			}
		},
		tbb::auto_partitioner());
}

template <typename T>
void fastStreamCompactionTBB64(std::vector<T> const &		  in,
							   std::vector<T> &				  out,
							   std::function<bool(T const &)> condition_func)
{
	std::vector<std::atomic<uint64_t>> bv(ceil(in.size(), 64));
	std::vector<uint64_t>			   prefix(bv.size() + 1);

	auto getBit = [&bv](size_t const i) {
		return bool(bv[i >> 6].load() & (1UL << (i & (64 - 1))));
	};
	auto setBit = [&bv](size_t const i) { bv[i >> 6].fetch_or(1UL << (i & (64 - 1))); };

	// initialize
	tbb::parallel_for_each(bv.begin(), bv.end(), [](std::atomic<uint64_t> & b) { b.store(0L); });

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, prefix.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					prefix[pos + off] = 0L;
				}
			}
		},
		tbb::auto_partitioner());

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					auto now = pos + off;
					if (condition_func(in[now])) {
						setBit(now);
					}
				}
			}
		},
		tbb::auto_partitioner());

	for (int i = 0; i < bv.size(); i++) {
		for (int j = 64 - 1; j >= 0; j--) {
			printf("%d", getBit(64 * i + j));
		}
		printf(" ");
	}
	printf("\n");

	// saving
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, bv.size()),
		0L,
		[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0UL; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += std::bitset<64>(bv[i]).count();
					if (isFinalScan) {
						prefix[i + 1] = temp;
					}
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	out.resize(prefix.back());

	if (out.size() == 0) {
		return;
	}

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, bv.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (auto pos = r.begin(); pos < r.end(); pos += r.grainsize()) {
				for (auto off = 0UL; off < r.grainsize(); off++) {
					auto now   = pos + off;
					auto bvnow = bv[now].load();
					for (int i = 0, j = 0; i < 64; i++) {
						if (bvnow & (1UL << i)) {
							out[prefix[now] + j] = in[64 * now + i];
							j++;
						}
					}
				}
			}
		},
		tbb::auto_partitioner());
}

#endif /* AFEC62DB_FAEC_416F_8494_CDCA01D89CC1 */