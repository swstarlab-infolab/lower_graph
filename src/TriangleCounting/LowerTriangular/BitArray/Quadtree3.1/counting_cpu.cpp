#include "base/type.h"
#include "counting.h"

#include <array>
#include <tbb/parallel_scan.h>

static uint32_t ulog2floor(uint32_t x)
{
	uint32_t r, q;
	r = (x > 0xFFFF) << 4;
	x >>= r;
	q = (x > 0xFF) << 3;
	x >>= q;
	r |= q;
	q = (x > 0xF) << 2;
	x >>= q;
	r |= q;
	q = (x > 0x3) << 1;
	x >>= q;
	r |= q;

	return (r | (x >> 1));
}

// Binary Search Intersection
static void bsInter(uint32_t const *	 arr,
					uint32_t const		 arrLen,
					uint32_t const		 candidate,
					unsigned long long * count)
{
	auto const maxLevel = ulog2floor(arrLen);

	int now = (arrLen - 1) >> 1;

	for (uint32_t level = 0; level <= maxLevel; level++) {
		auto const movement = 1 << (maxLevel - level - 1);

		if (now < 0) {
			now += movement;
		} else if ((int)arrLen <= now) {
			now -= movement;
		} else {
			if (arr[now] < candidate) {
				now += movement;
			} else if (candidate < arr[now]) {
				now -= movement;
			} else {
				(*count)++;
				break;
			}
		}
	}
}

// Binary Search Position
static ssize_t bsPos(uint32_t const * arr, uint32_t const arrLen, uint32_t const candidate)
{
	// auto const maxLevel = uint32_t(ceil(log2(arrLen + 1))) - 1;
	// ceil(log2(a)) == floor(log2(a-1))+1
	auto const maxLevel = ulog2floor(arrLen);

	int now = (arrLen - 1) >> 1;

	for (uint32_t level = 0; level <= maxLevel; level++) {
		auto const movement = 1 << (maxLevel - level - 1);

		if (now < 0) {
			now += movement;
		} else if ((int)arrLen <= now) {
			now -= movement;
		} else {
			if (arr[now] < candidate) {
				now += movement;
			} else if (candidate < arr[now]) {
				now -= movement;
			} else {
				return now;
			}
		}
	}

	return -1;
}

static void genLookupTemp(uint32_t const tid, uint32_t const ts, Grid const & G, Lookup & L)
{
	for (uint32_t i = tid; i < G[0].count(); i += ts) {
		L[G[0][i]] = G[1][i + 1] - G[1][i];
	}
}

static void resetLookupTemp(uint32_t const tid, uint32_t const ts, Grid const & G, Lookup & L)
{
	for (uint32_t i = tid; i < G[0].count(); i += ts) {
		L[G[0][i]] = 0;
	}
}

static Count kernel(uint32_t const tid, uint32_t const ts, Grids const & Gs, Lookups const & Ls)
{
	Count mycount = 0UL;

	for (uint32_t g1row_iter = tid; g1row_iter < Gs[1][0].count(); g1row_iter += ts) {

		// This makes huge difference!!!
		// Without "Existing Row" information: loop all 2^24 and check it all
		// With "Existing Row" information: extremely faster than without-version
		auto const g1row	   = Gs[1][0][g1row_iter];
		auto const g2col_idx_s = Ls[2][g1row], g2col_idx_e = Ls[2][g1row + 1];

		if (g2col_idx_s == g2col_idx_e) {
			continue;
		}

		auto const g1col_idx_s = Gs[1][1][g1row_iter];
		auto const g1col_idx_e = Gs[1][1][g1row_iter + 1];

		// variable for binary tree intersection
		auto const g1col_length = g1col_idx_e - g1col_idx_s;
		mycount++;

		for (uint32_t g2col_idx = g2col_idx_s; g2col_idx < g2col_idx_e; g2col_idx++) {
			auto const g2col = Gs[2][2][g2col_idx];

			auto const g0col_idx_s = Ls[0][g2col], g0col_idx_e = Ls[0][g2col + 1];
			if (g0col_idx_s == g0col_idx_e) {
				continue;
			}

			auto const g0col_length = g0col_idx_e - g0col_idx_s;

			if (g1col_length >= g0col_length) {
				for (uint32_t g0col_idx = 0; g0col_idx < g0col_idx_e; g0col_idx++) {
					// bsInter(&Gs[1][2][g1col_idx_s], g1col_length, Gs[0][2][g0col_idx], &mycount);
				}
			} else {
				for (uint32_t g1col_idx = 0; g1col_idx < g1col_idx_e; g1col_idx++) {
					// bsInter(&Gs[0][2][g0col_idx_s], g0col_length, Gs[1][2][g1col_idx], &mycount);
				}
			}
		}
	}

	return mycount;
}

#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>
#include <thread>

Count countingCPU(Grids const & Gs, Lookups & Ls)
{

	std::vector<std::thread> T(std::thread::hardware_concurrency());
	std::vector<Count>		 R(T.size());
	for (size_t t = 0; t < T.size(); t++) {
		T[t] = std::thread([&, t] {
			Count myCount = 0;

			auto fill = [&](Grid const & G, Lookup & Ltemp, Lookup & Lout) {
				genLookupTemp(t, T.size(), G, Lout);

				tbb::parallel_scan(
					tbb::blocked_range<size_t>(0, Ltemp.size()),
					0,
					[&](tbb::blocked_range<size_t> const & r, uint32_t sum, bool isFinalScan) {
						auto temp = sum;
						for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
							for (size_t offset = 0UL; offset < r.grainsize(); offset++) {
								auto i = grain + offset;
								temp += Ltemp[i];
								if (isFinalScan) {
									Lout[i + 1] = temp;
								}
							}
						}
						return temp;
					},
					[&](size_t const & l, size_t const & r) { return l + r; },
					tbb::auto_partitioner());

				resetLookupTemp(t, T.size(), G, Lout);
			};

			// fill Ls[0]
			fill(Gs[0], Ls[1], Ls[0]);

			// fill Ls[2]
			fill(Gs[2], Ls[1], Ls[2]);

			// run
			R[t] = kernel(t, T.size(), Gs, Ls);
		});
	}

	for (auto & t : T) {
		if (t.joinable()) {
			t.join();
		}
	}

	return std::accumulate(R.begin(), R.end(), 0UL);
}