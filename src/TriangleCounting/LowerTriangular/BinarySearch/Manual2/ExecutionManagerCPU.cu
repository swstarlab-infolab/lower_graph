#include "ExecutionManager.cuh"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
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

static Count intersection(Vertex const * Arr, uint32_t const ArrLen, Vertex const candidate)
{
	auto const maxLevel = ulog2floor(ArrLen);

	int now = (ArrLen - 1) >> 1;

	for (uint32_t level = 0; level <= maxLevel; level++) {
		auto const movement = 1 << (maxLevel - level - 1);

		if (now < 0) {
			now += movement;
		} else if (ArrLen <= now) {
			now -= movement;
		} else {
			if (Arr[now] < candidate) {
				now += movement;
			} else if (candidate < Arr[now]) {
				now -= movement;
			} else {
				return 1;
			}
		}
	}

	return 0;
}

static auto kernel(Grids const g, Lookup const * lookup0, Lookup const * lookup2, Count * count)
{

	Count totalResult = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, g[1][0].count()),
		0,
		[&](tbb::blocked_range<uint32_t> const & r, Count sum) {
			Count myCount = sum;

			for (auto G1RowIdx = r.begin(); G1RowIdx != r.end(); G1RowIdx++) {
				auto const G1Row		 = g[1][0][G1RowIdx];
				auto const G1ColIdxStart = g[1][1][G1RowIdx];
				auto const G1ColIdxEnd	 = g[1][1][G1RowIdx + 1];
				auto const G1ColLen		 = G1ColIdxEnd - G1ColIdxStart;

				if (G1ColLen == 0) {
					continue;
				}

				auto const G2ColIdxStart = lookup0[G1Row];
				auto const G2ColIdxEnd	 = lookup0[G1Row + 1];

				for (uint32_t G2ColIdx = G2ColIdxStart; G2ColIdx < G2ColIdxEnd; G2ColIdx++) {
					auto const G2Col = g[2][2][G2ColIdx];

					auto const G0ColIdxStart = lookup0[G2Col];
					auto const G0ColIdxEnd	 = lookup0[G2Col + 1];
					auto const G0ColLen		 = G0ColIdxEnd - G0ColIdxStart;

					if (G0ColLen == 0) {
						continue;
					}

					if (G1ColLen >= G0ColLen) {
						for (uint32_t G0ColIdx = G0ColIdxStart; G0ColIdx < G0ColIdxEnd;
							 G0ColIdx++) {
							myCount +=
								intersection(&g[1][2][G1ColIdxStart], G1ColLen, g[0][2][G0ColIdx]);
						}
					} else {
						for (uint32_t G1ColIdx = G1ColIdxStart; G1ColIdx < G1ColIdxEnd;
							 G1ColIdx++) {
							myCount +=
								intersection(&g[0][2][G0ColIdxStart], G0ColLen, g[1][2][G1ColIdx]);
						}
					}
				}
			}

			return myCount;
		},
		[&](Count const & l, Count const & r) { return l + r; },
		tbb::auto_partitioner());

	return totalResult;
}

static auto exclusiveSum(MemInfo<Vertex> & in, MemInfo<Vertex> & out)
{
	// exclusive sum
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, in.count()),
		0,
		[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += in[i];
					if (isFinalScan) {
						out[i + 1] = temp;
					}
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());
}

static auto genLookupTemp(Grid & g, MemInfo<Vertex> & luTemp)
{
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, g[0].count()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i			= grain + offset;
					luTemp[g[0][i]] = g[1][i + 1] - g[1][i];
				}
			}
		},
		tbb::auto_partitioner());
}

static auto resetLookupTemp(Grid & g, MemInfo<Vertex> & luTemp)
{
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, g[0].count()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i			= grain + offset;
					luTemp[g[0][i]] = 0;
				}
			}
		},
		tbb::auto_partitioner());
}

Count launchKernelCPU(Context & ctx, DeviceID myID, Grids & G)
{
	printf("start launchKernelCPU\n");
	auto & myCtx   = ctx.executionManagerCtx[myID];
	auto & blocks  = ctx.setting[1];
	auto & threads = ctx.setting[2];

	genLookupTemp(G[0], myCtx.lookup.temp);
	printf("genLookupTemp Done\n");
	exclusiveSum(myCtx.lookup.temp, myCtx.lookup.G0);
	printf("exclusive Sum Done\n");
	resetLookupTemp(G[0], myCtx.lookup.temp);
	printf("resetLookupTemp Done\n");

	genLookupTemp(G[2], myCtx.lookup.temp);
	printf("genLookupTemp Done\n");
	exclusiveSum(myCtx.lookup.temp, myCtx.lookup.G2);
	printf("exclusive Sum Done\n");
	resetLookupTemp(G[2], myCtx.lookup.temp);
	printf("resetLookupTemp Done\n");

	return kernel(G, myCtx.lookup.G0.ptr, myCtx.lookup.G2.ptr, myCtx.count.ptr);
}