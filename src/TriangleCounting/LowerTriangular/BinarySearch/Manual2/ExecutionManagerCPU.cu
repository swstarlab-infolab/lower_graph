#include "ExecutionManager.cuh"

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/paralle_for.h>
#include <tbb/parallel_scan.h>

static void intersection(MemInfo<Vertex> & longArr, MemInfo<Vertex> & shortArr, Count * count)
{
	auto const ma;
}

static void kernel(Grids const g, Lookup const * lookup0, Lookup const * lookup2, Count * count)
{
	Count myCount = 0;

	for (uint32_t G1RowIdx = 0; G1RowIdx < g[1][0].count(); G1RowIdx++) {
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
				for (uint32_t G0ColIdx = G0ColIdxStart; G0ColIdx < G0ColIdxEnd; G0ColIdx++) {
					intersection(g[1][2][G1ColIdxStart], g[0][2][G0ColIdx], &myCount);
				}
			} else {
				for (uint32_t G1ColIdx = G1ColIdxStart; G1ColIdx < G1ColIdxEnd; G1ColIdx++) {
					intersection(g[0][2][G0ColIdxStart], g[1][2][G1ColIdx], &myCount);
				}
			}
		}
	}

	return myCount;
	/*
	Count myCount = 0;

	for (uint32_t G1RowIdx = 0; G1RowIdx < g[1][0].count(); G1RowIdx++) {
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
				for (uint32_t G0ColIdx = G0ColIdxStart; G0ColIdx < G0ColIdxEnd; G0ColIdx++) {
					intersection(g[1][2][G1ColIdxStart], g[0][2][G0ColIdx], &myCount);
				}
			} else {
				for (uint32_t G1ColIdx = G1ColIdxStart; G1ColIdx < G1ColIdxEnd; G1ColIdx++) {
					intersection(g[0][2][G0ColIdxStart], g[1][2][G1ColIdx], &myCount);
				}
			}
		}
	}

	return myCount;
	*/
}

static auto exclusiveSum(MemInfo<Vertex> & in, MemInfo<Vertex> & out)
{
	// exclusive sum
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, in->size()),
		0,
		[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += in[i];
					if (isFinalScan) {
						pSumRes[i + 1] = temp;
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

static auto resetLookupTemp(MemInfo<Vertex> & specimen, MemInfo<Vertex> & target)
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
	auto & myCtx   = ctx.executionManagerCtx[myID];
	auto & blocks  = ctx.setting[1];
	auto & threads = ctx.setting[2];

	Count cnt = 0;

	memset(myCtx.count.ptr, 0, myCtx.count.byte);

	genLookupTemp(G[0], myCtx.lookup.temp.ptr);
	exclusiveSum(myCtx.lookup.temp, myCtx.lookup.G0);
	resetLookupTemp(G[0], myCtx.lookup.temp.ptr);

	genLookupTemp(G[2], myCtx.lookup.temp.ptr);
	exclusiveSum(myCtx.lookup.temp, myCtx.lookup.G2);
	resetLookupTemp(G[2], myCtx.lookup.temp.ptr);

	kernel(G, myCtx.lookup.G0.ptr, myCtx.lookup.G2.ptr, myCtx.count.ptr);

	return cnt;
}