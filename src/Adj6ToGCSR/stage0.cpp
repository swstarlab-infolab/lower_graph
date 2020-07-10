#include "type.h"
#include "util.h"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_sort.h>
#include <thread>
#include <vector>

sp<std::vector<uint64_t>> stage0(fs::path const & inFolder,
								 fs::path const & outFolder,
								 uint64_t const	  maxVID,
								 uint64_t const	  reorderType)
{
	tbb::concurrent_vector<Reorder> temp(maxVID + 1);

	auto workers = std::thread::hardware_concurrency();
	parallelDo(workers, [&](size_t const idx) {
		for (auto i = idx; i < temp.size(); i += workers) {
			temp[i].key = uint64_t(i);
			temp[i].val = 0;
		}
	});

	stopwatch("Stage0, Count degree", [&] {
		auto fListChan = fileList(inFolder, "");
		parallelDo(8, [&](size_t const i) {
			for (auto & fPath : *fListChan) {
				auto adj6		 = fileLoad<uint8_t>(fPath);
				auto sRawDatChan = splitAdj6(adj6);

				parallelDo(64, [&](size_t const j) {
					for (auto & dat : *sRawDatChan) {
						auto s = dat.src;
						temp[s].val += dat.cnt;

						for (auto i = uint64_t(0); i < dat.cnt; i++) {
							auto d = be6_le8(&(adj6->at(dat.dstStart + i * 6)));

							if (s != d) {
								temp[d].val++;
							} else {
								temp[s].val--;
							}
						}
					}
				});
			}
		});
	});

	stopwatch("Stage0, Reorder vertices by rank", [&] {
		switch (reorderType) {
		case 1:
			tbb::parallel_sort(temp.begin(), temp.end(), [](Reorder const & l, Reorder const & r) {
				if (l.val == 0 && r.val != 0) {
					return false;
				} else if (l.val != 0 && r.val == 0) {
					return true;
				} else if (l.val == 0 && r.val == 0) {
					return false;
				} else {
					if (l.val == r.val) {
						return l.key < r.key;
					} else {
						return l.val < r.val;
					}
				}
			});
			break;
		case 2:
			tbb::parallel_sort(temp.begin(), temp.end(), [](Reorder const & l, Reorder const & r) {
				if (l.val == r.val) {
					return l.key > r.key;
				} else {
					return l.val > r.val;
				}
			});
			break;
		default:
			exit(EXIT_FAILURE);
		}

		auto workers = std::thread::hardware_concurrency();

		parallelDo(workers, [&](size_t const idx) {
			for (auto i = idx; i < temp.size(); i += workers) {
				temp[i].val = uint64_t(i);
			}
		});

		tbb::parallel_sort(temp.begin(), temp.end(), [](Reorder const & l, Reorder const & r) {
			return l.key < r.key;
		});
	});

	auto out = makeSp<std::vector<uint64_t>>(temp.size());

	for (size_t i = 0; i < temp.size(); i++) {
		out->at(i) = temp[i].val;
	}

	// for (auto & o : *out) { printf("%ld ", o); } printf("\n");

	return out;
}