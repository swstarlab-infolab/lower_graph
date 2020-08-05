#include "util.h"

#include <thread>

size_t ceil(size_t const x, size_t const y) { return (x != 0) ? (1 + ((x - 1) / y)) : 0; }

void pThread(size_t const workers, std::function<void(size_t const)> func)
{
	/*
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0L, workers, 1L),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t i = r.begin(); i != r.end(); i++) {
				func(i);
			}
		},
		tbb::static_partitioner());
		*/
	std::vector<std::thread> wlist(workers);
	for (size_t i = 0; i < workers; i++) {
		wlist[i] = std::thread([=] { func(i); });
	}

	for (size_t i = 0; i < workers; i++) {
		if (wlist[i].joinable()) {
			wlist[i].join();
		}
	}
}