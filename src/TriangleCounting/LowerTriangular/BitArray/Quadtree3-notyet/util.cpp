#include "util.h"

#include <thread>
#include <vector>

size_t ceil(size_t const x, size_t const y) { return (x != 0) ? (1 + ((x - 1) / y)) : 0; }

void pThread(size_t const workers, std::function<void(size_t const)> func)
{
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