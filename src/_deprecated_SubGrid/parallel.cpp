#include "parallel.h"

#include <thread>

void parallelThread(size_t const workers, std::function<void(size_t const)> func)
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

void parallelFiber(size_t const workers, std::function<void(size_t const)> func)
{
	std::vector<fiber> wlist(workers);
	for (size_t i = 0; i < workers; i++) {
		wlist[i] = fiber([=] { func(i); });
	}

	for (size_t i = 0; i < workers; i++) {
		if (wlist[i].joinable()) {
			wlist[i].join();
		}
	}
}