#include "context.h"
#include "util.h"

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <vector>

int main(int argc, char * argv[])
{
	ctx.init(argc, argv);

	auto start = std::chrono::system_clock::now();

	for (auto & m : ctx.DM) {
		m.second->run();
	}
	for (auto & m : ctx.EM) {
		m.second->run();
	}
	ctx.SM->run();

	ctx.SM->wait();

	ctx.finalize();

	auto end = std::chrono::system_clock::now();

	std::cout << "REALTIME: " << std::chrono::duration<double>(end - start).count() << std::endl;

	return 0;
}