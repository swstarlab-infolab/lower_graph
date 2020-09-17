#include "cache.h"
#include "logging.h"
#include "scheduler.h"
#include "type.h"
#include "util.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

size_t counting() { return 0UL; }

int main(int argc, char * argv[])
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <folderPath>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	auto folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

	Scheduler sched;
	Cache	  cache;
	cache.folderPath = folderPath;

	sched.init(folderPath);
	LOG("Sched init complete");
	cache.init();
	LOG("Cache init complete");

	/*
		// select some jobs
		LOGF("Sched Criteria: %ld", sched.criteria);

		std::vector<std::thread> runner(sched.gpus + 1);

		for (size_t i = 0; i < runner.size(); i++) {
			runner[i] = std::thread([&, i] {
				for (;;) {
					std::vector<std::vector<std::string>> result;
					if (sched.fetchJob(i, result)) {
						size_t grid0_id = strtol(result[0][0].c_str(), nullptr, 10);
						size_t grid1_id = strtol(result[0][1].c_str(), nullptr, 10);
						size_t grid2_id = strtol(result[0][2].c_str(), nullptr, 10);

						// LOGF("I am %ld, I got %ld, %ld, %ld", i, grid0_id, grid1_id, grid2_id);

						size_t triangles = 0L;
						double load_time = 0.0, kernel_time = 0.0;

						{
							auto start = std::chrono::system_clock::now();
							// cache.load();
							auto end  = std::chrono::system_clock::now();
							load_time = std::chrono::duration<double>(end - start).count();
						}
						{
							auto start	= std::chrono::system_clock::now();
							triangles	= counting();
							auto end	= std::chrono::system_clock::now();
							kernel_time = std::chrono::duration<double>(end - start).count();
						}

						// cache.done();

						sched.finishJob(
							grid0_id, grid1_id, grid2_id, triangles, load_time, kernel_time);
					} else {
						break;
					}
				}
			});
		}

		for (size_t i = 0; i < runner.size(); i++) {
			if (runner[i].joinable()) {
				runner[i].join();
			}
		}
		*/

	return 0;
}