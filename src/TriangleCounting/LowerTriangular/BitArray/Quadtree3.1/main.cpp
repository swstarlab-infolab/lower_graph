#include "base/type.h"
//#include "counting.h"
#include "counting.h"
#include "kvfilecache.h"
#include "scheduler.h"
#include "util/logging.h"
#include "util/util.h"
#include "util/util_parallel.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char * argv[])
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <folderPath>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	auto folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

	GridInfo gridInfo;
	gridInfo.init(folderPath);
	LOG("Complete: gridInfo init");

	KeyValueFileCache cache;
	cache.init(gridInfo);
	LOG("Complete: cache init");

	cache.devices = 0;

	Scheduler sched;
	sched.init(gridInfo);
	LOG("Complete: scheduler init");

	std::vector<std::thread> runner(cache.devices + 1);

#ifdef CPUOFF
	for (int myDevID = 0; myDevID < cache.devices; myDevID++) {
#else
	for (int myDevID = -1; myDevID < cache.devices; myDevID++) {
#endif
		runner[myDevID + 1] = std::thread([&, myDevID] {
			LOGF("runner %d launching...", myDevID);
			std::array<std::array<DataInfo<void>, 3>, 3> info;
			Job											 job;

			Lookups Ls;
			for (auto & L : Ls) {
				L.resize((1UL << 24) + 1);
			}

			while (sched.fetchJob(myDevID, job)) {
				// LOGF("I am %d ==> Job: <%d, %d, %d>", myDevID, job[0], job[1], job[2]);
				Count  triangles = 0L;
				double load_time = 0.0, kernel_time = 0.0;

				{
					auto start = std::chrono::system_clock::now();

					boost::asio::thread_pool myPool(9);

					for (int g = 0; g < 3; g++) {
						for (int t = 0; t < 3; t++) {
							boost::asio::post(myPool, [&, g, t] {
								DataManagerKey key;
								key.gridID	 = job[g];
								key.fileType = t;

								info[g][t] = cache.mustPrepare(myDevID, key);
							});
						}
					}

					myPool.join();

					auto end  = std::chrono::system_clock::now();
					load_time = std::chrono::duration<double>(end - start).count();
				}

				{
					auto start = std::chrono::system_clock::now();
					if (myDevID < 0) {
						Grids Gs;
						for (int g = 0; g < 3; g++) {
							for (int t = 0; t < 3; t++) {
								Gs[g][t].addr = (uint32_t *)info[g][t].addr;
								Gs[g][t].byte = info[g][t].byte;
							}
						}
						triangles = countingCPU(Gs, Ls);
					} else {
						// triangles	= countingGPU(info);
					}
					auto end	= std::chrono::system_clock::now();
					kernel_time = std::chrono::duration<double>(end - start).count();
				}

				boost::asio::thread_pool myPool(9);
				for (int g = 0; g < 3; g++) {
					for (int t = 0; t < 3; t++) {
						boost::asio::post(myPool, [&, g, t] {
							DataManagerKey key;
							key.gridID	 = job[g];
							key.fileType = t;

							cache.done(myDevID, key);
						});
					}
				}

				myPool.join();

				sched.recordJobResult(job, triangles, load_time, kernel_time);

				// std::cin.ignore();
				LOGF("I am %2d ==> Job: <%4d, %4d, %4d> done: triangles=%lld, loadtime=%lf, "
					 "kerneltime=%lf",
					 myDevID,
					 job[0],
					 job[1],
					 job[2],
					 triangles,
					 load_time,
					 kernel_time);
			}
		});
	}

	for (size_t i = 0; i < runner.size(); i++) {
		if (runner[i].joinable()) {
			runner[i].join();
		}
	}

	return 0;
}