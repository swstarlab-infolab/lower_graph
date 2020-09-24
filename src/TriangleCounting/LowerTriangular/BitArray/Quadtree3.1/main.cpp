#include "base/type.h"
//#include "counting.h"
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

	Scheduler sched;
	sched.init(gridInfo);
	LOG("Complete: scheduler init");

	/*
		std::vector<std::thread> runner(gpus + 1);

		for (int myDevID = -1; myDevID < gpus; myDevID++) {
			LOGF("runner %d launching...", myDevID);
			runner[myDevID] = std::thread([&, myDevID] {
				std::array<std::array<DataInfo, 3>, 3> info;

				LOGF("I am %d ==> start!", myDevID);

				while (true) {
					auto grid3 = sched.fetchJob(myDevID);

					if (grid3 != sched.jobHalt) {

						LOGF("I am %d ==> Job: <%ld, %ld, %ld>", myDevID, grid3[0], grid3[1],
	   grid3[2]);

						size_t triangles = 0L;
						double load_time = 0.0, kernel_time = 0.0;

						{
							auto start = std::chrono::system_clock::now();

							boost::asio::thread_pool myPool(9);

							for (int g = 0; g < 3; g++) {
								for (int t = 0; t < 3; t++) {
									boost::asio::post(myPool, [=, &info, &cache] {
										MemReqInfo reqInfo;
										reqInfo.device_id = myDevID;
										reqInfo.grid_id	  = grid3[g];
										reqInfo.file_type = t;

										// LOGF("reqInfo device_id grid_id file_type = %d, %ld, %d",
										// reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);
										info[g][t] = cache.load(reqInfo);
									});
								}
							}

							myPool.join();

							auto end  = std::chrono::system_clock::now();
							load_time = std::chrono::duration<double>(end - start).count();
						}

						{
							auto start	= std::chrono::system_clock::now();
							triangles	= counting(info);
							auto end	= std::chrono::system_clock::now();
							kernel_time = std::chrono::duration<double>(end - start).count();
						}

						boost::asio::thread_pool myPool(9);
						for (int g = 0; g < 3; g++) {
							for (int t = 0; t < 3; t++) {
								boost::asio::post(myPool, [&, g, t] {
									MemReqInfo reqInfo;
									reqInfo.device_id = myDevID;
									reqInfo.grid_id	  = grid3[g];
									reqInfo.file_type = t;

									cache.done(reqInfo);
								});
							}
						}

						myPool.join();

						sched.finishJob(grid3, triangles, load_time, kernel_time);

						// std::cin.ignore();

					} else {
						LOGF("I am %d ==> Job: <%ld, %ld, %ld> Halting",
							 myDevID,
							 grid3[0],
							 grid3[1],
							 grid3[2]);
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