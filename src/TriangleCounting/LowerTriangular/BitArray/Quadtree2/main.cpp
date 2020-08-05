#include "data_man.h"
#include "exec_man.h"
#include "sched_man.h"
#include "type.h"
#include "util.h"

#include <atomic>
#include <cuda_runtime.h>

int main(int argc, char * argv[])
{
	// Argument
	if (argc != 5) {
		fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	auto folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

	Exec::GPUSetting setting;
	setting.stream = atoi(argv[2]);
	setting.block  = atoi(argv[3]);
	setting.thread = atoi(argv[4]);

	int devices = 0;
	cudaGetDeviceCount(&devices);

	auto sched = std::make_shared<Sched::Manager>(folderPath);

	std::vector<std::shared_ptr<Data::Manager>> data(devices);
	for (int i = 0; i < devices; i++) {
		data[i] = std::make_shared<Data::Manager>(i, folderPath);
	}

	std::vector<std::shared_ptr<Exec::Manager>> exec(devices * setting.stream);
	for (int i = 0; i < devices; i++) {
		for (int j = 0; j < setting.stream; j++) {
			exec[i * setting.stream + j] =
				std::make_shared<Exec::Manager>(i, j, setting, sched, data[i]);
		}
	}

	sched->run();
	for (int i = 0; i < devices; i++) {
		data[i]->run();
	}

	// Start!!!
	std::mutex	 resultMtx;
	Exec::Result result = {
		0,
	};
	std::vector<Exec::OutType> execOut(devices * setting.stream);

	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < devices * setting.stream; i++) {
		execOut[i] = exec[i]->run();
	}

	/// Sync workers
	pThread(devices * setting.stream, [&](size_t const i) {
		Count  myTotal = 0;
		double myLoad = 0.0, myKernel = 0.0;
		for (auto & res : *execOut[i]) {
			fprintf(stdout,
					"Device=%d / Stream=%d / %s / %s / %s => %lld / %.6lf(sec) / %.6lf(sec)\n",
					res.deviceID,
					res.streamID,
					res.job[0].string().c_str(),
					res.job[1].string().c_str(),
					res.job[2].string().c_str(),
					res.triangle,
					res.time.load,
					res.time.kernel);
			myTotal += res.triangle;
			myLoad += res.time.load;
			myKernel += res.time.kernel;
		}

		{
			std::lock_guard<std::mutex> lg(resultMtx);
			result.triangle += myTotal;
			result.time.load += myLoad;
			result.time.kernel += myKernel;
		}
	});

	auto end = std::chrono::system_clock::now();

	fprintf(stdout,
			"total triangles: %lld\n"
			"total load time: %.6lf, per-Exec load time: %.6lf\n"
			"total kernel time: %.6lf, per-Exec load time: %.6lf\n"
			"REALTIME: %.6lf\n",
			result.triangle,
			result.time.load,
			result.time.load / double(devices),
			result.time.kernel,
			result.time.kernel / double(devices),
			std::chrono::duration<double>(end - start).count());

	return 0;
}