#include "exec_man.h"

#include "util.h"

#include <chrono>

namespace Exec
{
OutType Manager::run()
{
	auto out = std::make_shared<boost::fibers::buffered_channel<Result>>(1 << 8);
	std::thread([=] {
		for (auto & job : *this->sched->out) {
			Result res;
			res.job = job;

			Grids g;

			{
				auto start = std::chrono::system_clock::now();
				pThread(9, [&](size_t const i) {
					auto key = job[i / 3].string() + EXTENSION[i % 3];

					g[i / 3][i % 3] = this->data->load(key);
				});
				auto end	  = std::chrono::system_clock::now();
				res.time.load = std::chrono::duration<double>(end - start).count();
			}

			{
				auto start		= std::chrono::system_clock::now();
				res.triangle	= this->launchKernelGPU(g);
				auto end		= std::chrono::system_clock::now();
				res.time.kernel = std::chrono::duration<double>(end - start).count();
			}

			res.deviceID = this->deviceID;
			res.streamID = this->streamID;

			out->push(res);

			for (int i = 0; i < 9; i++) {
				auto key = job[i / 3].string() + EXTENSION[i % 3];
				this->data->done(key);
			}
		}

		out->close();
	}).detach();
	return out;
}

Manager::~Manager() { printf("Destructor: Exec::Manager, No Error\n"); }

} // namespace Exec