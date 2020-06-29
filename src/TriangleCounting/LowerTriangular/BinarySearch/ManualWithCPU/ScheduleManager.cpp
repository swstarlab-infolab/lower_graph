#include "ScheduleManager.h"

#include "context.h"
#include "util.h"

#include <cstdio>

static auto merge(std::unordered_map<int, sp<bchan<Report>>> & in)
{
	auto out = makeSp<bchan<Report>>(ctx.chanSz);

	std::thread([&, out] {
		std::unordered_map<int, std::thread> waitGroup;

		for (auto & kv : in) {
			waitGroup[kv.first] = std::thread([&, kv] {
				for (auto & e : *(kv.second)) {
					out->push(e);
				}
			});
		}

		for (auto & f : waitGroup) {
			if (f.second.joinable()) {
				f.second.join();
			}
		}

		out->close();
	}).detach();

	return out;
}

void ScheduleManager::run()
{
	std::thread([&] {
		for (uint32_t row = 0; row < ctx.grid.count; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Order order = {{{i, col}, {row, col}, {row, i}}};

					std::array<size_t, 3> fbyte;
					for (int i = 0; i < 3; i++) {
						auto fname = ctx.folder / filenameEncode(order[i]);
						fbyte[i]   = fs::file_size(fname.string() + ".row") +
								   fs::file_size(fname.string() + ".ptr") +
								   fs::file_size(fname.string() + ".col");
					}

					if (fbyte[0] < ctx.threshold && fbyte[1] < ctx.threshold &&
						fbyte[2] < ctx.threshold) {
						ctx.chan.orderCPU->push(order);
					} else {
						ctx.chan.orderGPU->push(order);
					}
				}
			}
		}
		ctx.chan.orderCPU->close();
		ctx.chan.orderGPU->close();
	}).detach();
}

void ScheduleManager::wait()
{
	Count totalTriangles = 0;

	for (auto & r : *(merge(ctx.chan.report))) {
		fprintf(stdout,
				"RESULT: DEV %d (%3d,%3d)(%3d,%3d)(%3d,%3d)=%16lld\n",
				r.deviceID,
				r.g3[0][0],
				r.g3[0][1],
				r.g3[1][0],
				r.g3[1][1],
				r.g3[2][0],
				r.g3[2][1],
				r.triangle);
		totalTriangles += r.triangle;
	}

	fprintf(stdout, "total triangles: %lld\n", totalTriangles);
}