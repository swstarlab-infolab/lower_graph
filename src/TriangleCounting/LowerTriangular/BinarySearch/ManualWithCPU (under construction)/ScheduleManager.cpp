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
		// printf("SM: start\n");

		for (uint32_t row = 0; row < ctx.grid.count; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Order order = {{{i, col}, {row, col}, {row, i}}};

					std::array<size_t, 3> fbyte = {
						0,
					};

					for (int i = 0; i < 3; i++) {
						auto fpath = ctx.folder / filenameEncode(order[i]);
						fbyte[i]   = size_t(fs::file_size(fs::path(fpath.string() + ".row"))) +
								   size_t(fs::file_size(fs::path(fpath.string() + ".ptr"))) +
								   size_t(fs::file_size(fs::path(fpath.string() + ".col")));
					}

					printf("SM: (%3d,%3d) %ld Bytes, (%3d,%3d) %ld Bytes, (%3d,%3d) %ld Bytes\n",
						   i,
						   col,
						   fbyte[0],
						   row,
						   col,
						   fbyte[1],
						   row,
						   i,
						   fbyte[2]);

					if (fbyte[0] < ctx.threshold && fbyte[1] < ctx.threshold &&
						fbyte[2] < ctx.threshold) {
						// printf("SM: pushed to CPU\n");
						ctx.chan.orderCPU->push(order);
					} else {
						// printf("SM: pushed to GPU\n");
						ctx.chan.orderGPU->push(order);
					}
				}
			}
		}

		ctx.chan.orderCPU->close();
		ctx.chan.orderGPU->close();

		// printf("SM: closed all channel\n");
	}).detach();
}

void ScheduleManager::wait()
{
	// printf("SM: wait\n");

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