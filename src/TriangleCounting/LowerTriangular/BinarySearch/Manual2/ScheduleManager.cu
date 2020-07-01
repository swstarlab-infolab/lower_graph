#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"
#include "util.h"

#include <memory>
#include <thread>
#include <vector>

std::pair<std::shared_ptr<bchan<Command>>, std::shared_ptr<bchan<Command>>>
ScheduleManager(Context const & ctx)
{
	auto outGPU = std::make_shared<bchan<Command>>(1 << 4);
	auto outCPU = std::make_shared<bchan<Command>>(1 << 4);

	std::thread([&, outGPU, outCPU] {
		auto const MAXROW = ctx.meta.info.count.row;

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Command req;
					req.gidx = {{{i, col}, {row, col}, {row, i}}};

					std::array<size_t, 3> fbyte = {
						0,
					};

					for (int j = 0; j < 3; j++) {
						auto fpath = ctx.folderPath / filenameEncode(req.gidx[j]);
						fbyte[j]   = size_t(fs::file_size(fs::path(fpath.string() + ".row"))) +
								   size_t(fs::file_size(fs::path(fpath.string() + ".ptr"))) +
								   size_t(fs::file_size(fs::path(fpath.string() + ".col")));
					}

					printf("SM: (%3d,%3d) %ld Bytes, (%3d,%3d) %ld Bytes, (%3d,%3d) %ld Bytes",
						   i,
						   col,
						   fbyte[0],
						   row,
						   col,
						   fbyte[1],
						   row,
						   i,
						   fbyte[2]);

					if (ctx.deviceCount > 0) {
						if (fbyte[0] < ctx.cpuGPUThreshold && fbyte[1] < ctx.cpuGPUThreshold &&
							fbyte[2] < ctx.cpuGPUThreshold) {
							printf(" -> CPU\n");
							outCPU->push(req);
						} else {
							printf(" -> GPU\n");
							outGPU->push(req);
						}
					} else {
						printf(" -> CPU\n");
						outCPU->push(req);
					}
				}
			}
		}

		outGPU->close();
		outCPU->close();
	}).detach();

	return std::make_pair(outGPU, outCPU);
}

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes)
{
	Count  totalTriangles	= 0;
	double totalElapsedTime = 0.0;

	for (auto & res : *executionRes) {
		fprintf(stdout,
				"Result: Device %2d (%3d,%3d)(%3d,%3d)(%3d,%3d)=%16lld,%16.6lf(sec)\n",
				res.deviceID,
				res.gidx[0][0],
				res.gidx[0][1],
				res.gidx[1][0],
				res.gidx[1][1],
				res.gidx[2][0],
				res.gidx[2][1],
				res.triangle,
				res.elapsedTime);
		totalTriangles += res.triangle;
		totalElapsedTime += res.elapsedTime;
	}

	fprintf(stdout, "total triangles: %lld\n", totalTriangles);
	fprintf(stdout, "total elapsed time: %lf (sec)\n", totalElapsedTime);
}