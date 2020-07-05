#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"
#include "util.h"

#include <memory>
#include <thread>
#include <vector>

// if true, cpu kernel runs.
static auto criteria(std::array<std::array<size_t, 3>, 3> const & fbyte)
{
	// size constraint
	bool Aa = (fbyte[0][0] < ((1 << 20) * sizeof(Vertex)));
	bool Ab = (fbyte[1][0] < ((1 << 21) * sizeof(Vertex)));
	bool Ac = (fbyte[2][0] < ((1 << 20) * sizeof(Vertex)));

	// ratio constraint
	//	bool B = ((double(fbyte[2][2]) / double(fbyte[2][0])) < 2.5);
	bool B = true;
	// bool C = ((double(fbyte[0][2]) / double(fbyte[0][0])) < 2.5);
	bool C = true;

	return Aa && Ab && Ac && B && C;
	// fbyte[0] < ctx.cpuGPUThreshold && fbyte[1] < ctx.cpuGPUThreshold && fbyte[2] <
	// ctx.cpuGPUThreshold);
}

std::pair<std::shared_ptr<bchan<Command>>, std::shared_ptr<bchan<Command>>>
ScheduleManager(Context const & ctx)
{
	auto outGPU = std::make_shared<bchan<Command>>(1 << 4);

	if (ctx.deviceCount > 0) {
		std::thread([&, outGPU] {
			auto const MAXROW = ctx.meta.info.count.row;

			for (uint32_t row = 0; row < MAXROW; row++) {
				for (uint32_t col = 0; col <= row; col++) {
					for (uint32_t i = col; i <= row; i++) {
						Command req;
						req.gidx = {{{i, col}, {row, col}, {row, i}}};

						std::array<std::array<size_t, 3>, 3> fbyte = {
							0,
						};

						for (int j = 0; j < 3; j++) {
							auto fpath	= ctx.folderPath / filenameEncode(req.gidx[j]);
							fbyte[j][0] = size_t(fs::file_size(fs::path(fpath.string() + ".row")));
							fbyte[j][1] = size_t(fs::file_size(fs::path(fpath.string() + ".ptr")));
							fbyte[j][2] = size_t(fs::file_size(fs::path(fpath.string() + ".col")));
						}

						if (!criteria(fbyte)) {
							outGPU->push(req);
						}
					}
				}
			}

			outGPU->close();
		}).detach();
	}

	auto outCPU = std::make_shared<bchan<Command>>(1 << 4);

	std::thread([&, outCPU] {
		auto const MAXROW = ctx.meta.info.count.row;

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Command req;
					req.gidx = {{{i, col}, {row, col}, {row, i}}};

					std::array<std::array<size_t, 3>, 3> fbyte = {
						0,
					};

					for (int j = 0; j < 3; j++) {
						auto fpath	= ctx.folderPath / filenameEncode(req.gidx[j]);
						fbyte[j][0] = size_t(fs::file_size(fs::path(fpath.string() + ".row")));
						fbyte[j][1] = size_t(fs::file_size(fs::path(fpath.string() + ".ptr")));
						fbyte[j][2] = size_t(fs::file_size(fs::path(fpath.string() + ".col")));
					}

					if (ctx.deviceCount > 0) {
						if (criteria(fbyte)) {
							outCPU->push(req);
						}
					} else {
						outCPU->push(req);
					}
				}
			}
		}

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
