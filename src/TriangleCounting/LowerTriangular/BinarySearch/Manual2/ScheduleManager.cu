#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"

#include <memory>
#include <thread>
#include <vector>

std::shared_ptr<bchan<Command>> ScheduleManager(Context const & ctx)
{
	// auto out = make<bchan<Command>>(1 << 4);
	auto out = std::make_shared<bchan<Command>>(1 << 4);

	std::thread([&, out] {
		auto const MAXROW = ctx.meta.info.count.row;

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Command req;
					req.gidx = {{{i, col}, {row, col}, {row, i}}};
					out.get()->push(req);
				}
			}
		}

		out.get()->close();
	}).detach();

	return out;
}

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes)
{
	Count  totalTriangles	= 0;
	double totalElapsedTime = 0.0;

	for (auto & res : *executionRes.get()) {
		fprintf(stdout,
				"RESULT: DEV %d (%3d,%3d)(%3d,%3d)(%3d,%3d)=%16lld,%16.6lf(sec)\n",
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