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
		printf("g0_idx_x,g0_idx_y,g1_idx_x,g1_idx_y,g2_idx_x,g2_idx_y,"
			   "g0_row_byte,g0_col_byte,g1_row_byte,g1_col_byte,g2_row_byte,g2_col_byte,"
			   "block,thread,triangle,time\n");
		auto const MAXROW = ctx.grid.count;

		for (uint32_t row = 0; row < MAXROW; row++) {
			for (uint32_t col = 0; col <= row; col++) {
				for (uint32_t i = col; i <= row; i++) {
					Command req;
					req.gidx = {{{i, col}, {row, col}, {row, i}}};

					for (int i = 0; i < 3; i++) {
						auto fpath = ctx.folderPath / (std::to_string(req.gidx[i][0]) + "-" +
													   std::to_string(req.gidx[i][1]));

						if (!fs::exists(fs::path(fpath.string() + ".row"))) {
							goto NOTHING;
						}
					}

					out.get()->push(req);

				NOTHING:
				}
			}
		}

		out.get()->close();
	}).detach();

	return out;
}

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes)
{
	Count totalTriangles = 0;

	for (auto & res : *executionRes.get()) {
		/*
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
				*/
		totalTriangles += res.triangle;
	}

	fprintf(stdout, "total triangles: %lld\n", totalTriangles);
}