#include "ScheduleManager.cuh"
#include "make.cuh"
#include "type.cuh"
#include "util.cuh"

#include <memory>
#include <thread>
#include <vector>

std::shared_ptr<bchan<ThreeGrids>> ScheduleManager(Context const & ctx)
{
	// auto out = make<bchan<Command>>(1 << 4);
	auto out = std::make_shared<bchan<ThreeGrids>>(1 << 4);

	std::thread([&, out] {
		auto const MAXROW = ctx.grid.count;

		for (size_t row = 0; row < MAXROW; row++) {
			for (size_t col = 0; col <= row; col++) {
				for (size_t i = col; i <= row; i++) {
					// using GridIndex as little different way
					GridIndex filesInfo[3];
					filesInfo[0].xy = {i, col};
					filesInfo[1].xy = {row, col};
					filesInfo[2].xy = {row, i};

					for (int j = 0; j < 3; j++) {
						auto content = std::to_string(filesInfo[j].xy[0]) + "-" +
									   std::to_string(filesInfo[j].xy[1]);
						auto shards = csrShardCount(ctx.folderPath, content);

						if (shards == 0) {
							goto NOTHING;
						} else {
							filesInfo[j].shard = int64_t(shards);
						}
					}

					ThreeGrids req;
					req[0].xy = {i, col};
					req[1].xy = {row, col};
					req[2].xy = {row, i};

					size_t s[3];
					for (s[0] = 0; s[0] < filesInfo[0].shard; s[0]++) {
						for (s[1] = 0; s[1] < filesInfo[1].shard; s[1]++) {
							for (s[2] = 0; s[2] < filesInfo[2].shard; s[2]++) {
								for (int j = 0; j < 3; j++) {
									req[j].shard = (filesInfo[j].shard > 1) ? int64_t(s[j]) : -1;
								}

								fprintf(stdout,
										"Start: "
										"<(%ld,%ld),%ld><(%ld,%ld),%ld><(%ld,%ld),%ld>\n",
										req[0].xy[0],
										req[0].xy[1],
										req[0].shard,
										req[1].xy[0],
										req[1].xy[1],
										req[1].shard,
										req[2].xy[0],
										req[2].xy[1],
										req[2].shard);
								out.get()->push(req);
							}
						}
					}

				NOTHING:
					continue;
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
				"Result: device%d<(%ld,%ld),%ld><(%ld,%ld),%ld><(%ld,%ld),%ld>=%lld,%.6lf(sec)\n",
				res.deviceID,
				res.gidx[0].xy[0],
				res.gidx[0].xy[1],
				res.gidx[0].shard,
				res.gidx[1].xy[0],
				res.gidx[1].xy[1],
				res.gidx[1].shard,
				res.gidx[2].xy[0],
				res.gidx[2].xy[1],
				res.gidx[2].shard,
				res.triangle,
				res.elapsedTime);
		totalTriangles += res.triangle;
		totalElapsedTime += res.elapsedTime;
	}

	fprintf(stdout, "total triangles: %lld\n", totalTriangles);
	fprintf(stdout, "total elapsed time: %lf (sec)\n", totalElapsedTime);
}