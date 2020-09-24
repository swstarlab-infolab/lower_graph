#include "scheduler.h"

#include "util/logging.h"
#include "util/my_mysql.h"
#include "util/util.h"
#include "util/util_parallel.h"

#include <array>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <cuda_runtime.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <thread>
#include <vector>

void Scheduler::init(GridInfo const & gridInfo)
{
	// 아랫 부분 처리해야 함
	/*
		tbb::parallel_sort(
			gridInfo.begin(), gridInfo.end(), [](GridInfoValue const & l, GridInfoValue const & r) {
				return l.byte[2] > r.byte[2];
			});

		this->criteria = gridInfo[gridInfo.size() / 5].byte[2];
		*/

	boost::asio::thread_pool myPool(std::thread::hardware_concurrency());

	for (uint32_t row = 0; row < gridInfo.matrix.size(); row++) {
		for (uint32_t col = 0; col <= row; col++) {
			for (uint32_t i = col; i <= row; i++) {
				boost::asio::post(myPool, [=, &gridInfo] {
					std::array<std::array<uint32_t, 2>, 3> gidx = {
						{{i, col}, {row, col}, {row, i}}};

					if (gridInfo.xy(gidx[0]).size() == 0 || gridInfo.xy(gidx[1]).size() == 0 ||
						gridInfo.xy(gidx[2]).size() == 0) {
						return;
					}

					for (auto & g0 : gridInfo.xy(gidx[0])) {
						for (auto & g1 : gridInfo.xy(gidx[1])) {
							for (auto & g2 : gridInfo.xy(gidx[2])) {
								bool condition1 = std::max(g0.range[1][0], g1.range[1][0]) <
												  std::min(g0.range[1][1], g1.range[1][1]);
								bool condition2 = std::max(g1.range[0][0], g2.range[0][0]) <
												  std::min(g1.range[0][1], g2.range[0][1]);
								bool condition3 = std::max(g0.range[0][0], g2.range[1][0]) <
												  std::min(g0.range[0][1], g2.range[1][1]);

								if (condition1 && condition2 && condition3) {
									if (g0.byte[2] < this->criteria &&
										g1.byte[2] < this->criteria &&
										g2.byte[2] < this->criteria) {

										this->jobsCPU.push(Job{g0.id, g1.id, g2.id});
									} else {
										this->jobsGPU.push(Job{g0.id, g1.id, g2.id});
									}
								}
							}
						}
					}
				});
			}
		}
	}
	myPool.join();

	LOGF("CPU jobs: %ld, GPU jobs: %ld, work stealing available",
		 this->jobsCPU.unsafe_size(),
		 this->jobsGPU.unsafe_size());
}

bool Scheduler::fetchJob(int const device_id, Job & job)
{
	/*
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	auto q = conn.query();

	if (device_id < 0) {
		// CPU
		q = conn.query(sprn("\
			UPDATE IGNORE jobs AS j\
			SET j.device_id = %ld, j.state = 'RUNNING'\
			WHERE\
				(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid0_id) < %ld AND\
				(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid1_id) < %ld AND\
				(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid2_id) < %ld AND\
				j.state = 'PENDING'\
			ORDER BY j.grid0_id, j.grid1_id, j.grid2_id ASC\
			LIMIT 1;",
							device_id,
							this->criteria,
							this->criteria,
							this->criteria));
	} else {
		// GPU
		q = conn.query(sprn("\
			UPDATE IGNORE jobs AS j\
			SET j.device_id = %ld, j.state = 'RUNNING'\
			WHERE\
				((SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid0_id) >= %ld OR\
				(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid1_id) >= %ld OR\
				(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid2_id) >= %ld) AND\
				j.state = 'PENDING'\
			ORDER BY j.grid0_id, j.grid1_id, j.grid2_id ASC\
			LIMIT 1;",
							device_id,
							this->criteria,
							this->criteria,
							this->criteria));
	}

	assert(q.exec());

	if (q.affected_rows()) {
		if (device_id < 0) {
			// CPU
			q = conn.query(sprn("\
				SELECT j.grid0_id, j.grid1_id, j.grid2_id\
				FROM jobs AS j\
				WHERE\
					(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid0_id) < %ld AND\
					(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid1_id) < %ld AND\
					(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid2_id) < %ld AND\
					j.device_id = %ld AND\
					j.state = 'RUNNING';",
								this->criteria,
								this->criteria,
								this->criteria,
								device_id));
		} else {
			// GPU
			q = conn.query(sprn("\
				SELECT j.grid0_id, j.grid1_id, j.grid2_id\
				FROM jobs AS j\
				WHERE\
					((SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid0_id) >= %ld OR\
					(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid1_id) >= %ld OR\
					(SELECT g.col_byte FROM grids AS g WHERE g.id = j.grid2_id) >= %ld) AND\
					j.device_id = %ld AND\
					j.state = 'RUNNING';",
								this->criteria,
								this->criteria,
								this->criteria,
								device_id));
		}

		auto result = q.store();
		return Grid3{s2l(result[0][0]), s2l(result[0][1]), s2l(result[0][2])};
	}

	q = conn.query(sprn("\
		UPDATE IGNORE jobs AS j\
		SET j.device_id = %ld, j.state = 'RUNNING'\
		WHERE\
			j.state = 'PENDING'\
		ORDER BY j.grid0_id, j.grid1_id, j.grid2_id ASC\
		LIMIT 1;",
						device_id));
	assert(q.exec());

	if (q.affected_rows()) {
		conn.query(sprn("\
			SELECT j.grid0_id, j.grid1_id, j.grid2_id\
			FROM jobs AS j\
			WHERE j.state = 'RUNNING' AND j.device_id = %ld;",
						device_id));
		auto result = q.store();
		return Grid3{s2l(result[0][0]), s2l(result[0][1]), s2l(result[0][2])};
	}

	LOG("No more jobs. Halting.");
	*/
	return false;
}

void Scheduler::recordJobResult(Job const &	 grid3,
								size_t const triangles,
								double const load_time,
								double const kernel_time)
{
	/*
	mysqlpp::Connection conn(false);
	mysqlConnect(conn);

	auto q = conn.query(sprn("\
		UPDATE jobs AS j\
		SET j.triangles = %ld, j.load_time = %lf, j.kernel_time = %lf, j.state='FINISH'\
		WHERE j.grid0_id = %ld AND j.grid1_id = %ld AND j.grid2_id = %ld AND j.state =
	'RUNNING';", triangles, load_time, kernel_time, grid3[0], grid3[1], grid3[2]));
	assert(q.exec());
	*/
}