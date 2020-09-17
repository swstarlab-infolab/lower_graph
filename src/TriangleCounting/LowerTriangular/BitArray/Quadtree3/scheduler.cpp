#include "scheduler.h"

#include "my_mysql.h"
#include "shard.h"
#include "type.h"
#include "util.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

void Scheduler::init(fs::path const & folderPath)
{
	// Initialize cache
	int _gpus;
	cudaGetDeviceCount(&_gpus);
	this->gpus = (size_t)_gpus;

	mysqlpp::Connection conn(false);
	mysqlConnecct(conn);

	conn.query("\
		CREATE TABLE IF NOT EXISTS grids (\
			id INT NOT NULL AUTO_INCREMENT,\
			row INT NOT NULL,\
			col INT NOT NULL,\
			depth INT NOT NULL,\
			shard_row INT NOT NULL,\
			shard_col INT NOT NULL,\
			range_row_from BIGINT NOT NULL, \
			range_row_to BIGINT NOT NULL, \
			range_col_from BIGINT NOT NULL, \
			range_col_to BIGINT NOT NULL, \
			row_byte BIGINT NOT NULL,\
			ptr_byte BIGINT NOT NULL,\
			col_byte BIGINT NOT NULL,\
			stem VARCHAR(1024),\
			PRIMARY KEY (id),\
			UNIQUE KEY (row, col, depth, shard_row, shard_col)\
	);");

	conn.query("\
		CREATE TABLE IF NOT EXISTS jobs (\
			grid0_id INT NOT NULL,\
			grid1_id INT NOT NULL,\
			grid2_id INT NOT NULL,\
			device_id INT DEFAULT NULL,\
			triangles BIGINT DEFAULT NULL,\
			load_time DOUBLE DEFAULT NULL,\
			kernel_time DOUBLE DEFAULT NULL,\
			state ENUM('PENDING','RUNNING','FINISH') DEFAULT 'PENDING',\
			last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP\
				ON UPDATE CURRENT_TIMESTAMP,\
			FOREIGN KEY (grid0_id)\
				REFERENCES grids (id)\
				ON UPDATE CASCADE\
				ON DELETE CASCADE,\
			FOREIGN KEY (grid1_id)\
				REFERENCES grids (id)\
				ON UPDATE CASCADE\
				ON DELETE CASCADE,\
			FOREIGN KEY (grid2_id)\
				REFERENCES grids (id)\
				ON UPDATE CASCADE\
				ON DELETE CASCADE,\
			UNIQUE KEY (grid0_id, grid1_id, grid2_id)\
		);");

	// Write all informations of grids in the folder
	for (fs::recursive_directory_iterator curr(folderPath), end; curr != end; ++curr) {
		if (fs::is_regular_file(curr->path()) && fs::file_size(curr->path()) > 0 &&
			curr->path().extension() == ".row") {
			auto stem = curr->path().stem().string();

			auto row_file_byte = fs::file_size(folderPath / fs::path(stem + ".row"));
			auto ptr_file_byte = fs::file_size(folderPath / fs::path(stem + ".ptr"));
			auto col_file_byte = fs::file_size(folderPath / fs::path(stem + ".col"));

			ShardIndex sIdx;
			sIdx.parse(stem);

			ShardRange sRange;
			sRange.conv(sIdx);
			sRange.increase(24);

			conn.query(sprn("\
								INSERT IGNORE INTO grids (\
									row, col, depth, shard_row, shard_col, range_row_from, range_row_to, range_col_from, range_col_to, row_byte, ptr_byte, col_byte, stem)\
									VALUES (\
									%d, %d, %ld, %d, %d, %ld, %ld, %ld, %ld, %ld, %ld, %ld, '%s')\
								",
							sIdx.grid[0],
							sIdx.grid[1],
							sIdx.depth,
							sIdx.shard[0],
							sIdx.shard[1],
							sRange.range[0][0],
							sRange.range[0][1],
							sRange.range[1][0],
							sRange.range[1][1],
							row_file_byte,
							ptr_file_byte,
							col_file_byte,
							stem.c_str()));
		}
	}

	auto MAXROW = mysqlSingleValueSizeT(conn, "SELECT MAX(g.row) FROM grids AS g");

	boost::asio::thread_pool pool(std::thread::hardware_concurrency());

	for (size_t row = 0; row < MAXROW; row++) {
		for (size_t col = 0; col <= row; col++) {
			for (size_t i = col; i <= row; i++) {

				// If you have errors, you can set:
				//     echo 1 |& sudo tee /proc/sys/net/ipv4/tcp_tw_reuse
				// and:
				//     mysql> set global max_connections = 1024;

				boost::asio::post(pool, [=] {
					MySQLConnection myConn;

					XY3 gidx = {{{i, col}, {row, col}, {row, i}}};

					// LOGF("try : (%ld, %ld), (%ld, %ld), (%ld, %ld)", gidx[0][0], gidx[0][1],
					// gidx[1][0], gidx[1][1], gidx[2][0], gidx[2][1]);

					for (int j = 0; j < 3; j++) {
						size_t exist = mysqlSingleValueSizeT(conn,
															 sprn("\
							SELECT COUNT(*)\
							FROM grids AS g\
							WHERE g.row = %d AND g.col = %d",
																  gidx[j][0],
																  gidx[j][1]));
						if (exist == 0) {
							// LOGF("skip: (%ld, %ld), (%ld, %ld), (%ld, %ld)", gidx[0][0],
							// gidx[0][1], gidx[1][0], gidx[1][1], gidx[2][0], gidx[2][1]);
							return;
						}
					}

					conn.query(sprn("\
		INSERT IGNORE INTO jobs (grid0_id, grid1_id, grid2_id)\
		SELECT\
			g0.id AS g0_id,\
			g1.id AS g1_id,\
			g2.id AS g2_id\
		FROM\
			(SELECT * FROM grids WHERE row = %ld AND col = %ld) AS g0,\
			(SELECT * FROM grids WHERE row = %ld AND col = %ld) AS g1,\
			(SELECT * FROM grids WHERE row = %ld AND col = %ld) AS g2\
		WHERE\
			GREATEST(g0.range_col_from, g1.range_col_from) < LEAST(g0.range_col_to, g1.range_col_to) AND\
			GREATEST(g1.range_row_from, g2.range_row_from) < LEAST(g1.range_row_to, g2.range_row_to) AND\
			GREATEST(g0.range_row_from, g2.range_col_from) < LEAST(g0.range_row_to, g2.range_col_to);",
									gidx[0][0],
									gidx[0][1],
									gidx[1][0],
									gidx[1][1],
									gidx[2][0],
									gidx[2][1]));

					// LOGF("done: (%ld, %ld), (%ld, %ld), (%ld, %ld)", gidx[0][0], gidx[0][1],
					// gidx[1][0], gidx[1][1], gidx[2][0], gidx[2][1]);
				});
			}
		}
	}

	pool.join();

	auto total_rows = mysqlSingleValueSizeT(conn, "SELECT COUNT(*) FROM grids");

	this->criteria = mysqlSingleValueSizeT(
		conn,
		sprn("SELECT col_byte FROM grids ORDER BY col_byte DESC LIMIT %ld, 1", total_rows / 5));
}

Grid3 Scheduler::fetchJob(size_t const device_id)
{
	mysqlpp::Connection conn(false);
	mysqlConnecct(conn);

	mysqlpp::Query q;

	if (device_id == -1) {
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

	if (q.affected_rows()) {
		if (device_id == -1) {
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

		return Grid3{s2l(q.store()[0][0]), s2l(q.store()[0][1]), s2l(q.store()[0][2])};
	}

	q = conn.query(sprn("\
		UPDATE IGNORE jobs AS j\
		SET j.device_id = %ld, j.state = 'RUNNING'\
		WHERE\
			j.state = 'PENDING'\
		ORDER BY j.grid0_id, j.grid1_id, j.grid2_id ASC\
		LIMIT 1;",
						device_id));

	if (q.affected_rows()) {
		conn.query(sprn("\
			SELECT j.grid0_id, j.grid1_id, j.grid2_id\
			FROM jobs AS j\
			WHERE j.state = 'RUNNING' AND j.device_id = %ld;",
						device_id));
		return Grid3{s2l(q.store()[0][0]), s2l(q.store()[0][1]), s2l(q.store()[0][2])};
	}

	LOG("No more jobs. Halting.");
	return jobHalt;
}

void Scheduler::finishJob(Grid3 const  grid3,
						  size_t const triangles,
						  double const load_time,
						  double const kernel_time)
{
	mysqlpp::Connection conn(false);
	mysqlConnecct(conn);

	conn.query(sprn("\
		UPDATE jobs AS j\
		SET j.triangles = %ld, j.load_time = %lf, j.kernel_time = %lf, j.state='FINISH'\
		WHERE j.grid0_id = %ld AND j.grid1_id = %ld AND j.grid2_id = %ld AND j.state = 'RUNNING';",
					triangles,
					load_time,
					kernel_time,
					grid3[0],
					grid3[1],
					grid3[2]));
}