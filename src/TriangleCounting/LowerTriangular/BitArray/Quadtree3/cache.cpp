#include "cache.h"

#include "my_mysql.h"
#include "util.h"

#include <cuda_runtime.h>

void Cache::init()
{
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	conn.query("\
		CREATE TABLE IF NOT EXISTS cache (\
			device_id INT NOT NULL,\
			grid_id INT NOT NULL,\
			file_type TINYINT DEFAULT NULL,\
			state ENUM('NOTEXIST', 'LOADING', 'EXIST', 'EVICTING') DEFAULT 'NOTEXIST',\
			ref_count INT DEFAULT 0,\
			last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP\
				ON UPDATE CURRENT_TIMESTAMP,\
			addr BIGINT DEFAULT NULL,\
			byte BIGINT DEFAULT NULL,\
			FOREIGN KEY (grid_id)\
				REFERENCES grids (id)\
				ON UPDATE CASCADE\
				ON DELETE CASCADE,\
			UNIQUE KEY (device_id, grid_id, file_type));");

	cudaGetDeviceCount(&this->gpus);

	// GPU cache pool init
	this->device_pool.resize(this->gpus);

	for (int i = 0; i < this->gpus; i++) {
		cudaSetDevice(i);
		cudaDeviceReset();

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		free -= (1UL << 29);

		auto * mr			 = rmm::mr::get_current_device_resource();
		this->device_pool[i] = std::make_shared<DevicePoolType>(mr, free, free);
		rmm::mr::set_current_device_resource(&(*(this->device_pool[i])));
	}

	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!
	// 9월 18일에 작업할 것: 이 부분!!! byte 컬럼도 insert해야 하는데 그게 제대로 안되어있음!!!

	// GPU cache metadata init
	for (int i = 0; i < this->gpus; i++) {
		conn.query(sprn("\
			INSERT IGNORE INTO cache (device_id, grid_id, file_type, byte)\
				SELECT *\
				FROM\
					(\
						(SELECT %d AS device_id, g.id, g AS grid_id FROM grids AS g) temp0\
						CROSS JOIN\
						(SELECT 0 AS file_type UNION ALL SELECT 1 UNION ALL SELECT 2) temp1\
					);",
						i));
	}

	// CPU cache metadata init
	conn.query("\
		INSERT IGNORE INTO cache (device_id, grid_id, file_type)\
			SELECT *\
			FROM\
				(\
					(SELECT -1 AS device_id, g.id AS grid_id FROM grids AS g) temp0\
					CROSS JOIN\
					(SELECT 0 AS file_type UNION ALL SELECT 1 UNION ALL SELECT 2) temp1\
				);");
}

GridDataInfo Cache::load(size_t const device_id, size_t const grid_id, uint8_t const file_type)
{
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	auto genDataInfo = [&conn, device_id, grid_id, file_type] {
		// 그리드가 있으면 SELECT문으로 정보를 얻고 함수를 종료한다
		auto q = conn.query(sprn("\
			SELECT c.addr, c.byte\
			FROM cache AS c\
			WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d\
			AND c.state = 'EXIST';",
								 device_id,
								 grid_id,
								 file_type));

		auto result = q.store();

		DataInfo info;
		info.addr = (void *)strtol(result[0][i].c_str(), nullptr, 10);
		info.byte = strtol(result[0][i].c_str(), nullptr, 10);

		return info;
	};

	mysqlpp::Query q;

	// 일단 그리드가 있는지 없는지 모르지만 시도해본다 (선점)
	q = conn.query(sprn("\
		UPDATE IGNORE cache AS c\
		SET c.ref_count = c.ref_count + 1\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d\
			AND c.state = 'EXIST';",
						device_id,
						grid_id,
						file_type));

	if (q.affected_rows()) {
		return genGridDataInfo();
	}

	// 내가 찾는 그리드가 EXIST 상태가 아니면, LOADING으로 표기한다 (선점)
	// 부분적으로 올라온 그리드 (PARTIAL)이나 전혀 존재하지 않는 그리드 (NOTEXIST)의 경우에만 아래
	// 쿼리문이 작동한다
	q = conn.query(sprn("\
		UPDATE IGNORE cache AS c\
		SET c.state = 'LOADING'\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d\
			AND c.state = 'NOTEXIST';",
						device_id,
						grid_id,
						file_type));

	// 다른 쓰레드는, LOADING으로 표시되어 있기 때문에 affected_rows = 0 이 된다
	if (!q.affected_rows()) {
		// 선점을 하지 못했다면 ...
		// 내가 원하는 그리드가 EXIST가 될 때까지 계속 시도 해본다.
		while (true) {
			q = conn.query(sprn("\
				UPDATE IGNORE cache AS c\
				SET c.ref_count = c.ref_count + 1\
				WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d\
					AND c.state = 'EXIST';",
								device_id,
								grid_id,
								file_type));
			if (q.affected_rows()) {
				return genGridDataInfo();
			}
		}
	}

	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정
	/// 이쪽 위에까지는 거의 다 확인함!!!!!!!!!! 아래쪽부터 수정

	// 내가 해당 그리드에 대해 LOADING 권리를 선점했다면, 어떤 그리드를 불러와야 하는지 조사한다
	conn.query(sprn("\
			SELECT g.row_byte, g.ptr_byte g.col_byte, g.stem\
			FROM grids AS g\
			WHERE g.id = %ld;",
					grid_id),
			   result);

	// 아래와 같은 정보를 얻을 수 있다.
	GridDataInfo info;
	for (int i = 0; i < 3; i++) {
		info[i].addr = nullptr;
		info[i].byte = strtol(result[0][i].c_str(), nullptr, 10);
		info[i].path = this->folderPath / fs::path(result[0][3] + EXTENSION[i]);
	}

	// 할당을 할 것이므로 아래와 같은 함수 2개를 준비한다. CPU와 GPU할당 코드를 똑같이 만들기
	// 위함이다
	auto tryAlloc = [=](void ** address, size_t byte) {
		if (device_id >= 0) {
			// GPU일 경우
			try {
				// nvidia의 rmm은 thread-safe하다
				*address = this->device_pool[device_id]->allocate(byte);
			} catch (rmm::bad_alloc e) {
				return false;
			}
		} else {
			// CPU일 경우 boost pool은 thread-safe하다
			*address = malloc(byte);

			if (address == nullptr) {
				return false;
			}
		}
		return true;
	};

	auto mustDealloc = [=](void * address, size_t byte) {
		if (device_id >= 0) {
			// GPU일 경우
			this->device_pool[device_id]->deallocate(address, byte);
		} else {
			// CPU일 경우 boost pool은 thread-safe하다
			free(address);
		}
	};

	while (true) {
		// 할당을 시도해 본다. 다른 그리드를 요청하는 쓰레드들이 여기 있을 수 있다.
		// 파일 3개에 대해 모두 할당을 해야 한다.

		for (int i = 0; i < 3; i++) {
			if (tryAlloc(&info[i].addr, info[i].byte)) {
				while (true) {
					// 안쓰는 놈을 찾아서 표기한다. 현재 쫒아내는중이라고 표시 (선점)
					myConn.query(sprn("\
					UPDATE IGNORE cache AS c\
					SET c.state = 'EVICTING'\
					WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'EXIST' AND c.ref_count = 0\
					ORDER BY c.last_used ASC LIMIT 1;",
									  device_id,
									  grid_id));

					// 한놈이라도 찾았으면,
					if (mysql_affected_rows(myConn.conn)) {
						myConn.query(sprn("\
					SELECT c.row_addr, g.row_byte, c.ptr_addr, g.row_addr, c.col_byte, g.col_byte\
					FROM cache AS c, grids AS g\
					WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'EVICTING' AND c.grid_id = g.id;",
										  device_id,
										  grid_id),
									 result);

						GridDataInfo evict;
						for (int i = 0; i < 6; i++) {
							if (i % 2) {
								evict[i / 2].byte = strtol(result[0][i].c_str(), nullptr, 10);
							} else {
								evict[i / 2].addr =
									(void *)strtol(result[0][i].c_str(), nullptr, 10);
							}
						}
						return evict;

						// 실제로 랜덤하게 아무거나 선택해서 쫒아내봄
						// mustDealloc(evict.row.addr, evict.row.byte);

						// 존재 안한다고 표기
						myConn.query(sprn("\
					UPDATE IGNORE cache AS c\
					SET c.state = 'NOTEXIST'\
					WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'EVICTING';",
										  device_id,
										  grid_id));
						break;
					}
				}
				// 뭔가 쫒아냈으니 다시 시도한다
				continue;
			} else {
				// 성공했으면 표기하고 바로 빠져나간다
				myConn.query(sprn("\
				UPDATE IGNORE cache AS c\
				SET c.state = 'EXIST'\
				WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'LOADING';",
								  device_id,
								  grid_id));
				break;
			}

			myConn.query(sprn("\
				UPDATE IGNORE cache AS c\
				SET c.state = 'NOTEXIST'\
				WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'EXIST' AND c.ref_count = 0",
							  device_id,
							  grid_id));

			myConn.query(sprn("\
				UPDATE IGNORE cache AS c\
				SET c.state = 'EVICTING'\
				WHERE c.device_id = %ld AND c.grid_id = %ld ",
							  device_id,
							  grid_id));

			if (mysql_affected_rows(myConn.conn)) {
				break;
			}
		}
	}
}

void Cache::done(size_t const device_id, size_t const grid_id, uint8_t const file_type)
{
	MySQLConnection myConn;

	std::vector<std::vector<std::string>> result;

	myConn.query(sprn("\
		UPDATE cache AS c\
		SET c.ref_count = c.ref_count - 1\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.state = 'EXIST'",
					  device_id,
					  grid_id));
}