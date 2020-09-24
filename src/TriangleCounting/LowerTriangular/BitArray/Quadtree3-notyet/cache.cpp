#include "cache.h"

#include "logging.h"
#include "util.h"

#include <cuda_runtime.h>
#include <fcntl.h>

#define STATE_NOTEXIST 0
#define STATE_LOADING  1
#define STATE_EXIST	   2

void Cache::init()
{
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	auto q = conn.query("\
		CREATE TABLE IF NOT EXISTS cache (\
			device_id INT NOT NULL,\
			grid_id INT NOT NULL,\
			file_type TINYINT DEFAULT NULL,\
			state TINYINT DEFAULT 0,\
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
	assert(q.exec());

	// GPU cache pool init
	this->device_pool.resize(this->gpus);
	this->cudaStream.resize(this->gpus);

	for (int i = 0; i < this->gpus; i++) {
		cudaSetDevice(i);
		cudaDeviceReset();

		size_t free;
		cudaMemGetInfo(&free, nullptr);
		free -= (1UL << 29);

		auto * mr			 = rmm::mr::get_current_device_resource();
		this->device_pool[i] = std::make_shared<DevicePoolType>(mr, free, free);
		rmm::mr::set_current_device_resource(&(*(this->device_pool[i])));

		cudaStreamCreate(&this->cudaStream[i]);
	}

	// GPU cache metadata init
	for (int i = -1; i < this->gpus; i++) {
		q << "INSERT IGNORE INTO cache (device_id, grid_id, file_type, byte) "
		  << "SELECT " << i << ", g.id, 0, g.row_byte FROM grids AS g;";
		assert(q.exec());

		q << "INSERT IGNORE INTO cache (device_id, grid_id, file_type, byte) "
		  << "SELECT " << i << ", g.id, 1, g.ptr_byte FROM grids AS g;";
		assert(q.exec());

		q << "INSERT IGNORE INTO cache (device_id, grid_id, file_type, byte) "
		  << "SELECT " << i << ", g.id, 2, g.col_byte FROM grids AS g;";
		assert(q.exec());
	}
}

Cache::~Cache()
{
	for (int i = 0; i < this->gpus; i++) {
		cudaStreamDestroy(this->cudaStream[i]);
	}
}

DataInfo Cache::genDataInfo(mysqlpp::Connection & conn, MemReqInfo const & reqInfo, int const state)
{
	// 그리드가 있으면 SELECT문으로 정보를 얻고 함수를 종료한다
	auto q = conn.query(sprn("\
			SELECT c.addr, c.byte\
			FROM cache AS c\
			WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d AND c.state = %d;",
							 reqInfo.device_id,
							 reqInfo.grid_id,
							 reqInfo.file_type,
							 state));

	auto result = q.store();

	DataInfo info;
	info.addr = (void *)strtol(result[0][0].c_str(), nullptr, 10);
	info.byte = strtol(result[0][1].c_str(), nullptr, 10);

	return info;
}
bool Cache::changeState(mysqlpp::Connection & conn,
						MemReqInfo const &	  reqInfo,
						int const			  state_from,
						int const			  state_to)
{
	auto q = conn.query(sprn("\
		UPDATE IGNORE cache AS c\
		SET c.state = %d\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d AND c.state = %d;",
							 state_to,
							 reqInfo.device_id,
							 reqInfo.grid_id,
							 reqInfo.file_type,
							 state_from));
	assert(q.exec());

	return q.affected_rows() > 0;
}

bool Cache::refCountUpForExist(mysqlpp::Connection & conn, MemReqInfo const & reqInfo)
{
	auto q = conn.query(sprn("\
		UPDATE IGNORE cache AS c\
		SET c.ref_count = c.ref_count + 1\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d AND c.state = %d;",
							 reqInfo.device_id,
							 reqInfo.grid_id,
							 reqInfo.file_type,
							 STATE_EXIST));
	assert(q.exec());

	return q.affected_rows() > 0;
}

DataInfo Cache::load(MemReqInfo const & reqInfo)
{
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	// 일단 그리드가 있는지 없는지 모르지만 시도해본다 (선점)
	// LOGF("devID=%d, gridID=%ld, file_type=%d try refCountUpForExist", reqInfo.device_id,
	// reqInfo.grid_id, reqInfo.file_type);
	if (this->refCountUpForExist(conn, reqInfo)) {
		// LOGF("devID=%d, gridID=%ld, file_type=%d try refCountUpForExist Success",
		// reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);
		// refcount 올리는데에 성공했으면 정보를 반환하고 끝낸다.
		return this->genDataInfo(conn, reqInfo, STATE_EXIST);
	}

	// 내가 찾는 그리드가 NOTEXIST면, LOADING으로 표기한다 (선점)
	// LOGF("devID=%d, gridID=%ld, file_type=%d try changeState NOTEXIST->LOADING",
	// reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);
	if (!this->changeState(conn, reqInfo, STATE_NOTEXIST, STATE_LOADING)) {
		// 만약에 선점에 실패했다면, EXIST 상태가 될 때까지 refCountUp 시도
		// LOGF("devID=%d, gridID=%ld, file_type=%d wait until become EXIST", reqInfo.device_id,
		// reqInfo.grid_id, reqInfo.file_type);

		while (true) {
			if (this->refCountUpForExist(conn, reqInfo)) {
				// refCount 올리는데 성공했으면 정보 반환.
				// LOGF("devID=%d, gridID=%ld, file_type=%d wait until become EXIST finish",
				// reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);
				return this->genDataInfo(conn, reqInfo, STATE_EXIST);
			}
		}
	}

	// LOGF("devID=%d, gridID=%ld, file_type=%d try changeState NOTEXIST->LOADING success",
	// reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);

	// 내가 찾는 그리드가 LOADING이면, 정보를 받아온다.
	auto info = this->genDataInfo(conn, reqInfo, STATE_LOADING);

	auto q = conn.query(sprn("SELECT g.stem FROM grids AS g WHERE g.id = %ld;", reqInfo.grid_id));

	// 아래와 같은 정보를 얻을 수 있다.
	info.path = this->folderPath /
				fs::path(std::string(q.store()[0][0].c_str()) + EXTENSION[reqInfo.file_type]);

	// LOGF("devID=%d, gridID=%ld, file_type=%d filepath is = %s", reqInfo.device_id,
	// reqInfo.grid_id, reqInfo.file_type, info.path.c_str());

	// 할당을 할 것이므로 아래와 같은 함수 2개를 준비한다.
	// CPU와 GPU할당 코드를 똑같이 만들기 위함이다
	auto tryAlloc = [=](void ** address, size_t byte) {
		if (reqInfo.device_id < 0) {
			// CPU일 경우 boost pool은 thread-safe하다
			*address = malloc(byte);

			if (address == nullptr) {
				return false;
			}
		} else {
			// GPU일 경우
			try {
				// nvidia의 rmm은 thread-safe하다
				*address = this->device_pool[reqInfo.device_id]->allocate(byte);
			} catch (rmm::bad_alloc e) {
				return false;
			}
			// LOGF("alloc %p, %ld", address, byte);
		}
		return true;
	};

	auto mustDealloc = [=](void * address, size_t byte) {
		if (reqInfo.device_id < 0) {
			free(address);
		} else {
			// GPU일 경우
			// LOGF("dealloc %p, %ld", address, byte);
			this->device_pool[reqInfo.device_id]->deallocate(address, byte);
		}
	};

	auto loadSSDtoCPU = [=](void * to, size_t byte) {
		// cpu
		auto const __CDEF = 1UL << 26;

		auto fp = open64(info.path.c_str(), O_RDONLY);

		uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
		uint64_t offset	   = 0;

		while (offset < byte) {
			chunkSize = (byte - offset > chunkSize) ? chunkSize : byte - offset;
			auto b	  = read(fp, &(((uint8_t *)to)[offset]), chunkSize);
			offset += b;
		}

		close(fp);
	};

	// utilize OS's page cache
	auto loadSSDtoGPU = [=](void * to, size_t byte) {
		// cpu
		auto const __CDEF = 1UL << 26;

		auto	  fp   = open64(info.path.c_str(), O_RDONLY);
		uint8_t * temp = (uint8_t *)malloc(byte);

		uint64_t chunkSize = (byte < __CDEF) ? byte : __CDEF;
		uint64_t offset	   = 0;

		cudaSetDevice(reqInfo.device_id);

		// overlapped loading
		while (offset < byte) {
			chunkSize = (byte - offset > chunkSize) ? chunkSize : byte - offset;

			void * cpuPtr = &(((uint8_t *)temp)[offset]);
			void * gpuPtr = &(((uint8_t *)to)[offset]);

			// read chunk from SSD to CPU
			auto b = read(fp, cpuPtr, chunkSize);

			// read chunk from CPU to GPU
			// asynchronous
			cudaMemcpyAsync(gpuPtr,
							cpuPtr,
							chunkSize,
							cudaMemcpyHostToDevice,
							this->cudaStream[reqInfo.device_id]);

			offset += b;
		}

		close(fp);

		cudaStreamSynchronize(this->cudaStream[reqInfo.device_id]);

		free(temp);
	};

	// Allocation
	while (true) {
		// 할당을 시도해 본다.
		if (!tryAlloc(&info.addr, info.byte)) {
			// 할당에 실패하면...
			while (true) {
				// 어쩔 수 없이 락을 먼저 해야 함. 여러개의 evicting 중에서 이 함수에서 딱 정한
				// 것만 골라내야 하기 때문
				q = conn.query("START TRANSACTION;");
				assert(q.exec());

				q = conn.query(sprn("\
					SELECT c.addr, c.byte, c.grid_id, c.file_type\
					FROM cache AS c\
					WHERE c.device_id = %ld AND c.state = %d AND c.ref_count = 0\
					LIMIT 1 FOR UPDATE;",
									reqInfo.device_id,
									STATE_EXIST));
				// ORDER BY c.last_used ASC LIMIT 1

				auto evictResult = q.store();

				if (evictResult.size() != 0) {
					auto evict = DataInfo{};
					evict.addr = (void *)s2l(evictResult[0][0]);
					evict.byte = s2l(evictResult[0][1]);

					// LOGF("devID=%d, gridID=%ld, file_type=%d gonna evict gridID=%s, "
					// "file_type=%s", reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type,
					// evictResult[0][2].c_str(), evictResult[0][3].c_str());

					// 실제로 랜덤하게 아무거나 선택해서 쫒아내봄
					mustDealloc(evict.addr, evict.byte);

					q = conn.query(sprn("\
					UPDATE IGNORE cache AS c\
					SET c.state = %d, c.addr = NULL\
					WHERE c.device_id = %ld AND c.state = %d AND c.ref_count = 0 AND c.addr = %ld AND c.byte = %ld;",
										STATE_NOTEXIST,
										reqInfo.device_id,
										STATE_EXIST,
										evict.addr,
										evict.byte));
					assert(q.exec());

					q = conn.query("COMMIT;");
					assert(q.exec());

					break;
				}

				q = conn.query("ROLLBACK;");
				assert(q.exec());
			}
			// 뭔가 쫒아냈으니 다시 시도한다
			continue;
		} else {
			break;
		}
	}

	// 성공했으면 로딩하고 바로 빠져나간다

	if (reqInfo.device_id < 0) {
		loadSSDtoCPU(info.addr, info.byte);
	} else {
		loadSSDtoGPU(info.addr, info.byte);
	}

	// LOGF("devID=%d, gridID=%ld, file_type=%d loading complete, gonna change state
	// LOADING->EXIST", reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);

	q = conn.query(sprn("\
		UPDATE IGNORE cache AS c\
		SET c.state = %d, c.ref_count = 1, c.addr = %ld, c.byte = %ld\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d AND c.state = %d;",
						STATE_EXIST,
						info.addr,
						info.byte,
						reqInfo.device_id,
						reqInfo.grid_id,
						reqInfo.file_type,
						STATE_LOADING));
	assert(q.exec());

	// LOGF("devID=%d, gridID=%ld, file_type=%d loading complete, gonna change state
	// LOADING->EXIST " "complete", reqInfo.device_id, reqInfo.grid_id, reqInfo.file_type);

	return this->genDataInfo(conn, reqInfo, STATE_EXIST);
}

void Cache::done(MemReqInfo const & reqInfo)
{
	mysqlpp::Connection conn;
	mysqlConnect(conn);

	auto q = conn.query(sprn("\
		UPDATE cache AS c\
		SET c.ref_count = c.ref_count - 1\
		WHERE c.device_id = %ld AND c.grid_id = %ld AND c.file_type = %d AND c.state = %d;",
							 reqInfo.device_id,
							 reqInfo.grid_id,
							 reqInfo.file_type,
							 STATE_EXIST));
	assert(q.exec());
}