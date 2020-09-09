#include "logging.h"
#include "shard.h"
#include "type.h"

#include <assert.h>
#include <memory>
#include <mysql/mysql.h>
#include <stdio.h>
#include <string>

template <typename... Args>
std::string string_format(const std::string & format, Args... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size <= 0) {
		throw std::runtime_error("Error during formatting.");
	}
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

int main(int argc, char * argv[])
{
	if (argc != 5) {
		fprintf(stderr, "usage: %s <folderPath> <streams> <blocks> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	auto folderPath = fs::path(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");

	MYSQL mysql;
	ASSERT_ERRNO(mysql_init(&mysql) != nullptr);

	struct {
		char const * addr = "127.0.0.1";
		char const * id	  = "root";
		char const * pw	  = "";
		char const * name = "GridCSR";
		int			 port = 3306;
	} db;

	auto * conn = mysql_real_connect(&mysql, db.addr, db.id, db.pw, db.name, db.port, nullptr, 0);
	ASSERT_ERRNO(conn != nullptr);
	ASSERT_ERRNO(mysql_select_db(&mysql, db.name) == 0);

	{
		auto query = string_format("\
		CREATE TABLE IF NOT EXISTS grid_info (\
			id INT NOT NULL AUTO_INCREMENT,\
			row_idx INT NOT NULL,\
			col_idx INT NOT NULL,\
			depth INT NOT NULL,\
			shard_row_idx INT NOT NULL,\
			shard_col_idx INT NOT NULL,\
			row_file_byte BIGINT NOT NULL,\
			ptr_file_byte BIGINT NOT NULL,\
			col_file_byte BIGINT NOT NULL,\
			PRIMARY KEY (id),\
			UNIQUE KEY (row_idx, col_idx, depth, shard_row_idx, shard_col_idx)\
			)\
		");
		assert(mysql_query(conn, query.c_str()) == 0);
	}

	{
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
				sRange.range[0][0]

					auto query = string_format("\
				INSERT IGNORE INTO grid_info (\
					row_idx, col_idx, depth, shard_row_idx, shard_col_idx, row_file_byte, ptr_file_byte, col_file_byte)\
					VALUES (\
					%d, %d, %d, %d, %d, %ld, %ld, %ld)\
				",
											   sIdx.grid[0],
											   sIdx.grid[1],
											   sIdx.depth,
											   sIdx.shard[0],
											   sIdx.shard[1],
											   row_file_byte,
											   ptr_file_byte,
											   col_file_byte);
				ASSERT_ERRNO(mysql_query(conn, query.c_str()) == 0);
			}
		}
	}

	size_t total_rows = 0;

	{
		auto query = string_format("SELECT COUNT(*) FROM grid_info");
		assert(mysql_query(conn, query.c_str()) == 0);

		auto *	  sql_result = mysql_store_result(conn);
		MYSQL_ROW sql_row;
		while ((sql_row = mysql_fetch_row(sql_result)) != NULL) {
			total_rows = strtol(sql_row[0], nullptr, 10);
		}
		mysql_free_result(sql_result);
	}

	size_t criteria = 0;
	{
		auto query = string_format(
			"SELECT col_file_byte FROM grid_info ORDER BY col_file_byte DESC LIMIT %ld, 1",
			total_rows / 5);
		assert(mysql_query(conn, query.c_str()) == 0);

		auto *	  sql_result = mysql_store_result(conn);
		MYSQL_ROW sql_row;
		while ((sql_row = mysql_fetch_row(sql_result)) != NULL) {
			criteria = strtol(sql_row[0], nullptr, 10);
		}
		mysql_free_result(sql_result);
	}

	printf("%ld\n", criteria);

	/*
	query = string_format(
		"SELECT row_idx, col_idx, depth, shard_row_idx, shard_col_idx FROM grid_info");
	ASSERT_ERRNO(mysql_query(conn, query.c_str()) == 0);

	auto * sql_result = mysql_store_result(conn);

	MYSQL_ROW sql_row;
	while ((sql_row = mysql_fetch_row(sql_result)) != NULL) {
		std::string key;
		if (sql_row[2] == "0") {
			key = string_format("%s-%s\n", sql_row[0], sql_row[1]);
		} else {
			key = string_format(
				"%s-%s,%s,%s-%s\n", sql_row[0], sql_row[1], sql_row[2], sql_row[3], sql_row[4]);
		}
		LOG(key.c_str());
	}
	mysql_free_result(sql_result);
	*/

	mysql_close(conn);

	return 0;
}