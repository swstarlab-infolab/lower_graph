#ifndef A8E117EE_1860_407A_8C05_55F81A83BD96
#define A8E117EE_1860_407A_8C05_55F81A83BD96

//#define throw(...)
#define MYSQLPP_MYSQL_HEADERS_BURIED

#include <mysql++/mysql++.h>
#include <string.h>

inline bool mysqlConnect(mysqlpp::Connection & conn)
{
	if (!conn.connect("GridCSR", "127.0.0.1", "root", "")) {
		return false;
	}
	return true;
}

inline size_t mysqlSingleValueSizeT(mysqlpp::Connection & conn, std::string const & queryString)
{
	auto q		= conn.query(queryString);
	auto result = q.store();
	return strtol(result[0][0].c_str(), nullptr, 10);
}

#endif /* A8E117EE_1860_407A_8C05_55F81A83BD96 */