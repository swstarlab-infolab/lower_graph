#ifndef A8E117EE_1860_407A_8C05_55F81A83BD96
#define A8E117EE_1860_407A_8C05_55F81A83BD96

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
	return strtol(conn.query(queryString).store()[0][0].c_str(), nullptr, 10);
}

#endif /* A8E117EE_1860_407A_8C05_55F81A83BD96 */