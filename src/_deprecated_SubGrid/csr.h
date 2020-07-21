#ifndef E2ED33E8_81C9_4F4B_A20B_F8A8B1CDB6C5
#define E2ED33E8_81C9_4F4B_A20B_F8A8B1CDB6C5

#include "file.h"
#include "type.h"

#include <string>

struct gcsrIndex {
	std::array<size_t, 2> xy = {
		0,
	};

	std::array<size_t, 2> sub = {
		0,
	};

	size_t depth = 0;
};

using csr4 = std::array<std::shared_ptr<std::vector<uint32_t>>, 3>;

std::shared_ptr<csr4> csrLoad(fs::path const & folder, fs::path const & filename);
void				  csrStore(fs::path const & folder, fs::path const & filename, csr4 const & in);
std::shared_ptr<gcsrIndex>			   gcsrNameParse(std::string const & in);
std::string							   gcsrNameMake(gcsrIndex const & in);
std::shared_ptr<std::vector<fs::path>> csrScan(fs::path const & folder, size_t const criteria);
size_t								   csrByte(fs::path const & folder, fs::path const & filename);

#endif /* E2ED33E8_81C9_4F4B_A20B_F8A8B1CDB6C5 */
