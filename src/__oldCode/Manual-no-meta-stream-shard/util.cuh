#ifndef A15BFCFD_5CCE_49E4_BFF2_B4588665B4BA
#define A15BFCFD_5CCE_49E4_BFF2_B4588665B4BA

#include "type.cuh"

#include <regex>

fs::path csrPath(fs::path const & folder, GridIndex const & gidx, DataType const dataType);
bool	 csrExist(fs::path const & folder, fs::path const & stem);
size_t	 csrShardCount(fs::path const & folder, std::string const & target);

#endif /* A15BFCFD_5CCE_49E4_BFF2_B4588665B4BA */
