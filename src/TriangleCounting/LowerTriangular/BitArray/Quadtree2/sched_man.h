#ifndef C50D9C38_16EF_43BA_87C0_E2C6D98AE2D2
#define C50D9C38_16EF_43BA_87C0_E2C6D98AE2D2

#include "type.h"

#include <array>
#include <boost/fiber/buffered_channel.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Sched
{

using XY  = std::array<uint32_t, 2>;
using XY3 = std::array<XY, 3>;

struct Hash {
	std::size_t operator()(XY const & k) const
	{
		auto a = std::hash<uint32_t>{}(k[0]);
		auto b = std::hash<uint32_t>{}(k[1]);
		return a ^ b;
	}
};

struct Equal {
	bool operator()(XY const & kl, XY const & kr) const { return kl == kr; }
};

struct ShardIndex {
	XY			grid, shard;
	uint32_t	depth;
	std::string string() const;
	bool		parse(std::string const & in);
};

struct ShardRange {
	uint32_t		  depth;
	std::array<XY, 2> range; // [x_s,x_t),[y_s,y_t)
	void			  conv(ShardIndex const & in);
	bool			  increase(uint32_t const depth);
};

using EntryType = std::unordered_map<XY, std::vector<ShardIndex>, Hash, Equal>;
using JobType	= std::array<ShardIndex, 3>;
using OutType	= std::shared_ptr<boost::fibers::buffered_channel<JobType>>;

class Manager
{
private:
	EntryType entry;
	bool	  checkAvail(JobType const & in);

	fs::path folderPath;

public:
	void	run();
	OutType out;
	Manager(fs::path const & folder);
	~Manager();
};
} // namespace Sched

#endif /* C50D9C38_16EF_43BA_87C0_E2C6D98AE2D2 */