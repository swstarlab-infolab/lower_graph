#ifndef BD43D1D7_5287_4591_98CA_FABEBDD76E44
#define BD43D1D7_5287_4591_98CA_FABEBDD76E44

#include "context.h"
#include "type.h"

namespace Sched
{
using XY  = std::array<uint32_t, 2>;
using XY3 = std::array<XY, 3>;

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

class Manager
{
public:
	sp<bchan<Job>> run(sp<Context> ctx);
};

} // namespace Sched

#endif /* BD43D1D7_5287_4591_98CA_FABEBDD76E44 */
