#ifndef BC2F77DF_5D94_41A8_98CC_36F417DB9A92
#define BC2F77DF_5D94_41A8_98CC_36F417DB9A92

#include "type.cuh"

#include <memory>

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

std::shared_ptr<bchan<Command>> ScheduleManager(Context const & ctx);

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes);

#endif /* BC2F77DF_5D94_41A8_98CC_36F417DB9A92 */