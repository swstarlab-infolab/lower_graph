#ifndef BC2F77DF_5D94_41A8_98CC_36F417DB9A92
#define BC2F77DF_5D94_41A8_98CC_36F417DB9A92

#include "type.cuh"

#include <memory>
#include <tuple>

// first return = GPU channel, second return = CPU channel
std::pair<std::shared_ptr<bchan<Command>>, std::shared_ptr<bchan<Command>>>
ScheduleManager(Context const & ctx);

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes);

#endif /* BC2F77DF_5D94_41A8_98CC_36F417DB9A92 */