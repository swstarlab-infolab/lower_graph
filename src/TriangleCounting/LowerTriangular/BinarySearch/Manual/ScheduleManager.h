#ifndef BC2F77DF_5D94_41A8_98CC_36F417DB9A92
#define BC2F77DF_5D94_41A8_98CC_36F417DB9A92

#include "type.h"

#include <memory>

std::shared_ptr<bchan<Command>> ScheduleManager(Context const & ctx);

void ScheduleWaiter(std::shared_ptr<bchan<CommandResult>> executionRes);

#endif /* BC2F77DF_5D94_41A8_98CC_36F417DB9A92 */