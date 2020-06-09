#ifndef E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83
#define E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83

#include "type.h"

#include <memory>

std::shared_ptr<bchan<CommandResult>>
ExecutionManager(Context & ctx, int myID, std::shared_ptr<bchan<Command>> in);

#endif /* E31BFF48_8BC2_4E2F_B7C1_BAB762E43F83 */
