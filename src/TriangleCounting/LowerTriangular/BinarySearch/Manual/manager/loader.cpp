#include "../manager.h"
#include "../context.h"
#include "../channel.h"

namespace Manager {

template <size_t Size>
void loader(
    Context const & ctx,
    Channel<MessageType::LoadReq> & loadReq,
    std::array<Channel<MessageType::LoadRes>, Size> loadRes)
{

}

} // namespace Manager