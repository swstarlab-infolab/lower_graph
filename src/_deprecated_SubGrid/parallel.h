#ifndef F391978F_364E_4A64_8881_3D404369D0F4
#define F391978F_364E_4A64_8881_3D404369D0F4

#include <boost/fiber/all.hpp>
#include <functional>

#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

void parallelThread(size_t const workers, std::function<void(size_t const)> func);
void parallelFiber(size_t const workers, std::function<void(size_t const)> func);

#endif /* F391978F_364E_4A64_8881_3D404369D0F4 */
