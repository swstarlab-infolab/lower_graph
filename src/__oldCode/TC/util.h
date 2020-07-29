#ifndef D6E7ABAE_5F76_4213_A037_8E9C14D387A9
#define D6E7ABAE_5F76_4213_A037_8E9C14D387A9

#include <functional>
#include <stddef.h>
#include <string>

void log(std::string const & s);
void stopwatch(std::string const & message, std::function<void()> function);

char		OX(bool cond);
std::string SIUnit(size_t const byte);
void		parallelFiber(size_t const workers, std::function<void(size_t const)> func);
void		parallelThread(size_t const workers, std::function<void(size_t const)> func);
size_t		ceil(size_t const x, size_t const y);

#endif /* D6E7ABAE_5F76_4213_A037_8E9C14D387A9 */
