#ifndef D966AE75_D60E_4008_BA68_BD71C9F1F85C
#define D966AE75_D60E_4008_BA68_BD71C9F1F85C

#include <functional>
#include <stdlib.h>

size_t ceil(size_t const x, size_t const y);
void   pThread(size_t const workers, std::function<void(size_t const)> func);

#endif /* D966AE75_D60E_4008_BA68_BD71C9F1F85C */
