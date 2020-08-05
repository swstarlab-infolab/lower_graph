#ifndef D5D78C16_48D8_4581_8C8D_E4336F18BDC9
#define D5D78C16_48D8_4581_8C8D_E4336F18BDC9

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <array>
#include <string>

#define GRIDWIDTH	(1UL << 24)
#define EXP_BITMAP0 (12UL)
#define EXP_BITMAP1 (5UL)

using Count					   = unsigned long long int;
char const * const EXTENSION[] = {".row", ".ptr", ".col"};

#endif /* D5D78C16_48D8_4581_8C8D_E4336F18BDC9 */
