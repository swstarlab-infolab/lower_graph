#ifndef FDAF3505_EC06_40B5_8DC4_D476BA62D49C
#define FDAF3505_EC06_40B5_8DC4_D476BA62D49C

#include <errno.h>
#include <mutex>
#include <stdio.h>
#include <string.h>

std::string currTimeDate();

//#define DEBUG

#ifdef DEBUG
#define LOG(str)                                                                            \
	do {                                                                                    \
		fprintf(stdout, "%s %s:%d: %s\n", currTimeDate().c_str(), __FILE__, __LINE__, str); \
	} while (0)
#define LOGF(fmt, ...)                  \
	do {                                \
		fprintf(stdout,                 \
				"%s %s:%d: " fmt "\n",  \
				currTimeDate().c_str(), \
				__FILE__,               \
				__LINE__,               \
				__VA_ARGS__);           \
	} while (0)
#define ERR(str)                                                                            \
	do {                                                                                    \
		fprintf(stderr, "%s %s:%d: %s\n", currTimeDate().c_str(), __FILE__, __LINE__, str); \
	} while (0)
#define ERRF(fmt, ...)                  \
	do {                                \
		fprintf(stderr,                 \
				"%s %s:%d: " fmt "\n",  \
				currTimeDate().c_str(), \
				__FILE__,               \
				__LINE__,               \
				__VA_ARGS__);           \
	} while (0)
#else
#define LOG(str)                                                 \
	do {                                                         \
		fprintf(stdout, "%s %s\n", currTimeDate().c_str(), str); \
	} while (0)
#define LOGF(fmt, ...)                                                        \
	do {                                                                      \
		fprintf(stdout, "%s " fmt "\n", currTimeDate().c_str(), __VA_ARGS__); \
	} while (0)
#define ERR(str)                                                 \
	do {                                                         \
		fprintf(stderr, "%s %s\n", currTimeDate().c_str(), str); \
	} while (0)
#define ERRF(fmt, ...)                                                        \
	do {                                                                      \
		fprintf(stderr, "%s " fmt "\n", currTimeDate().c_str(), __VA_ARGS__); \
	} while (0)
#endif

#define ASSERT_ERRNO(x)                                   \
	do {                                                  \
		if (!(x)) {                                       \
			ERRF("errno:%d, %s", errno, strerror(errno)); \
			exit(EXIT_FAILURE);                           \
		}                                                 \
	} while (0);
#endif /* FDAF3505_EC06_40B5_8DC4_D476BA62D49C */
