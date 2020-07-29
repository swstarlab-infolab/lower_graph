#ifndef F7EBC687_3A79_4A2C_B01C_52E84F0EC479
#define F7EBC687_3A79_4A2C_B01C_52E84F0EC479

#include "type.h"

#include <cuda_runtime.h>
#include <vector>

struct Context {
	bool	 verbose = false;
	fs::path inFolder;
	size_t	 streams, blocks, threads;

	std::array<std::string, 3> extension = {".row", ".ptr", ".col"};

	std::vector<cudaDeviceProp> cuda;

	void parse(int argc, char * argv[]);

	void printCUDAInfo();
};

#endif /* F7EBC687_3A79_4A2C_B01C_52E84F0EC479 */
