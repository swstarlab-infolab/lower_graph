#include "csr.h"
#include "file.h"
#include "parallel.h"
#include "string.h"
#include "type.h"

#include <tuple>

static auto ceil(size_t const x, size_t const y) { return (x != 0L) ? (1L + ((x - 1L) / y)) : 0L; }

static auto csrShard(std::shared_ptr<csr4> csr, size_t const maxShardByte)
{
	auto & row = *csr->at(0);
	auto & ptr = *csr->at(1);
	auto & col = *csr->at(2);

	std::vector<std::array<std::pair<size_t, size_t>, 3>> cutPos;

	size_t latest = 0;
	for (size_t i = 1; i < ptr.size(); i++) {
		size_t byte = (i - latest) * sizeof(uint32_t) +			 // row size
					  (i - latest + 1) * sizeof(uint32_t) +		 // ptr size
					  (ptr[i] - ptr[latest]) * sizeof(uint32_t); // col size

		if (byte > maxShardByte) {
			printf("row: [%ld,%ld)\n", latest, i - 1);
			latest = i - 1;
		}
	}
}

int main(int argc, char * argv[])
{
	if (argc != 2) {
		fprintf(stderr,
				"usage: \n"
				"%s <folder> <max capacity>\n",
				argv[0]);
		exit(EXIT_FAILURE);
	}

	fs::path folder = fs::absolute(fs::path(std::string(argv[1])));

	// over 1GB;
	auto criteria	  = 1L << 30;
	auto maxShardByte = csrScan(folder, criteria);

	for (auto & p : *scanned) {
		printf("%s: %s\n", p.c_str(), unit(csrByte(p.parent_path(), p.stem())).c_str());
	}

	for (auto & p : *scanned) {
		auto csr = csrLoad(p.parent_path(), p.stem());
		csrShard(csr, maxShardByte);
	}

	return 0;
}