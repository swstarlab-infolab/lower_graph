#include "stage.h"
#include "util.h"

#include <cstdio>
#include <string>
#include <thread>

int main(int argc, char * argv[])
{
	// Variables
	fs::path inFolder, outFolder;
	bool	 lowerTriangular = false;

	// Parse argument
	switch (argc) {
	case 5:
		inFolder  = fs::absolute(fs::path(std::string(argv[1])));
		outFolder = fs::absolute(fs::path(std::string(argv[2]))) / fs::path(std::string(argv[3]));
		lowerTriangular = (strtol(argv[4], nullptr, 10) != 0);
		break;
	default:
		fprintf(stderr,
				"usage: \n"
				"%s <inFolder> <outFolder> <outName> <LowerTriangular>\n",
				argv[0]);
		exit(EXIT_FAILURE);
	}

	// Create output folder
	if (!fs::exists(outFolder)) {
		if (!fs::create_directories(outFolder)) {
			fprintf(stderr, "failed to create folder: %s\n", outFolder.c_str());
			exit(EXIT_FAILURE);
		}
	}

	// Start procedure
	stopwatch("Total Procedure", [&] {
		sp<std::vector<uint64_t>> relabelTable;
		stopwatch("Stage1", [&] { stage1(inFolder, outFolder, (1 << 24), lowerTriangular); });
		stopwatch("Stage2", [&] { stage2(outFolder, outFolder); });
	});

	// Finish procedure
	log(std::string(inFolder) + "->" + std::string(outFolder) + ", completed");

	return 0;
}