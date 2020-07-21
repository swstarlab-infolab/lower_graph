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
	uint64_t maxVID			 = 0;
	uint64_t reorderType	 = 0;

	// Parse argument
	switch (argc) {
	case 7:
		maxVID		= (1 << strtol(argv[5], nullptr, 10));
		reorderType = strtol(argv[6], nullptr, 10);
	case 5:
		inFolder  = fs::absolute(fs::path(std::string(argv[1])));
		outFolder = fs::absolute(fs::path(std::string(argv[2]))) / fs::path(std::string(argv[3]));
		lowerTriangular = (strtol(argv[4], nullptr, 10) != 0);
		break;
	default:
		fprintf(
			stderr,
			"usage: \n"
			"%s <inFolder> <outFolder> <outName> <LowerTriangular>\n"
			"%s <inFolder> <outFolder> <outName> <LowerTriangular> <maxVIDexp> <reorderType> \n",
			argv[0],
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
		sp<std::vector<uint64_t>> reorderTable;
		if (reorderType > 0) {
			stopwatch("Stage0",
					  [&] { reorderTable = stage0(inFolder, outFolder, maxVID, reorderType); });
		}
		stopwatch("Stage1", [&] {
			stage1(
				inFolder, outFolder, (1 << 24), lowerTriangular, (reorderType > 0), reorderTable);
		});
		stopwatch("Stage2", [&] { stage2(outFolder, outFolder); });
	});

	// Finish procedure
	log(std::string(inFolder) + "->" + std::string(outFolder) +
		", reorder type: " + std::to_string(reorderType) + ", completed");

	return 0;
}