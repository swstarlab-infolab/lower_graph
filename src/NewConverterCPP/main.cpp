#include "main.h"

#include <chrono>

void init(Context & ctx, int argc, char * argv[])
{
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <inFolder> <outFolder> <outName>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	ctx.inFolder  = fs::absolute(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	ctx.outFolder = fs::absolute(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
	ctx.outName	  = std::string(argv[3]);
}

int main(int argc, char * argv[])
{
	Context ctx;
	init(ctx, argc, argv);

	auto start = std::chrono::system_clock::now();

	phase1(ctx);
	phase2(ctx);
	phase3(ctx);

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed = end - start;

	log("Total Elapsed time: " + std::to_string(elapsed.count()) + " (sec)");

	return 0;
}