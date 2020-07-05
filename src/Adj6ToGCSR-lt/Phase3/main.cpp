#include "../main.h"

#include <GridCSR/GridCSR.h>
#include <chrono>

static void routine(Context const & ctx)
{
	auto metaDataPath = ctx.outFolder / fs::path("meta.json");

	GridCSR::MetaData m;

	m.dataname		= ctx.outName;
	m.extension.row = std::string(__OutFileExts[0]).substr(1, std::string(__OutFileExts[0]).size());
	m.extension.ptr = std::string(__OutFileExts[1]).substr(1, std::string(__OutFileExts[1]).size());
	m.extension.col = std::string(__OutFileExts[2]).substr(1, std::string(__OutFileExts[2]).size());

	m.info.width.row = m.info.width.col = __GridWidth;

	auto files = walk(ctx.outFolder, ".row");

	uint32_t maxRow = 0, maxCol = 0;
	for (auto & f : *files) {
		auto gidx32 = filenameDecode(f.stem().string());
		maxRow		= (maxRow > gidx32[0]) ? maxRow : gidx32[0];
		maxCol		= (maxCol > gidx32[1]) ? maxCol : gidx32[1];
	}
	m.info.count.row = maxRow + 1;
	m.info.count.col = maxCol + 1;

	m.info.max_vid = 0;

	m.grid.each.resize(files->size());

	size_t i = 0;
	for (auto & f : *files) {
		auto gidx32 = filenameDecode(f.stem().string());

		m.grid.each[i].name		 = f.stem().string();
		m.grid.each[i].index.row = gidx32[0];
		m.grid.each[i].index.col = gidx32[1];
		i++;
	}

	m.Save(metaDataPath);

	log("Phase 3 (Json Metadata) " + metaDataPath.string() + " Written");
}

static void init(Context & ctx, int argc, char * argv[])
{
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <Folder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	ctx.inFolder  = fs::absolute(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	ctx.outFolder = ctx.inFolder;
	ctx.outName	  = ctx.outFolder.parent_path().stem().string();
}

int main(int argc, char * argv[])
{
	Context ctx;
	init(ctx, argc, argv);

	{
		auto start = std::chrono::system_clock::now();

		routine(ctx);

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		log("Phase 3 (Json Metadata) Complete, Elapsed Time: " + std::to_string(elapsed.count()) +
			" (sec)");
	}

	return 0;
}