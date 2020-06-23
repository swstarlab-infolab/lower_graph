#include "main.h"

#include <GridCSR/GridCSR.h>

static auto filenameDecode(std::string const & in)
{
	GridIndex32 gidx32 = {0, 0};

	auto delimPos = in.find(__FilenameDelimiter);
	gidx32[0]	  = atoi(in.substr(0, delimPos).c_str());
	gidx32[1]	  = atoi(in.substr(delimPos + 1, in.size()).c_str());

	return gidx32;
}
// metadata writing
void phase3(Context const & ctx)
{
	auto metaDataPath = ctx.outFolder / ctx.outName / fs::path("meta.json");

	log("Metadata: " + metaDataPath.string() + " Start");

	GridCSR::MetaData m;

	m.dataname		= ctx.outName;
	m.extension.row = std::string(__OutFileExts[0]).substr(1, std::string(__OutFileExts[0]).size());
	m.extension.ptr = std::string(__OutFileExts[1]).substr(1, std::string(__OutFileExts[1]).size());
	m.extension.col = std::string(__OutFileExts[2]).substr(1, std::string(__OutFileExts[2]).size());

	m.info.width.row = m.info.width.col = __GridWidth;

	auto files = walk(ctx.outFolder / ctx.outName, ".row");

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

	log("Metadata: " + metaDataPath.string() + " Finished");
}