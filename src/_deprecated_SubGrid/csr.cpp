#include "csr.h"

#include "parallel.h"
#include "string.h"

std::shared_ptr<csr4> csrLoad(fs::path const & folder, fs::path const & filename)
{
	auto out = std::make_shared<csr4>();

	std::array<std::string, 3> extension = {".row", ".ptr", ".col"};

	parallelFiber(3, [&](size_t const i) {
		auto target = fs::path(std::string(folder / filename) + extension[i]);
		out->at(i)	= fileLoad<Vertex>(target);
	});

	return out;
}

static void csrShow() {}

void csrStore(csr4 const & in)
{
	/*
	for (auto i = 0; i < out->size(); ++i) {
		auto target = fs::path(std::string(folder / filename) + extension[i]);
		out->at(i)	= fileLoad<Vertex>(target);
	}
	*/
}

std::shared_ptr<gcsrIndex> csrNameParse(std::string const & in)
{
	auto out = std::make_shared<gcsrIndex>();

	std::string delim = "-";
	size_t		pos	  = 0;

	auto temp = split(in, ",");

	switch (temp->size()) {
	case 3: {
		auto fin = split(temp->at(2), "-");
		for (size_t i = 0; i < fin->size(); i++) {
			out->sub[i] = strtol(fin->at(i).c_str(), nullptr, 10);
		}
		out->depth = strtol(temp->at(1).c_str(), nullptr, 10);
	}
	case 1: {
		auto fin = split(temp->at(0), "-");
		for (size_t i = 0; i < fin->size(); i++) {
			out->xy[i] = strtol(fin->at(i).c_str(), nullptr, 10);
		}
		break;
	}
	}
	return out;
}

std::string csrNameMake(gcsrIndex const & in)
{
	return std::to_string(in.xy[0]) + "-" + std::to_string(in.xy[1]) + "," +
		   std::to_string(in.depth) + "," + std::to_string(in.sub[0]) + "-" +
		   std::to_string(in.sub[1]);
}

std::shared_ptr<std::vector<fs::path>> csrScan(fs::path const & folder, size_t const criteria)
{
	auto out = std::make_shared<std::vector<fs::path>>();
	// recursive iteration
	for (fs::recursive_directory_iterator iter(folder), end; iter != end; iter++) {
		// check file is not directory and size is not zero
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
			if (extension[0] != iter->path().extension()) {
				continue;
			}

			if (criteria < csrByte(folder, iter->path().stem())) {
				out->push_back(iter->path().parent_path() / iter->path().stem());
			}
		}
	}

	return out;
}

size_t csrByte(fs::path const & folder, fs::path const & filename)
{
	size_t total = 0;

	for (size_t i = 0; i < 3; i++) {
		total += fs::file_size(fs::path(std::string(folder / filename) + extension[i]));
	}

	return total;
}
