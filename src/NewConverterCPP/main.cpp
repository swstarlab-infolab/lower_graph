#include <stdio.h>

#if __GNUC__ < 8
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <GridCSR/GridCSR.h>
#include <array>
#include <boost/fiber/all.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

// shorten long name
#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

#define WORDSZ 6
#define CHANSZ 16
#define GWIDTH (1 << 24)

// Primitive Types
using Vertex32		  = uint32_t;
using Vertex64		  = uint64_t;
using Edge32		  = std::array<Vertex32, 2>;
using Edge64		  = std::array<Vertex64, 2>;
using GridIndex32	  = std::array<uint32_t, 2>;
using GridAndEdge	  = std::pair<GridIndex32, Edge32>;
using GridAndEdgeList = std::vector<GridAndEdge>;
using RawData		  = std::vector<uint8_t>;
using FileList		  = std::vector<fs::path>;

struct KeyHash {
	std::size_t operator()(GridIndex32 const & k) const
	{
		auto a = std::hash<uint64_t>{}(uint64_t(k[0]) << 32);
		auto b = std::hash<uint64_t>{}(uint64_t(k[1]));
		return a ^ b;
	}
};

struct KeyEqual {
	bool operator()(GridIndex32 const & kl, GridIndex32 const & kr) const
	{
		return (kl[0] == kr[0] && kl[1] == kr[1]);
	}
};

using ShuffleMap =
	std::unordered_map<GridIndex32, std::shared_ptr<bchan<GridAndEdgeList>>, KeyHash, KeyEqual>;

struct Context {
	fs::path	inFolder;
	fs::path	outFolder;
	std::string outName;
};

struct SplittedRawData {
	Vertex64  src;
	size_t	  cnt;
	uint8_t * dst;
};

std::string filename(GridIndex32 in) { return std::to_string(in[0]) + "-" + std::to_string(in[1]); }

uint64_t convBE6toLE8(uint8_t * in, size_t start)
{
	return (uint64_t(in[start]) << (8 * 5)) + (uint64_t(in[start]) << (8 * 4)) +
		   (uint64_t(in[start]) << (8 * 3)) + (uint64_t(in[start]) << (8 * 2)) +
		   (uint64_t(in[start]) << (8 * 1)) + (uint64_t(in[start]) << (8 * 0));
}

std::string filename(GridIndex32 in) { return std::to_string(in[0]) + "-" + std::to_string(in[1]); }

auto walk(Context & ctx)
{
	auto out = std::make_shared<FileList>();
	for (fs::recursive_directory_iterator iter(ctx.inFolder), end; iter != end; iter++) {
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path().filename()) != 0) {
			out->push_back(iter->path().filename());
		}
	}
	return out;
}

auto load(fs::path inFile)
{
	std::ifstream f(inFile, std::ios::binary);

	auto fbyte = fs::file_size(inFile);
	auto out   = std::make_shared<RawData>(fbyte);

	f.read((char *)(out->data()), fbyte);

	return out;
}

auto split(std::shared_ptr<RawData> in)
{
	auto out = std::make_shared<bchan<SplittedRawData>>(CHANSZ);
	std::thread([&] {
		for (size_t i = 0; i < in->size();) {
			auto src = convBE6toLE8(in->data(), i);
			i += WORDSZ;
			auto cnt = convBE6toLE8(in->data(), i);
			i += WORDSZ;

			SplittedRawData sRawData;
			sRawData.src = src;
			sRawData.cnt = cnt;
			sRawData.dst = &(in->at(i));

			out->push(sRawData);
			i += WORDSZ * cnt;
		}
		out->close();
	}).detach();
	return out;
}

auto map(std::shared_ptr<bchan<SplittedRawData>> in)
{
	auto out = std::make_shared<bchan<GridAndEdgeList>>(CHANSZ);
	std::thread([&] {
		for (auto & listb : *in) {
			auto el = std::make_shared<GridAndEdgeList>(listb.cnt / WORDSZ);
			for (size_t i = 0; i < listb.cnt; i += WORDSZ) {
				auto s = listb.src;
				auto d = convBE6toLE8(listb.dst, i);

				if (s < d) {
					std::swap(s, d);
				} else if (s == d) {
					continue;
				}

				GridAndEdge a;
				a.first	 = GridIndex32{uint32_t(s / GWIDTH), uint32_t(d / GWIDTH)};
				a.second = Edge32{uint32_t(s % GWIDTH), uint32_t(d % GWIDTH)};

				el->at(i / WORDSZ) = a;
			}

			out->push(*el);
		}
		out->close();
	}).detach();

	return out;
}

auto shuffle(ShuffleMap & shuffleMap, std::shared_ptr<bchan<GridAndEdgeList>> in) {}

auto cleanup(ShuffleMap & shuffleMap) {}

void phase1(Context & ctx)
{
	auto fn = [](fs::path fpath) {
		auto rawData		 = load(fpath);
		auto splittedRawData = split(rawData);

		auto map = std::make_shared<ShuffleMap>();
	};

	auto jobs = [&ctx] {
		auto out = std::make_shared<bchan<fs::path>>(CHANSZ);
		std::thread([&ctx, out] {
			auto fileList = walk(ctx);
			for (auto & f : *fileList) {
				out->push(f);
			}
			out->close();
		}).detach();
		return out;
	}();

	std::vector<std::thread> threads(2);

	for (auto & t : threads) {
		t = std::thread([&ctx, jobs, fn] {
			for (auto & path : *jobs) {
				fn(path);
			}
		});
	}

	for (auto & t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
}

void init(Context & ctx, int argc, char * argv[])
{
	if (argc != 5) {
		fprintf(stderr, "usage: %s <inFolder> <outFolder> <outName>\n", argv[0]);
	}

	ctx.inFolder  = fs::absolute(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	ctx.outFolder = fs::absolute(fs::path(std::string(argv[2]) + "/").parent_path().string() + "/");
	ctx.outName	  = std::string(argv[3]);
}

int main(int argc, char * argv[])
{
	Context ctx;
	init(ctx, argc, argv);

	phase1(ctx);

	return 0;
}