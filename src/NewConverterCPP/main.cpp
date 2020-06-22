#include "main.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <tuple>

std::string filename(GridIndex32 in) { return std::to_string(in[0]) + "-" + std::to_string(in[1]); }

uint64_t convBE6toLE8(uint8_t * in)
{
	return (uint64_t(in[0]) << (8 * 5)) + (uint64_t(in[1]) << (8 * 4)) +
		   (uint64_t(in[2]) << (8 * 3)) + (uint64_t(in[3]) << (8 * 2)) +
		   (uint64_t(in[4]) << (8 * 1)) + (uint64_t(in[5]) << (8 * 0));
}

auto walk(fs::path const & inFolder, std::string const & ext)
{
	auto out = std::make_shared<FileList>();
	for (fs::recursive_directory_iterator iter(inFolder), end; iter != end; iter++) {
		if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
			if (ext != "" && iter->path().extension() != ext) {
				continue;
			}
			out->push_back(iter->path());
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
	std::thread([=] {
		for (size_t i = 0; i < in->size();) {
			auto src = convBE6toLE8(&in->at(i));
			i += WORDSZ;
			auto cnt = convBE6toLE8(&in->at(i));
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
	auto out = std::make_shared<bchan<std::shared_ptr<GridAndEdgeList>>>(CHANSZ);
	std::thread([=] {
		for (auto & listb : *in) {
			auto   el		= std::make_shared<GridAndEdgeList>(listb.cnt);
			size_t selfloop = 0;

			for (size_t i = 0; i < listb.cnt; i++) {
				auto s = listb.src;
				auto d = convBE6toLE8(&listb.dst[i * WORDSZ]);

				if (s < d) {
					std::swap(s, d);
				} else if (s == d) {
					selfloop++;
					continue;
				}

				el->at(i - selfloop).first =
					GridIndex32{uint32_t(s / GWIDTH), uint32_t(d / GWIDTH)};
				el->at(i - selfloop).second = Edge32{uint32_t(s % GWIDTH), uint32_t(d % GWIDTH)};

				printf("<(%2d,%2d),(%2d,%2d)>o\n",
					   el->at(i - selfloop).first[0],
					   el->at(i - selfloop).first[1],
					   el->at(i - selfloop).second[0],
					   el->at(i - selfloop).second[1]);
			}

			// 도대체 뭐가 문제야 ?!?!?!??!

			el->resize(el->size() - selfloop);

			for (auto & e : *el) {
				printf("<(%2d,%2d),(%2d,%2d)>\n", e.first[0], e.first[1], e.second[1], e.second[2]);
			}

			out->push(el);
		}
		out->close();
	}).detach();

	return out;
}

void writer(Context const & ctx,
			GridIndex32		gidx32,
			WriterEntry &	writerEntry,
			bchan<bool> &	writeDone)
{
	auto & myChan = writerEntry[gidx32];

	auto outFolder = ctx.outFolder / ctx.outName;
	if (!fs::exists(outFolder)) {
		fs::create_directories(outFolder);
	}

	auto outFile = outFolder / fs::path(filename(gidx32) + TEMPFILEEXT);

	std::ofstream out(outFile, std::ios::binary | std::ios::app | std::ios::out);

	for (auto & el : *myChan) {
		out.write((char *)(el->data()), el->size() * sizeof(Edge32));
	}

	out.close();

	writeDone.push(true);
}

void shuffle(Context const &										  ctx,
			 std::shared_ptr<bchan<std::shared_ptr<GridAndEdgeList>>> in,
			 bchan<bool> &											  shuffleDone,
			 bchan<bool> &											  writeDone,
			 WriterEntry &											  writerEntry,
			 std::mutex &											  writerEntryMutex,
			 std::atomic<uint32_t> &								  writerCnt)
{
	std::thread([&, in] {
		// something
		using TempEntry =
			std::unordered_map<GridIndex32, std::shared_ptr<EdgeList32>, KeyHash, KeyEqual>;
		for (auto & dat : *in) {
			TempEntry temp(UNORDEREDMAPSZ);
			for (auto & gnel : *dat) {
				if (temp.find(gnel.first) == temp.end()) {
					temp.insert_or_assign(gnel.first, std::make_shared<EdgeList32>());
				}
				temp[gnel.first]->push_back(gnel.second);
			}

			for (auto & kv : temp) {
				std::unique_lock<std::mutex> ul(writerEntryMutex);

				if (writerEntry.find(kv.first) == writerEntry.end()) {
					writerEntry.insert_or_assign(
						kv.first, std::make_shared<bchan<std::shared_ptr<EdgeList32>>>(CHANSZ));
					writerCnt.fetch_add(1);
					std::thread(
						writer, std::ref(ctx), kv.first, std::ref(writerEntry), std::ref(writeDone))
						.detach();
				}
				ul.unlock();

				writerEntry[kv.first]->push(kv.second);
			}
		}

		shuffleDone.push(true);
	}).detach();
}

void phase1(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		bchan<bool>			  shuffleDone(CHANSZ);
		bchan<bool>			  writeDone(CHANSZ);
		WriterEntry			  writerEntry(UNORDEREDMAPSZ);
		std::mutex			  writerEntryMutex;
		std::atomic<uint32_t> writerCnt;

		auto rawData		 = load(fpath);
		auto splittedRawData = split(rawData);

		std::vector<decltype(map(splittedRawData))> mapper(MAPPERSZ);
		for (auto & chan : mapper) {
			chan = map(splittedRawData);
			shuffle(ctx, chan, shuffleDone, writeDone, writerEntry, writerEntryMutex, writerCnt);
		}

		for (size_t i = 0; i < mapper.size(); i++) {
			bool temp;
			shuffleDone.pop(temp);
		}

		for (auto & kv : writerEntry) {
			kv.second->close();
		}

		for (size_t i = 0; i < writerCnt.load(); i++) {
			bool temp;
			writeDone.pop(temp);
		}
	};

	auto jobs = [&ctx] {
		auto out = std::make_shared<bchan<fs::path>>(CHANSZ);
		std::thread([&ctx, out] {
			auto fileList = walk(ctx.inFolder, "");
			for (auto & f : *fileList) {
				out->push(f);
			}
			out->close();
		}).detach();
		return out;
	}();

	std::vector<std::thread> threads(WORKERSZ);

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

void dedup()
{
	// sort
	// prepare bit array
	// prepare exclusive sum array
	// set bit 1 which is left != right (not the case: left == right)
	// exclusive sum
	// count bit using parallel reduce
	// reduce out vector
}

void writeCSR()
{
	// CSR file write
}

void phase2()
{
	// phase 2
}

void init(Context & ctx, int argc, char * argv[])
{
	if (argc != 4) {
		fprintf(stderr, "usage: %s <inFolder> <outFolder> <outName>\n", argv[0]);
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

	phase1(ctx);

	return 0;
}