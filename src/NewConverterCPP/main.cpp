#include "main.h"

#include <GridCSR/GridCSR.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tuple>

std::mutex logmtx;

std::string filename(GridIndex32 in) { return std::to_string(in[0]) + "-" + std::to_string(in[1]); }

uint64_t convBE6toLE8(uint8_t * in)
{
	uint64_t temp = 0;
	temp |= in[0];

	for (int i = 1; i <= 5; i++) {
		temp <<= 8;
		temp |= in[i];
	}

	return temp;
	/*
	return (uint64_t(in[0]) << (8 * 5)) + (uint64_t(in[1]) << (8 * 4)) +
		   (uint64_t(in[2]) << (8 * 3)) + (uint64_t(in[3]) << (8 * 2)) +
		   (uint64_t(in[4]) << (8 * 1)) + (uint64_t(in[5]) << (8 * 0));
		   */
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

template <typename T>
auto load(fs::path inFile)
{
	std::ifstream f(inFile, std::ios::binary);

	auto fbyte = fs::file_size(inFile);
	auto out   = std::make_shared<std::vector<T>>(fbyte / sizeof(T));

	f.read((char *)(out->data()), fbyte);
	f.close();

	return out;
}

auto split(std::shared_ptr<RawData> in)
{
	auto out = std::make_shared<bchan<SplittedRawData>>(CHANSZ);
	std::thread([=] {
		for (size_t i = 0; i < in->size();) {
			size_t src = convBE6toLE8(&(in->at(i)));
			i += WORDSZ;
			size_t cnt = convBE6toLE8(&(in->at(i)));
			i += WORDSZ;

			SplittedRawData sRawData;
			sRawData.src = src;
			sRawData.cnt = cnt;
			sRawData.dst = &(in->at(i));

			/*
						{
							std::lock_guard<std::mutex> lg(logmtx);
							printf("SPLIT %ld[%ld] -> ", src, cnt);
							for (size_t j = 0; j < cnt; j++) {
								printf("%ld ", convBE6toLE8(&(in->at(i + j * WORDSZ))));
							}
							printf("\n");
						}
						*/

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

			auto el = std::make_shared<GridAndEdgeList>(listb.cnt);

			size_t selfloop = 0;

			for (size_t i = 0; i < listb.cnt; i++) {
				auto s = listb.src;
				auto d = convBE6toLE8(&(listb.dst[i * WORDSZ]));

				if (s < d) {
					std::swap(s, d);
				} else if (s == d) {
					selfloop++;
					continue;
				}

				auto & target = el->at(i - selfloop);
				target.first  = GridIndex32{uint32_t(s / GWIDTH), uint32_t(d / GWIDTH)};
				target.second = Edge32{uint32_t(s % GWIDTH), uint32_t(d % GWIDTH)};

				/*
								{
									std::lock_guard<std::mutex> lg(logmtx);
									printf("map intermediate <(%d,%d),(%d,%d)>\n",
										   target2.first[0],
										   target2.first[1],
										   target2.second[0],
										   target2.second[1]);
								}
								*/
			}

			el->resize(el->size() - selfloop);

			/*
						for (size_t i = 0; i < el->size(); i++) {
							{
								std::lock_guard<std::mutex> lg(logmtx);
								printf("map result <(%d,%d),(%d,%d)>\n",
									   el2.at(i).first[0],
									   el2.at(i).first[1],
									   el2.at(i).second[1],
									   el2.at(i).second[2]);
							}
						}
						*/

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
	auto & myChan = writerEntry.at(gidx32);

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

	/*
	for (auto & el : *myChan) {
		for (auto & e : *el) {
			{
				std::lock_guard<std::mutex> lg(logmtx);
				printf("WRITER (%d,%d)\n", e[0], e[1]);
			}
		}
	}
	*/

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

			/*
			{
				std::lock_guard<std::mutex> lg(logmtx);
				printf("shuffle sorted\n");
				for (auto & kv : temp) {
					printf("<(%d,%d),{", kv.first[0], kv.first[1]);
					for (auto & e : *kv.second) {
						printf("(%d,%d) ", e[1], e[2]);
					}
					printf("}>\n");
				}
			}
			*/

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

				writerEntry.at(kv.first)->push(kv.second);
			}
		}

		shuffleDone.push(true);
	}).detach();
}

void phase1(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		{
			// std::lock_guard<std::mutex> lg(logmtx);
			printf("FILE: %s\n", fpath.string().c_str());
		}
		bchan<bool> shuffleDone(CHANSZ);
		bchan<bool> writeDone(CHANSZ);
		// auto shuffleDone = std::make_shared<bchan<bool>>(CHANSZ);
		// auto writeDone	 = std::make_shared<bchan<bool>>(CHANSZ);
		// auto writerEntry = std::make_shared<WriterEntry>(UNORDEREDMAPSZ);
		WriterEntry writerEntry(UNORDEREDMAPSZ);

		std::mutex			  writerEntryMutex;
		std::atomic<uint32_t> writerCnt = 0;

		auto rawData		 = load<uint8_t>(fpath);
		auto splittedRawData = split(rawData);

		std::vector<decltype(map(splittedRawData))> mapper(MAPPERSZ);
		for (auto & chan : mapper) {
			chan = map(splittedRawData);
			shuffle(ctx, chan, shuffleDone, writeDone, writerEntry, writerEntryMutex, writerCnt);
		}

		for (size_t i = 0; i < mapper.size(); i++) {
			bool temp;
			shuffleDone.pop(temp);
			/*
			{
				std::lock_guard<std::mutex> lg(logmtx);
				printf("fn: shuffle done: %d\n", i);
			}
			*/
		}

		for (auto & kv : writerEntry) {
			kv.second->close();
		}

		for (size_t i = 0; i < writerCnt.load(); i++) {
			bool temp;
			writeDone.pop(temp);
			/*
			{
				std::lock_guard<std::mutex> lg(logmtx);
				printf("fn: write done: %d\n", i);
			}
			*/
		}
	};

	auto jobs = [&] {
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

auto dedup(std::shared_ptr<EdgeList32> in)
{
	// init
	auto out = std::make_shared<EdgeList32>(in->size());

	std::vector<std::atomic<uint32_t>> bitvec(size_t(ceil(double(in->size()) / 32.0)));

	auto getBit = [&bitvec](size_t const i) {
		return bool(bitvec[i / 32].load() & (1 << (i % 32)));
	};
	auto setBit = [&bitvec](size_t const i) { bitvec[i / 32].fetch_or(1 << (i % 32)); };

	std::vector<uint64_t> pSumRes(in->size() + 1);

	// prepare bit array
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, bitvec.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					bitvec[i] = 0;
				}
			}
		},
		tbb::auto_partitioner());

	// prepare exclusive sum array
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, pSumRes.size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					pSumRes[i] = 0;
				}
			}
		},
		tbb::auto_partitioner());

	// sort
	tbb::parallel_sort(in->begin(), in->end(), [&](Edge32 const & l, Edge32 const & r) {
		return (l[0] < r[0]) || ((l[0] == r[0]) && (l[1] < r[1]));
	});

	// set bit 1 which is left != right (not the case: left == right)
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in->size() - 1),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto curr = grain + offset;
					auto next = grain + offset + 1;
					if (in->at(curr) != in->at(next)) {
						setBit(curr);
					}
				}
			}
		},
		tbb::auto_partitioner());
	setBit(in->size() - 1);

	// exclusive sum
	tbb::parallel_scan(
		tbb::blocked_range<size_t>(0, in->size()),
		0,
		[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += (getBit(i) ? 1 : 0);
					if (isFinalScan) {
						pSumRes[i + 1] = temp;
					}
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	// count bit using parallel reduce
	size_t ones = tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, 32 * bitvec.size()),
		0,
		[&](tbb::blocked_range<size_t> const & r, size_t sum) {
			auto temp = sum;
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;
					temp += (getBit(i) ? 1 : 0);
				}
			}
			return temp;
		},
		[&](size_t const & l, size_t const & r) { return l + r; },
		tbb::auto_partitioner());

	out->resize(ones);

	// reduce out vector
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, in->size()),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
				for (size_t offset = 0; offset < r.grainsize(); offset++) {
					auto i = grain + offset;

					if (getBit(i)) {
						out->at(pSumRes[i]) = in->at(i);
					}
				}
			}
		},
		tbb::auto_partitioner());

	return out;
}

void writeCSR(Context const & ctx, fs::path tempFilePath, std::shared_ptr<EdgeList32> in)
{
	std::vector<std::vector<Vertex32>> out(3);

	auto fillCol = std::thread([&, in] {
		// fill col
		out[2].resize(in->size());

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						out[2][i] = in->at(i)[1];
					}
				}
			},
			tbb::auto_partitioner());
	});

	auto fillRowAndPtr = std::thread([&, in] {
		// fill
		std::vector<std::atomic<uint32_t>> bitvec(size_t(ceil(double(in->size()) / 32.0)));

		auto getBit = [&bitvec](size_t const i) {
			return bool(bitvec[i / 32].load() & (1 << (i % 32)));
		};
		auto setBit = [&bitvec](size_t const i) { bitvec[i / 32].fetch_or(1 << (i % 32)); };

		std::vector<uint64_t> pSumRes(in->size() + 1);

		// prepare bit array
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, bitvec.size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						bitvec[i] = 0;
					}
				}
			},
			tbb::auto_partitioner());

		// prepare exclusive sum array
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, pSumRes.size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						pSumRes[i] = 0;
					}
				}
			},
			tbb::auto_partitioner());

		// set bit 1 which is left != right (not the case: left == right)
		tbb::parallel_for(
			tbb::blocked_range<size_t>(1, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto prev = grain + offset - 1;
						auto curr = grain + offset;
						if (in->at(prev)[0] != in->at(curr)[0]) {
							setBit(curr);
						}
					}
				}
			},
			tbb::auto_partitioner());
		setBit(0);

		// exclusive sum
		tbb::parallel_scan(
			tbb::blocked_range<size_t>(0, in->size()),
			0,
			[&](tbb::blocked_range<size_t> const & r, uint64_t sum, bool isFinalScan) {
				auto temp = sum;
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;
						temp += (getBit(i) ? 1 : 0);
						if (isFinalScan) {
							pSumRes[i + 1] = temp;
						}
					}
				}
				return temp;
			},
			[&](size_t const & l, size_t const & r) { return l + r; },
			tbb::auto_partitioner());

		// count bit using parallel reduce
		size_t ones = tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, 32 * bitvec.size()),
			0,
			[&](tbb::blocked_range<size_t> const & r, size_t sum) {
				auto temp = sum;
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;
						temp += (getBit(i) ? 1 : 0);
					}
				}
				return temp;
			},
			[&](size_t const & l, size_t const & r) { return l + r; },
			tbb::auto_partitioner());

		out[0].resize(ones);
		out[1].resize(ones + 1);

		// fill row
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						if (getBit(i)) {
							out[0][pSumRes[i]] = in->at(i)[0];
						}
					}
				}
			},
			tbb::auto_partitioner());

		// fill ptr
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, in->size()),
			[&](tbb::blocked_range<size_t> const & r) {
				for (size_t grain = r.begin(); grain != r.end(); grain += r.grainsize()) {
					for (size_t offset = 0; offset < r.grainsize(); offset++) {
						auto i = grain + offset;

						if (getBit(i)) {
							out[1][pSumRes[i]] = i;
						}
					}
				}
			},
			tbb::auto_partitioner());
		out[1].back() = in->size();
	});

	if (fillCol.joinable()) {
		fillCol.join();
	}

	if (fillRowAndPtr.joinable()) {
		fillRowAndPtr.join();
	}
	/*
	printf("ROW ");
	for (auto & i : out[0]) {
		printf("%d ", i);
	}
	printf("\n");
	printf("PTR ");
	for (auto & i : out[1]) {
		printf("%d ", i);
	}
	printf("\n");
	printf("COL ");
	for (auto & i : out[2]) {
		printf("%d ", i);
	}
	printf("\n");
	*/

	std::vector<std::thread> writeThreads(3);
	for (size_t i = 0; i < writeThreads.size(); i++) {
		writeThreads[i] = std::thread([&, i] {
			auto outFile = (ctx.outFolder / ctx.outName) /
						   fs::path(tempFilePath.stem().string() + OUTFILEEXT[i]);

			std::ofstream f(outFile, std::ios::binary | std::ios::out);

			f.write((char *)(out[i].data()), out[i].size() * sizeof(Vertex32));
			f.close();
		});
	}

	for (auto & t : writeThreads) {
		if (t.joinable()) {
			t.join();
		}
	}

	fs::remove(tempFilePath);
}

void phase2(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		{
			// std::lock_guard<std::mutex> lg(logmtx);
			printf("FILE: %s\n", fpath.string().c_str());
		}
		auto rawData = load<Edge32>(fpath);
		/*
		{
			std::lock_guard<std::mutex> lg(logmtx);
			printf("FILE: %s loadCompleted\n", fpath.string().c_str());
		}
		*/
		auto deduped = dedup(rawData);
		/*
		for (auto & e : *deduped) {
			printf("(%d,%d) ", e[0], e[1]);
		}
		printf("\n");
		*/
		writeCSR(ctx, fpath, deduped);
	};

	auto jobs = [&] {
		auto out = std::make_shared<bchan<fs::path>>(CHANSZ);
		std::thread([&ctx, out] {
			auto fileList = walk(ctx.outFolder, TEMPFILEEXT);
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

// metadata writing
void phase3(Context const & ctx) {}

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

	printf("ElapsedTime: %lf (sec)\n", elapsed.count());

	return 0;
}