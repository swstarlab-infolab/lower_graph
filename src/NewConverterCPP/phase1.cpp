#include "main.h"

#include <atomic>
#include <iostream>
#include <thread>

static uint64_t convBE6toLE8(uint8_t * in)
{
	uint64_t temp = 0;
	temp |= in[0];

	for (int i = 1; i <= 5; i++) {
		temp <<= 8;
		temp |= in[i];
	}

	return temp;
}

static auto split(std::shared_ptr<RawData> in)
{
	auto out = std::make_shared<bchan<SplittedRawData>>(__ChannelSize);
	std::thread([=] {
		for (size_t i = 0; i < in->size();) {
			size_t src = convBE6toLE8(&(in->at(i)));
			i += __WordByteLength;
			size_t cnt = convBE6toLE8(&(in->at(i)));
			i += __WordByteLength;

			SplittedRawData sRawData;
			sRawData.src = src;
			sRawData.cnt = cnt;
			sRawData.dst = &(in->at(i));

			out->push(sRawData);
			i += __WordByteLength * cnt;
		}
		out->close();
	}).detach();
	return out;
}

static auto map(std::shared_ptr<bchan<SplittedRawData>> in)
{
	auto out = std::make_shared<bchan<std::shared_ptr<GridAndEdgeList>>>(__ChannelSize);
	std::thread([=] {
		for (auto & listb : *in) {

			auto el = std::make_shared<GridAndEdgeList>(listb.cnt);

			size_t selfloop = 0;

			for (size_t i = 0; i < listb.cnt; i++) {
				auto s = listb.src;
				auto d = convBE6toLE8(&(listb.dst[i * __WordByteLength]));

				if (s < d) {
					std::swap(s, d);
				} else if (s == d) {
					selfloop++;
					continue;
				}

				auto & target = el->at(i - selfloop);
				target.first  = GridIndex32{uint32_t(s / __GridWidth), uint32_t(d / __GridWidth)};
				target.second = Edge32{uint32_t(s % __GridWidth), uint32_t(d % __GridWidth)};
			}

			el->resize(el->size() - selfloop);

			out->push(el);
		}
		out->close();
	}).detach();

	return out;
}

static void shuffle(Context const &											 ctx,
					std::shared_ptr<bchan<std::shared_ptr<GridAndEdgeList>>> in,
					bchan<bool> &											 shuffleDone)
{
	std::thread([&, in] {
		// something
		using TempEntry =
			std::unordered_map<GridIndex32, std::shared_ptr<EdgeList32>, KeyHash, KeyEqual>;
		for (auto & dat : *in) {
			TempEntry temp(__UnorderedMapSize);
			for (auto & gnel : *dat) {
				if (temp.find(gnel.first) == temp.end()) {
					temp.insert_or_assign(gnel.first, std::make_shared<EdgeList32>());
				}
				temp[gnel.first]->push_back(gnel.second);
			}

			for (auto & kv : temp) {
				auto outFolder = ctx.outFolder / ctx.outName;
				if (!fs::exists(outFolder)) {
					fs::create_directories(outFolder);
				}

				auto outFile = outFolder / fs::path(filename(kv.first) + __TempFileExt);

				std::ofstream out(outFile, std::ios::binary | std::ios::app | std::ios::out);

				out.write((char *)(kv.second->data()), kv.second->size() * sizeof(Edge32));
				out.close();
			}
		}

		shuffleDone.push(true);
	}).detach();
}

void phase1(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		// log("Adj6->EdgeList: " + fpath.string() + " Start");

		bchan<bool> shuffleDone(__ChannelSize);

		auto rawData		 = load<uint8_t>(fpath);
		auto splittedRawData = split(rawData);

		std::vector<decltype(map(splittedRawData))> mapper(__MapperCount);
		for (auto & chan : mapper) {
			chan = map(splittedRawData);
			shuffle(ctx, chan, shuffleDone);
		}

		for (size_t i = 0; i < mapper.size(); i++) {
			bool temp;
			shuffleDone.pop(temp);
		}

		log("Phase 1 (Adj6->Edgelist) " + fpath.string() + " Converted");
	};

	auto jobs = [&] {
		auto out = std::make_shared<bchan<fs::path>>(__ChannelSize);
		std::thread([&ctx, out] {
			auto fileList = walk(ctx.inFolder, "");
			for (auto & f : *fileList) {
				out->push(f);
			}
			out->close();
		}).detach();
		return out;
	}();

	std::vector<std::thread> threads(__WorkerCount);

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