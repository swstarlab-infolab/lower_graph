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

			if (src == 0 || cnt == 0) {
				log("src: " + std::to_string(src) + " cnt: " + std::to_string(cnt));
			}

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

void phase1(Context const & ctx)
{
	auto fn = [&](fs::path fpath) {
		log("Adj6->EdgeList: " + fpath.string() + " Start");

		auto rawData		 = load<uint8_t>(fpath);
		auto splittedRawData = split(rawData);

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