#include "type.h"
#include "util.h"

#include <thread>
#include <unordered_map>
#include <vector>

template <class Key, class Val, class Hash = std::hash<Key>, class Equal = std::equal_to<Key>>
static auto make_unordered_map(
	typename std::unordered_map<Key, Val, Hash, Equal>::size_type bucket_count = (1 << 16),
	Hash const &												  hash		   = Hash(),
	Equal const &												  equal		   = Equal())
{
	return std::unordered_map<Key, Val, Hash, Equal>(bucket_count, hash, equal);
}

static auto mapper(sp<std::vector<uint8_t>> adj6,
				   sp<bchan<RowPos>>		in,
				   uint32_t const			gridWidth,
				   bool const				lowerTriangular)
{
	auto out = makeSp<bchan<sp<std::vector<GE32>>>>(16);
	std::thread([=] {
		for (auto dat : *in) {
			auto el		  = makeSp<std::vector<GE32>>(dat.cnt);
			auto selfloop = uint64_t(0);

			for (uint64_t i = 0; i < dat.cnt; i++) {
				auto src = dat.src;
				auto dst = be6_le8(&(adj6->at(dat.dstStart + i * 6)));

				if (lowerTriangular && src < dst) {
					std::swap(src, dst);
				} else if (src == dst) {
					selfloop++;
					continue;
				}

				GE32 ge32;

				ge32[0][0] = (uint32_t)(src / gridWidth);
				ge32[0][1] = (uint32_t)(dst / gridWidth);
				ge32[1][0] = (uint32_t)(src % gridWidth);
				ge32[1][1] = (uint32_t)(dst % gridWidth);

				el->at(i - selfloop) = ge32;
			}

			el->resize(el->size() - selfloop);
			out->push(el);
		}
		out->close();
	}).detach();
	return out;
}

static void
shuffler(sp<bchan<sp<std::vector<GE32>>>> in, fs::path const & folder, std::string const & ext)
{

	auto map = make_unordered_map<E32, sp<std::vector<E32>>>(
		2048,
		[](E32 const & k) {
			auto a = std::hash<uint64_t>{}(uint64_t(k[0]) << (8 * sizeof(k[0])));
			auto b = std::hash<uint64_t>{}(k[1]);
			return a ^ b;
		},
		[](E32 const & kl, E32 const & kr) { return (kl[0] == kr[0] && kl[1] == kr[1]); });

	for (auto dat : *in) {
		for (auto ge : *dat) {
			if (map.find(ge[0]) == map.end()) {
				map[ge[0]] = makeSp<std::vector<E32>>();
			}
			map[ge[0]]->push_back(ge[1]);

			// flush
			if (map[ge[0]]->size() > (1 << 20)) {
				auto targetFile = folder / fs::path(fileNameEncode(ge[0], ext));
				fileSaveAppend(targetFile, map[ge[0]]->data(), map[ge[0]]->size() * sizeof(ge[0]));
				map[ge[0]]->resize(0);
			}
		}
	}

	// flush
	for (auto & kv : map) {
		if (kv.second->size() > 0) {
			auto targetFile = folder / fs::path(fileNameEncode(kv.first, ext));
			fileSaveAppend(targetFile, kv.second->data(), kv.second->size() * sizeof(kv.first));
		}
	}
}

void stage1(fs::path const & inFolder,
			fs::path const & outFolder,
			uint32_t const	 gridWidth,
			bool const		 lowerTriangular)
{

	auto fListChan = fileList(inFolder, "");

	parallelDo(8, [&](size_t const i) {
		for (auto & fPath : *fListChan) {
			stopwatch("Stage1, " + std::string(fPath), [&] {
				auto adj6		= fileLoad<uint8_t>(fPath);
				auto rowPosChan = splitAdj6(adj6);

				parallelDo(64, [&](size_t const i) {
					auto mapped = mapper(adj6, rowPosChan, gridWidth, lowerTriangular);
					shuffler(mapped, outFolder, ".el32");
				});
			});
		}
	});
}
