#ifndef B0343A5C_B8D6_4967_809A_3487B01AAA67
#define B0343A5C_B8D6_4967_809A_3487B01AAA67

#include "make.h"
#include "type.h"
#include <BuddySystem/BuddySystem.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

auto MyMemoryMap()
{
}

auto DataManagerStorage(Context& ctx, int myID)
{
    std::unordered_map<GridIndex, FileInfo> memmap;
    std::mutex mtx;

    MyMemoryMap();
    for (auto& req : *ctx.fileChan[myID].get()) {
        switch (req.method) {
        case Method::QUERY:
            std::unique_lock<std::mutex> ul(mtx);
            if (memmap.find(req.gidx) == memmap.end()) {
                return FileInfo {

                };
            }
            break;
        case Method::READY:
            break;
        case Method::DONE:
            break;
        case Method::DESTROY:
            break;
        }
    }
    /*
        req.gidx;
        req.callback;
        //ctx.fileChan[myID].close
        // generate file path
        auto const baseString
            = std::string(ctx.folderPath) + std::to_string(k[0]) + "-" + std::to_string(k[1]) + ".";

        std::array<std::string, 3> path
        {
            fs::path(baseString + ctx.meta.extension.row),
                fs::path(baseString + ctx.meta.extension.ptr),
                fs::path(baseString + ctx.meta.extension.col),
        }

        // check existence
        for (auto& p : path) {
            if (!fs::exists(p)) {
                return false;
            }
        }
    }
    */
}

auto DataManagerMainMemory(Context const& ctx, int myID)
{
}

auto DataManagerDeviceMemory(Context const& ctx, int myID)
{
    auto& myChan = ctx.
}

void DataManager(Context& ctx, int myID)
{
    if (myID < 0) {
        fiber([&, myID] { DataManagerStorage(ctx, myID); }).detach();
    } else if (myID == 0) {
        fiber([&, myID] { DataManagerMainMemory(ctx, myID); }).detach();
    } else {
        fiber([&, myID] { DataManagerDeviceMemory(ctx, myID); }).detach();
    }
}

/*
static size_t getFileSize(fs::path const& path)
{
    std::ifstream f;
    f.open(path);
    f.seekg(0, std::ios::end);
    auto const fileSize = f.tellg();
    f.seekg(0, std::ios::beg);
    f.close();
    return fileSize;
}

static void loadFile(fs::path const& path, void* ptr, size_t byte)
{
    std::ifstream f;
    f.open(path, std::ios::binary);
    f.read((char*)ptr, byte);
    f.close();
}

static auto tryLoad(
    Context const& ctx,
    portable_buddy_system& buddy,
    DataType::GridIndex const& k,
    DataType::CacheValue& v)
{

    // get file size
    std::array<DataType::Memory, 3> mem;
    for (unsigned int i = 0; i < mem.size(); i++) {
        mem[i].second = getFileSize(path[i]);
        mem[i].first = buddy.allocate(mem[i].second);
        if (mem[i].first == nullptr) {
            return false;
        }
        loadFile(path[i], mem[i].first, mem[i].second);
    }

    // success
    v.arr = mem;

    return true;
}

void loader(
    Context const& ctx,
    ChanLoadReq& loadReq,
    std::vector<std::shared_ptr<ChanLoadRes>>& loadRes,
    ChanLoadComp& loadComp)
{
    fprintf(stdout, "[LOADER] START\n");

    // memory allocation
    auto map = makeHashMap<DataType::GridIndex, DataType::CacheValue>(
        1024,
        [](DataType::GridIndex const& key) {
            using KeyType = std::remove_const<std::remove_reference<decltype(key.front())>::type>::type;
            return std::hash<KeyType>()(key[0]) ^ (std::hash<KeyType>()(key[1]) << 1);
        });

    DataType::MemoryShared myMemory;
    myMemory.second = 1024L * 1024L * 1024L;
    myMemory.first = allocCUDA<char>(myMemory.second);

    portable_buddy_system buddy;
    buddy.init(memrgn_t { (void*)myMemory.first.get(), myMemory.second }, 8, 1);

    fprintf(stdout, "[LOADER] Init Complete\n");

    auto garbageCollector = boost::fibers::fiber([&] {
        for (auto& comp : loadComp) {
            decltype(map)::accessor a;
            map.find(a, comp.idx);
            a->second.counter--;
        }
    });

    auto cacheman = boost::fibers::fiber([&] {
        for (auto& req : loadReq) {
            fprintf(stdout, "[LOADER] Got Request!: (%d,%d)\n", req.idx[0], req.idx[1]);

            DataType::CacheValue v;
            decltype(map)::accessor a;

            if (map.find(a, req.idx)) {
                fprintf(stdout, "[LOADER] Present: (%d,%d), %p,%ld\n", req.idx[0], req.idx[1], v.arr[0].first, v.arr[0].second);

                v.arr = a->second.arr;
                a->second.counter++;
                a.release();
            } else {
                // not present
                if (tryLoad(ctx, buddy, req.idx, v)) {
                    fprintf(stdout, "[LOADER] Try Load Success: (%d,%d), %p,%ld\n", req.idx[0], req.idx[1], v.arr[0].first, v.arr[0].second);
                    v.counter = 1;
                    map.insert(a, req.idx);
                    a->second.arr = v.arr;
                    a->second.counter = v.counter;
                    a.release();
                } else {
                    // find evictable
                    a.release();
                }
            }

            MessageType::LoadRes res;
            // enqueue;
            res.idx = req.idx;
            res.arr = v.arr;

            loadRes[req.deviceID].get()->push(res);
        }

        loadComp.close();
    });

    garbageCollector.join();
    cacheman.join();

    for (auto& c : loadRes) {
        c->close();
    }
}
*/
#endif /* B0343A5C_B8D6_4967_809A_3487B01AAA67 */
