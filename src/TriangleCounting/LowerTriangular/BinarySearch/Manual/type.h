#ifndef C26BEB06_5F3E_48D4_AB98_5DE67AD09131
#define C26BEB06_5F3E_48D4_AB98_5DE67AD09131
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
#include <memory>
#include <unordered_map>

using Lookup = uint32_t;
using Count = unsigned long long;

using GridIndex = std::array<uint32_t, 2>;
using ThreeGrids = std::array<GridIndex, 3>;

#define bchan boost::fibers::buffered_channel
#define uchan boost::fibers::unbuffered_channel
#define fiber boost::fibers::fiber

struct MemoryInfo {
    void* ptr;
    size_t byte;
};

struct FileInfo {
    fs::path path;
    size_t byte;
};

enum Method : uint32_t {
    QUERY,
    READY,
    DONE,
    DESTROY
};

template <typename CallbackType>
struct DataRequest {
    Method method;
    GridIndex gidx;
    bchan<CallbackType> callback;
};

struct CommandResult {
    ThreeGrids gidxs;
    Count triangles;
    double elapsedTime;
    int deviceID;
};

struct Command {
    ThreeGrids gidx;
};

/*
template <typename T>
struct Request {
    T data;
};

template <typename T>
struct Response {
    T data;
    bool ok;
};
*/

struct Context {
    // folder path
    fs::path folderPath;

    // device count
    int deviceCount = -1;

    // setting (cudaStreams, cudaBlocks, cudaThreads)
    std::array<size_t, 3> setting;

    // Grid metadata
    GridCSR::MetaData meta;

    struct Connections {
        int32_t upstream;
        std::vector<int32_t> neighbor;
    };

    // Data Manager's request channel
    std::unordered_map<int32_t, std::shared_ptr<bchan<DataRequest<MemoryInfo>>>> memChan;
    std::unordered_map<int32_t, std::shared_ptr<bchan<DataRequest<FileInfo>>>> fileChan;

    // Data Manager's connection request graph
    std::unordered_map<int32_t, Connections> conn;
};
#endif /* C26BEB06_5F3E_48D4_AB98_5DE67AD09131 */
