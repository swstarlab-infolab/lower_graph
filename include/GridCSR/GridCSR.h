#ifndef __GridCSR_GridCSR_h__
#define __GridCSR_GridCSR_h__

#if __GNUC__ < 8
#include <experimental/filesystem>
#else
#include <filesystem>
#endif


#include <stdint.h>
#include <string>
#include <vector>

namespace GridCSR {

#if __GNUC__ < 8
namespace FS = std::experimental::filesystem;
#else
namespace FS = std::filesystem;
#endif

using Vertex = uint32_t;

struct Edge {
    Vertex src, dst;
};

struct MetaData {
    std::string dataname;

    struct {
        std::string row, ptr, col;
    } extension;

    struct {
        struct {
            size_t row, col;
        } count;

        struct {
            size_t row, col;
        } width;

        size_t max_vid;
    } info;

    struct {
        struct GridInfo {
            struct {
                size_t row, col;
            } index;
            std::string name;
        };
        std::vector<GridInfo> each;
    } grid;

    void Load(FS::path const & filePath);
    void Save(FS::path const & filePath);
};

} // namespace GridCSR

#endif