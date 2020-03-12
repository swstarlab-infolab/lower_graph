#ifndef __META_H__
#define __META_H__

#include "common.h"

#include <string>
#include <vector>

struct meta_t {
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
       struct each_t {
           struct {
               size_t row, col;
           } index;
           std::string name;
       };
       std::vector<each_t> each;
    } grid;

    void marshal_to_file(fs::path const & filePath);
    void unmarshal_from_file(fs::path const & filePath);
};

#endif