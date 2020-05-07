#include "Converter.h"

#include <ArgParser/ArgParser.h>

#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    typename ArgParser::KV arg_kv;

    std::string in_type, out_type, out_sort;
    arg_kv["in.type"] = [&](std::string & val){ in_type = val; };
    arg_kv["out.type"] = [&](std::string & val){ out_type = val; };
    arg_kv["out.sort"] = [&](std::string & val){ out_sort = val; };

    GridCSR::FS::path inFolder, outFolder;
    arg_kv["in.folder"] = [&](std::string & val){ inFolder = GridCSR::FS::path(GridCSR::FS::path(val + "/").parent_path().string() + "/"); };
    arg_kv["out.folder"] = [&](std::string & val){ outFolder = GridCSR::FS::path(GridCSR::FS::path(val + "/").parent_path().string() + "/"); };

    size_t grid_width_power;
    arg_kv["out.width"] = [&](std::string & val){ grid_width_power = strtol(val.c_str(), nullptr, 10); };

    std::string outDataname;
    arg_kv["out.name"] = [&](std::string & val){ outDataname = val; };

    ArgParser::Parse(argc, argv, arg_kv);

    GridCSRConverter converter(1 << grid_width_power);

    {
        using type = GridCSRConverter::GridCSRConverterMode::SortingType;
        if (out_type == "directed") {
            if (out_sort == "no") {
                converter.setMode(true, type::not_sort);
                printf("Output: Directed graph; No edge manipulations\n");
            } else if (out_sort == "lt") {
                converter.setMode(true, type::lower_triangle);
                printf("Output: Directed graph + GridCSR::Edge manipulation for leaving edge (v,w) where v > w\n");
            } else if (out_sort == "deg") {
                converter.setMode(true, type::degree);
                printf("Output: Directed graph + GridCSR::Edge manipulation for leaving edge (v,w) where deg(v) < deg(w)\n");
            } else {
                return 0;
            }
        } else if (out_type == "undirected") {
            converter.setMode(false, type::not_sort);
            printf("Output: Undirected graph; edge manipulation for leaving edge (v,w) and (w,v)\n");
        } else {
            return 0;
        }

    }

    {
        auto start = std::chrono::system_clock::now();

        if (in_type == "tsv") {
            converter.loadTSV(inFolder);
        } else if (in_type == "adj6") {
            converter.loadAdj6(inFolder);
        } else {
            return 0;
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> sec = end - start;
        std::cout << "Complete: Load, time: " << sec.count() << "(sec)" << std::endl;
    }

    {
        auto start = std::chrono::system_clock::now();

        converter.run();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> sec = end - start;
        std::cout << "Complete: Convert, time: " << sec.count() << "(sec)" << std::endl;
    }

    {
        auto start = std::chrono::system_clock::now();

        converter.storeGCSR(outFolder, outDataname);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> sec = end - start;
        std::cout << "Complete: Store, time: " << sec.count() << "(sec)" << std::endl;
    }

    return 0;
}