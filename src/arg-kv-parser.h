#ifndef __ARG_KV_PARSER_H__
#define __ARG_KV_PARSER_H__

#include <functional>
#include <string>
#include <vector>
#include <map>

using arg_kv_t = std::map<std::string, std::function<void(std::string &)>>;

void arg_parse(int argc, char * argv[], arg_kv_t & arg_kv)
{
    auto usage = [](std::vector<std::string> const & argv, arg_kv_t & arg_kv){
        fprintf(stderr, "Usage: %s --[KEY]=[VALUE]\n\t[KEYs]\n", argv[0].c_str());

        for (auto iter = arg_kv.begin(); iter != arg_kv.end(); iter++) {
            fprintf(stderr, "\t\t%s\n", iter->first.c_str());
        }
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    };

    std::vector<std::string> argv_s(argc);
    for (int i = 0; i < argc; i++) {
        argv_s[i] = std::string(argv[i]);
    }

	if (argv_s.size() < 2) {
		usage(argv_s, arg_kv);
	}

	// parse and store all arg to vector kv
	std::map<std::string, std::string> _kv;
	for (unsigned int i = 1; i < argv_s.size(); i++) {
		if (argv_s[i].size() < 5) {
			fprintf(stderr, "Argument %s doesn't satisfy '--[KEY]=[VALUE]' form\n\n", argv_s[i].c_str());
			usage(argv_s, arg_kv);
		}

		if (argv_s[i][0] != '-' || argv_s[i][1] != '-') {
			fprintf(stderr, "Argument %s doesn't satisfy '--[KEY]=[VALUE]' form\n\n", argv_s[i].c_str());
			usage(argv_s, arg_kv);
		}

		std::string s = argv_s[i];
		auto eq_pos = s.find('=');
		auto key = s.substr(2, eq_pos - 2);
		auto value = s.substr(eq_pos + 1, s.size() - eq_pos - 1);

		if (key.size() == 0 || value.size() == 0) {
			fprintf(stderr, "Argument %s doesn't satisfy '--[KEY]=[VALUE]' form\n\n", argv_s[i].c_str());
			usage(argv_s, arg_kv);
		}

		_kv[key] = value;
	}

	for (auto _kv_iter = _kv.begin(); _kv_iter != _kv.end(); ++_kv_iter) {
		auto arg_kv_iter = arg_kv.find(_kv_iter->first);
		if (arg_kv_iter != arg_kv.end()) {
			arg_kv_iter->second(_kv_iter->second);
		} else {
			fprintf(stderr, "Key %s is unrecognized\n\n", _kv_iter->first.c_str());
			usage(argv_s, arg_kv);
		}
	}
}

#endif