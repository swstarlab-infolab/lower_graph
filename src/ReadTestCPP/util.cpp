#include "main.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

std::shared_ptr<FileList> walk(fs::path const & inFolder, std::string const & ext)
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

static std::string currentTimeAndDate()
{
	auto			  now		= std::chrono::system_clock::now();
	auto			  in_time_t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
	return ss.str();
}

void log(std::string const & s) { printf("%s %s\n", currentTimeAndDate().c_str(), s.c_str()); }