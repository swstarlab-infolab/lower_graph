#include "logging.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

std::mutex __logging_mutex;

std::string currTimeDate()
{
	auto now	   = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y/%m/%d %X");
	return ss.str();
}