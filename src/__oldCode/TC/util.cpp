#include "util.h"

#include <boost/fiber/fiber.hpp>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::mutex __logMtx;

void log(std::string const & s)
{
	std::lock_guard<std::mutex> lg(__logMtx);

	auto currentTimeAndDate = [] {
		auto now	   = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);

		std::stringstream ss;
		ss << std::put_time(std::localtime(&in_time_t), "%Y/%m/%d %X");
		return ss.str();
	};

	fprintf(stdout, "%s %s\n", currentTimeAndDate().c_str(), s.c_str());
}

void stopwatch(std::string const & message, std::function<void()> function)
{
	{
		std::stringstream ss;
		ss << "[START] " << message;
		log(ss.str());
	}

	auto start = std::chrono::system_clock::now();
	function();
	auto end	  = std::chrono::system_clock::now();
	auto duration = std::chrono::duration<double>(end - start);

	{
		std::stringstream ss;
		ss.precision(6);
		ss << "[DONE ] " << message << ", time=" << std::fixed << duration.count() << " (sec)";
		log(ss.str());
	}
}

char		OX(bool cond) { return (cond) ? 'O' : 'X'; }
std::string SIUnit(size_t const byte)
{
	std::stringstream ss;
	ss << std::fixed << std::showpoint << std::setprecision(3);
	if (byte < (1L << 10)) {
		ss << std::to_string(byte) << "Byte";
		return ss.str();
	} else if (byte < (1L << 20)) {
		ss << double(byte) / double(1L << 10) << "KiB";
	} else if (byte < (1L << 30)) {
		ss << double(byte) / double(1L << 20) << "MiB";
	} else if (byte < (1L << 40)) {
		ss << double(byte) / double(1L << 30) << "GiB";
	} else if (byte < (1L << 50)) {
		ss << double(byte) / double(1L << 40) << "TiB";
	} else {
		ss << double(byte) / double(1L << 50) << "PiB";
	}
	return ss.str();
}

void parallelFiber(size_t const workers, std::function<void(size_t const)> func)
{
	std::vector<boost::fibers::fiber> wlist(workers);
	for (size_t i = 0; i < workers; i++) {
		wlist[i] = boost::fibers::fiber([=] { func(i); });
	}

	for (size_t i = 0; i < workers; i++) {
		if (wlist[i].joinable()) {
			wlist[i].join();
		}
	}
}

void parallelThread(size_t const workers, std::function<void(size_t const)> func)
{
	std::vector<std::thread> wlist(workers);
	for (size_t i = 0; i < workers; i++) {
		wlist[i] = std::thread([=] { func(i); });
	}

	for (size_t i = 0; i < workers; i++) {
		if (wlist[i].joinable()) {
			wlist[i].join();
		}
	}
}

size_t ceil(size_t const x, size_t const y) { return (x > 0) ? (1 + ((x - 1) / y)) : 0; }