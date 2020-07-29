#include "util.h"

#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdio.h>
//#include <tbb/blocked_range.h>
//#include <tbb/parallel_for.h>
//#include <tbb/task_arena.h>
#include <thread>

static std::mutex __logMtx;

size_t ceil(size_t const x, size_t const y)
{
	if (x > 0) {
		return 1 + ((x - 1) / y);
	} else {
		return 0;
	}
}

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

uint64_t be6_le8(uint8_t * in)
{
	uint64_t out = 0;
	out |= in[0];
	for (uint8_t i = 1; i < 6; i++) {
		out <<= 8;
		out |= in[i];
	}
	return out;
}

sp<bchan<fs::path>>
fileListOver(fs::path const & folder, std::string const & extension, size_t const over)
{
	auto out = makeSp<bchan<fs::path>>(16);
	std::thread([=] {
		// recursive iteration
		for (fs::recursive_directory_iterator iter(folder), end; iter != end; iter++) {
			// check file is not directory and size is not zero
			if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) > over) {
				if (extension != "" && extension != iter->path().extension()) {
					continue;
				}

				out->push(fs::absolute(iter->path()));
			}
		}
		out->close();
	}).detach();
	return out;
}

sp<bchan<fs::path>> fileList(fs::path const & folder, std::string const & extension)
{
	auto out = makeSp<bchan<fs::path>>(16);
	std::thread([=] {
		// recursive iteration
		for (fs::recursive_directory_iterator iter(folder), end; iter != end; iter++) {
			// check file is not directory and size is not zero
			if (fs::is_regular_file(iter->status()) && fs::file_size(iter->path()) != 0) {
				if (extension != "" && extension != iter->path().extension()) {
					continue;
				}

				out->push(fs::absolute(iter->path()));
			}
		}
		out->close();
	}).detach();
	return out;
}

sp<bchan<RowPos>> splitAdj6(sp<std::vector<uint8_t>> adj6)
{
	auto out = makeSp<bchan<RowPos>>(16);
	std::thread([=] {
		for (size_t i = 0; i < adj6->size();) {
			RowPos rPos;
			rPos.src = be6_le8(&(adj6->at(i)));
			i += 6;
			rPos.cnt = be6_le8(&(adj6->at(i)));
			i += 6;
			rPos.dstStart = i;
			out->push(rPos);
			i += (6 * rPos.cnt);
		}
		out->close();
	}).detach();
	return out;
}

std::string fileNameEncode(E32 const & grid, std::string const & ext)
{
	return std::to_string(grid[0]) + "-" + std::to_string(grid[1]) + ext;
}

void parallelDo(size_t const workers, std::function<void(size_t const)> func)
{
	/*
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0L, workers, 1L),
		[&](tbb::blocked_range<size_t> const & r) {
			for (size_t i = r.begin(); i != r.end(); i++) {
				func(i);
			}
		},
		tbb::static_partitioner());
		*/
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