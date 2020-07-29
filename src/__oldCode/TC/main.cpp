#include "context.h"
#include "data.h"
#include "exec.h"
#include "sched.h"
#include "type.h"
#include "util.h"

#include <iostream>
#include <unordered_map>

static sp<bchan<JobResult>> mergeJobResult(sp<std::vector<sp<bchan<JobResult>>>> in)
{
	auto fibers = makeSp<std::vector<std::thread>>(in->size());
	auto out	= makeSp<bchan<JobResult>>(16);

	for (size_t i = 0; i < in->size(); i++) {
		fibers->at(i) = std::thread([=] {
			for (auto & res : *(in->at(i))) {
				out->push(res);
			}
		});
	}

	std::thread([=] {
		for (auto & f : *fibers) {
			if (f.joinable()) {
				f.join();
			}
		}
	}).detach();

	return out;
}

int main(int argc, char * argv[])
{
	// Parse user input
	auto ctx = makeSp<Context>();
	ctx->parse(argc, argv);

	if (ctx->verbose) {
		ctx->printCUDAInfo();
	}

	if (ctx->verbose) {
		printf("GPU=%ld, STREAM=%ld, BLOCK=%ld, THREAD=%ld, GRIDWIDTH=%ld\n",
			   ctx->cuda.size(),
			   ctx->streams,
			   ctx->blocks,
			   ctx->threads,
			   GRID_WIDTH);
	}

	// auto data = makeSp<std::vector<Data::Manager>>(ctx->cuda.size() * ctx->streams);

	Data::init(ctx);

	auto exec = makeSp<std::vector<Exec::Manager>>(ctx->cuda.size() * ctx->streams);
	for (size_t gpu = 0; gpu < ctx->cuda.size(); gpu++) {
		for (size_t s = 0; s < ctx->streams; s++) {
			auto idx	  = gpu * ctx->streams + s;
			exec->at(idx) = Exec::Manager();
			exec->at(idx).init(ctx, gpu);
		}
	}

	auto sched = makeSp<Sched::Manager>();

	Data::run(ctx);

	Count totalTriangle = 0UL;

	stopwatch("Triangle Counting", [=, &totalTriangle]() {
		// Launch
		auto jobQueue = sched->run(ctx);
		auto jobResults =
			makeSp<std::vector<sp<bchan<JobResult>>>>(ctx->cuda.size() * ctx->streams);
		for (size_t gpu = 0; gpu < ctx->cuda.size(); gpu++) {
			for (size_t s = 0; s < ctx->streams; s++) {
				auto idx			= gpu * ctx->streams + s;
				jobResults->at(idx) = exec->at(idx).run(ctx, jobQueue);
			}
		}

		// Wait
		for (auto & res : *(mergeJobResult(jobResults))) {
			printf("%s, %s, %s = %llu\n",
				   res.job[0].c_str(),
				   res.job[1].c_str(),
				   res.job[2].c_str(),
				   res.triangle);
			totalTriangle += res.triangle;
		}
	});

	printf("totalTriangle: %llu\n", totalTriangle);

	return 0;
}