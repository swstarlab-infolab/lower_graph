#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <malloc.h>

int main(int argc, char * argv[])
{
	/*
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <inFolder>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	auto inFolder = fs::absolute(fs::path(std::string(argv[1]) + "/").parent_path().string() + "/");
	*/

	constexpr size_t count = (1 << 30);

	auto arrayA = (int *)aligned_alloc(32, count * sizeof(int));
	auto arrayB = (int *)aligned_alloc(32, count * sizeof(int));
	auto arrayR = (int *)aligned_alloc(32, count * sizeof(int));

	for (size_t i = 0; i < count; i++) {
		arrayA[i] = i;
	}

	for (size_t i = 0; i < count; i++) {
		arrayB[i] = 1;
	}

	{
		auto start = std::chrono::system_clock::now();

		for (size_t i = 0; i < count; i++) {
			arrayR[i] = arrayA[i] + arrayB[i];
		}

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		printf("No SIMD, Time: %lf (sec)\n", elapsed.count());
	}

	{
		auto start = std::chrono::system_clock::now();

		for (size_t i = 0; i < count; i += (128 / (sizeof(int) * 8))) {
			__m128i xmm0 = _mm_load_si128((__m128i *)&arrayA[i]);
			__m128i xmm1 = _mm_load_si128((__m128i *)&arrayB[i]);
			__m128i xmmR = _mm_add_epi32(xmm0, xmm1);
			_mm_store_si128((__m128i *)&arrayR[i], xmmR);
		}

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		printf("SSE2   , Time: %lf (sec)\n", elapsed.count());
	}

	{
		auto start = std::chrono::system_clock::now();

		for (size_t i = 0; i < count; i += (256 / (sizeof(int) * 8))) {
			__m256i xmm0 = _mm256_load_si256((__m256i *)&arrayA[i]);
			__m256i xmm1 = _mm256_load_si256((__m256i *)&arrayB[i]);
			__m256i xmmR = _mm256_add_epi32(xmm0, xmm1);
			_mm256_store_si256((__m256i *)&arrayR[i], xmmR);
		}

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		printf("AVX2   , Time: %lf (sec)\n", elapsed.count());
	}

	return 0;
}