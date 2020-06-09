#ifndef __GridCSR_CUDA_Kernel_cuh__
#define __GridCSR_CUDA_Kernel_cuh__

namespace GridCSR
{
namespace CUDA
{

__device__ uint32_t ulog2floor(uint32_t x)
{
	uint32_t r, q;
	r = (x > 0xFFFF) << 4;
	x >>= r;
	q = (x > 0xFF) << 3;
	x >>= q;
	r |= q;
	q = (x > 0xF) << 2;
	x >>= q;
	r |= q;
	q = (x > 0x3) << 1;
	x >>= q;
	r |= q;

	return (r | (x >> 1));
}

__device__ void BinarySearchIntersection(Vertex const *		  Arr,
										 uint32_t const		  ArrLen,
										 Vertex const		  candidate,
										 unsigned long long * count)
{
	auto const maxLevel = ulog2floor(ArrLen);

	int now = (ArrLen - 1) >> 1;

	for (uint32_t level = 0; level <= maxLevel; level++) {
		auto const movement = 1 << (maxLevel - level - 1);

		if (now < 0) {
			now += movement;
		} else if (ArrLen <= now) {
			now -= movement;
		} else {
			if (Arr[now] < candidate) {
				now += movement;
			} else if (candidate < Arr[now]) {
				now -= movement;
			} else {
				(*count)++;
				break;
			}
		}
	}
}

__device__ int
BinarySearchPosition(Vertex const * Arr, uint32_t const ArrLen, Vertex const candidate)
{
	// auto const maxLevel = uint32_t(ceil(log2(ArrLen + 1))) - 1;
	// ceil(log2(a)) == floor(log2(a-1))+1
	auto const maxLevel = ulog2floor(ArrLen);

	int now = (ArrLen - 1) >> 1;

	for (uint32_t level = 0; level <= maxLevel; level++) {
		auto const movement = 1 << (maxLevel - level - 1);

		if (now < 0) {
			now += movement;
		} else if (ArrLen <= now) {
			now -= movement;
		} else {
			if (Arr[now] < candidate) {
				now += movement;
			} else if (candidate < Arr[now]) {
				now -= movement;
			} else {
				return now;
			}
		}
	}

	return -1;
}

} // namespace CUDA
} // namespace GridCSR

#endif