#include "make.cuh"

template <>
std::shared_ptr<void> allocCUDA<void>(size_t const byte)
{
	return std::shared_ptr<void>(
		[&] {
			void * p;
			cudaMalloc((void **)&p, byte);
			return p;
		}(),
		[](void * p) {
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}

template <>
std::shared_ptr<void> allocHost<void>(size_t const byte)
{
	return std::shared_ptr<void>(
		[&] {
			void * p;
			cudaHostAlloc((void **)&p, byte, cudaHostAllocPortable);
			return p;
		}(),
		[](void * p) {
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}