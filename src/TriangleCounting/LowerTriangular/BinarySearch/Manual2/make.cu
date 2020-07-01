#include "make.cuh"

#include <sys/mman.h>

int lock_memory(char * addr, size_t size)
{
	unsigned long page_offset, page_size;
	page_size	= sysconf(_SC_PAGE_SIZE);
	page_offset = (unsigned long)addr % page_size;
	addr -= page_offset;
	size += page_offset;
	return mlock(addr, size);
}

int unlock_memory(char * addr, size_t size)
{
	unsigned long page_offset, page_size;
	page_size	= sysconf(_SC_PAGE_SIZE);
	page_offset = (unsigned long)addr % page_size;
	addr -= page_offset;
	size += page_offset;
	return munlock(addr, size);
}

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
			int	   devices = 0;
			cudaGetDeviceCount(&devices);
			if (devices) {
				cudaHostAlloc((void **)&p, byte, cudaHostAllocPortable);
			} else {
				p = malloc(byte);
				lock_memory((char *)p, byte);
			}
			return p;
		}(),
		[byte](void * p) {
			if (p != nullptr) {
				int devices = 0;
				cudaGetDeviceCount(&devices);
				if (devices) {
					cudaFree(p);
				} else {
					unlock_memory((char *)p, byte);
					free(p);
				}
			}
		});
}