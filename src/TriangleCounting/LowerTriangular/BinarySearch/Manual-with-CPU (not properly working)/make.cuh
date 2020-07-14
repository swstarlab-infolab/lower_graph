#ifndef C5E0387C_9D84_4946_BD11_40F8FD59937A
#define C5E0387C_9D84_4946_BD11_40F8FD59937A

#include "type.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <vector>

/*
#include <array>
#include <type_traits>
template <typename... Args>
auto makeVector(Args &&... args)
{
	return std::vector<std::common_type_t<Args...>>{std::forward<Args>(args)...};
}

template <typename... Args>
auto makeArray(Args &&... args)
{
	return std::array<std::common_type_t<Args...>, sizeof...(args)>{std::forward<Args>(args)...};
}
*/

int lock_memory(char * addr, size_t size);
int unlock_memory(char * addr, size_t size);

template <typename ChanType>
auto merge(std::vector<std::shared_ptr<ChanType>> & v)
{
	// auto out = make<ChanType>(1 << 4);
	auto out = std::make_shared<ChanType>(1 << 4);

	std::thread([&, v, out] {
		auto vSize = v.size();

		std::vector<std::thread> waitGroup(vSize);

		for (size_t i = 0; i < vSize; ++i) {
			waitGroup[i] = std::thread([&, i] {
				for (auto & e : *v[i].get()) {
					out.get()->push(e);
				}
			});
		}

		for (auto & f : waitGroup) {
			if (f.joinable()) {
				f.join();
			}
		}

		out.get()->close();
	}).detach();

	return out;
}

template <typename Type>
std::shared_ptr<Type> allocCUDA(size_t const count)
{
	return std::shared_ptr<Type>(
		[&] {
			Type * p;
			cudaMalloc((void **)&p, count * sizeof(Type));
			return p;
		}(),
		[](Type * p) {
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}

template <typename Type>
std::shared_ptr<Type> allocHost(size_t const count)
{
	return std::shared_ptr<Type>(
		[&] {
			Type * p;
			int	   devices = 0;
			cudaGetDeviceCount(&devices);
			if (devices) {
				cudaHostAlloc((void **)&p, count * sizeof(Type), cudaHostAllocPortable);
			} else {
				p = malloc(count * sizeof(Type));
				lock_memory(p, count * sizeof(Type));
			}
			return p;
		}(),
		[count](Type * p) {
			if (p != nullptr) {
				int devices = 0;
				cudaGetDeviceCount(&devices);
				if (devices) {
					cudaFree(p);
				} else {
					unlock_memory(p, count * sizeof(Type));
					free(p);
				}
			}
		});
}

template <>
std::shared_ptr<void> allocCUDA<void>(size_t const byte);

template <>
std::shared_ptr<void> allocHost<void>(size_t const byte);

#endif /* C5E0387C_9D84_4946_BD11_40F8FD59937A */
