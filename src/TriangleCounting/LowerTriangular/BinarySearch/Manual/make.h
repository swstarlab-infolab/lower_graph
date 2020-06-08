#ifndef D49D3D8E_9143_4B4F_A153_3DE8772A2896
#define D49D3D8E_9143_4B4F_A153_3DE8772A2896

#include "type.h"

#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <type_traits>
#include <unordered_map>

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

/*
template <typename Type, typename... Args>
auto make(Args&&... args)
{
	return std::shared_ptr<Type>(
		[&] {
			Type* p = new Type(std::forward<Args>(args)...);
			printf("NEW: %p\n", p);
			return p;
		}(),
		[](Type* p) {
			printf("DEL: %p\n", p);
			if (p != nullptr) {
				delete p;
			}
		});
}

template <typename Type, typename... Args>
auto makeArray(Args&&... args)
{
	using MyType = std::array<Type, sizeof...(args)>;
	return std::shared_ptr<MyType>(
		[&] {
			MyType* p = new MyType { { std::forward<Args>(args)... } };
			return p;
		}(),
		[](MyType* p) {
			if (p != nullptr) {
				delete p;
			}
		});
}
*/

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
auto allocCUDA(size_t const count)
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

auto allocCUDAByte(size_t const byte)
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

template <typename Type>
auto allocHost(size_t const count)
{
	return std::shared_ptr<Type>(
		[&] {
			Type * p;
			cudaHostAlloc((void **)&p, count * sizeof(Type), cudaHostAllocPortable);
			return p;
		}(),
		[](Type * p) {
			if (p != nullptr) {
				cudaFree(p);
			}
		});
}

auto allocHostByte(size_t const byte)
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

/*
template <class Key,
	class Value,
	typename Hash = std::hash<Key>,
	typename Equal = std::equal_to<Key>>
auto makeHashMap(
	Hash const& hashFunction,
	Equal const& equalFunction)
{
	return std::unordered_map<Key, Value, hashFunction,
equalFunction>(bucketCount);
}
*/

#endif /* D49D3D8E_9143_4B4F_A153_3DE8772A2896 */
