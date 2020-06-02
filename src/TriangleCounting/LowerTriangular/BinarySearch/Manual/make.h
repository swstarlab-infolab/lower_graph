#ifndef D49D3D8E_9143_4B4F_A153_3DE8772A2896
#define D49D3D8E_9143_4B4F_A153_3DE8772A2896

#include "type.h"
#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <tbb/concurrent_hash_map.h>
#include <type_traits>

template <typename... Args>
auto makeVector(Args&&... args)
{
    return std::vector<std::common_type_t<Args...>> { std::forward<Args>(args)... };
}

template <typename... Args>
auto makeArray(Args&&... args)
{
    return std::array<std::common_type_t<Args...>, sizeof...(args)> { std::forward<Args>(args)... };
}

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

template <typename ChanType>
auto merge(std::vector<std::shared_ptr<ChanType>>& v)
{
    auto out = make<ChanType>(1 << 4);

    std::thread([&] {
        auto vSize = v.size();

        std::vector<std::thread> waitGroup(vSize);

        for (size_t i = 0; i < vSize; ++i) {
            waitGroup[i] = std::thread([&, i] {
                for (auto& e : *v[i].get()) {
                    out.get()->push(e);
                }
            });
        }

        for (auto& f : waitGroup) {
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
            Type* p;
            cudaMalloc((void**)&p, count * sizeof(Type));
            return p;
        }(),
        [](Type* p) {
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
            Type* p;
            cudaHostAlloc((void**)&p, count * sizeof(Type), cudaHostAllocPortable);
            return p;
        }(),
        [](Type* p) {
            if (p != nullptr) {
                cudaFree(p);
            }
        });
}

/*
template <class Key,
    class Value,
    typename Hash = decltype(tbb::tbb_hash_compare<Key>::hash()),
    typename Equal = decltype(tbb::tbb_hash_compare<Key>::equal())>
auto makeHashMap(
    Hash const& hashFunction,
    Equal const& equalFunction,
    typename tbb::concurrent_hash_map<Key, Value, Hash>::size_type bucketCount = 16384)
{
    tbb::tbb_hash_compare<Key> _tmpstruct;
    _tmpstruct.hash = hashFunction;
    _tmpstruct.equal = equalFunction;
    return tbb::concurrent_hash_map<Key, Value, _tmpstruct>(bucketCount, _tmpstruct);
}
*/

#endif /* D49D3D8E_9143_4B4F_A153_3DE8772A2896 */
