#pragma once

#include <vector>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <array>
#include <type_traits>
#include <functional>
#include <stdio.h>

template <typename Type>
struct Channel {
public:
    struct __Elem {
        Type __data;
        bool __ok;
    };

    moodycamel::BlockingConcurrentQueue<__Elem> __q;
};

template <typename Type>
void operator<<(Channel<Type> & c, Type const & in) {
    c.__q.enqueue(typename Channel<Type>::__Elem{in, true});
}

template <typename Type>
bool operator<<(Type & out, Channel<Type> & c) {
    typename Channel<Type>::__Elem tmp = {0, };
    c.__q.wait_dequeue(tmp);
    out = tmp.__data;
    return tmp.__ok;
}

template <typename Type>
void ForChan(Type & elem, Channel<Type> & chan, std::function<void(void)> && func) {
    while (true) {
        if (!(elem << chan)) {
            break;
        } else {
            func();
        }
    }
}

template <typename ... Types>
constexpr auto ArgMerge(Types&& ... types)
    -> std::array<std::common_type_t<Types...>, sizeof...(types)>
{
    return { std::forward<Types>(types)... };
}

template <typename Type>
void Close(Channel<Type>& c) {
    c.__q.enqueue(typename Channel<Type>::__Elem{0,});
}