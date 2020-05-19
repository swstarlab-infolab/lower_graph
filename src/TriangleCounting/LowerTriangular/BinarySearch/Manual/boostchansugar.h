#pragma once

#include <vector>
#include <type_traits>

#include <boost/fiber/buffered_channel.hpp>


template <typename Type>
void BoostForRange(Type & var, boost::fibers::buffered_channel<Type> & chan, std::function<void()> && func) {
    while (true) {
        switch (chan.pop(var)) {
            case boost::fibers::channel_op_status::success:
                func();
                break;
            case boost::fibers::channel_op_status::closed:
                return;
        }
    }
}

template <typename Type>
void BoostForRangeMerge(Type & var, std::vector<std::shared_ptr<boost::fibers::buffered_channel<Type>>> & chan, std::function<void()> && func) {
    size_t size = chan.size();
    if (size == 0) {
        return;
    }

    size_t completed = 0;
    size_t now = 0;

    while (true) {
        printf("while\n");
        switch (chan[now].get()->pop(var)) {
            case boost::fibers::channel_op_status::success:
                printf("start function \n");
                func();
                break;
            case boost::fibers::channel_op_status::closed:
                completed++;
                printf("completed: %d\n", completed);
                break;
        }

        if (completed == size) {
            break;
        }

        if (now != size - 1) {
            now++;
        } else {
            now = 0;
        }
    }
}