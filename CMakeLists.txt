cmake_minimum_required(VERSION 3.0)
project(GridCSR)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND CUDA_NVCC_FLAGS
    -gencode arch=compute_70,code=sm_70
    -O3 -std=c++14)

foreach(_ITEM adj6-gcsr)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/converter/${_ITEM}/*)
add_executable(${_ITEM} ${_FILES})
target_link_libraries(${_ITEM} pthread tbb stdc++fs)
endforeach(_ITEM)

foreach(_ITEM tc)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/query/${_ITEM}/*)
cuda_add_executable(${_ITEM} ${_FILES})
target_link_libraries(${_ITEM} pthread tbb stdc++fs)
endforeach(_ITEM)