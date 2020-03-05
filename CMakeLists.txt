cmake_minimum_required(VERSION 3.0)
project(GridCSR)

set(CUDA_STREAMS)
set(CUDA_BLOCKS)
set(CUDA_THREADS)
set(FORMAT_GRID_POWER)

add_definitions(-DCUDA_STREAMS=${CUDA_STREAMS})
add_definitions(-DCUDA_BLOCKS=${CUDA_BLOCKS})
add_definitions(-DCUDA_THREADS=${CUDA_THREADS})
add_definitions(-DFORMAT_GRID_POWER=${FORMAT_GRID_POWER})

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

foreach(_ITEM dataformat)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
add_library(${_ITEM} ${_FILES})
target_link_libraries(${_ITEM} pthread tbb stdc++fs)
endforeach(_ITEM)

foreach(_ITEM converter newconverter)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
add_executable(${_ITEM} ${_FILES})
target_link_libraries(${_ITEM} pthread tbb stdc++fs dataformat)
endforeach(_ITEM)

foreach(_ITEM query-tc)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
cuda_add_executable(${_ITEM} ${_FILES})
target_link_libraries(${_ITEM} pthread tbb stdc++fs dataformat)
endforeach(_ITEM)