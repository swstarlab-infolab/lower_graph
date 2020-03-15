cmake_minimum_required(VERSION 3.0)
project(GridCSR)

#set(CUDA_STREAMS)
#set(CUDA_BLOCKS)
#set(CUDA_THREADS)

#add_definitions(-DCUDA_STREAMS=${CUDA_STREAMS})
#add_definitions(-DCUDA_BLOCKS=${CUDA_BLOCKS})
#add_definitions(-DCUDA_THREADS=${CUDA_THREADS})

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -O0 -std=c++17 -g")

include_directories(${CMAKE_SOURCE_DIR}/include)
if(CMAKE_SYSTEM_PROCESSOR EQUAL ppc64le)
	link_directories(${CMAKE_SOURCE_DIR}/lib-ppc64le)
elseif(CMAKE_SYSTEM_PROCESSOR EQUAL x86_64)
	link_directories(${CMAKE_SOURCE_DIR}/lib-x86_64)
endif()

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND CUDA_NVCC_FLAGS
    -gencode arch=compute_70,code=sm_70
	#-O0 -std=c++14 -g -lineinfo)
    -O3 -std=c++14)

foreach(_ITEM meta)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
add_library(${_ITEM} ${_FILES})
endforeach(_ITEM)

foreach(_ITEM converter)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
add_executable(${_ITEM} ${_FILES})
add_dependencies(${_ITEM} meta)
target_link_libraries(${_ITEM} pthread tbb stdc++fs meta)
endforeach(_ITEM)

foreach(_ITEM triangle-new)
file(GLOB_RECURSE _FILES ${CMAKE_SOURCE_DIR}/src/${_ITEM}/*)
cuda_add_executable(${_ITEM} ${_FILES})
add_dependencies(${_ITEM} meta)
target_link_libraries(${_ITEM} pthread tbb stdc++fs meta)
endforeach(_ITEM)

add_custom_target(
	gen ALL
	COMMENT "Golang build gen"
	COMMAND go build ${CMAKE_SOURCE_DIR}/script/gen.go)
add_custom_target(
	run ALL
	COMMENT "Golang build run"
	COMMAND go build ${CMAKE_SOURCE_DIR}/script/run.go)
