project(Converter)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/boost_1_73_0/include
)

link_directories(
    ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}
    /usr/local/boost_1_73_0/lib
)

add_subdirectory(Phase1)
add_subdirectory(Phase2)
add_subdirectory(Phase3)