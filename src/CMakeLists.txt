set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -Og -std=c++17 -g0")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/boost/include
    /usr/local/gdrcopy/include
    /usr/local/rmm/include
    /usr/lib/gcc/powerpc64le-linux-gnu/7/include
    /usr/local/tbb/include
    /usr/local/jemalloc/include
)

link_directories(
    ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}
    /usr/local/boost/lib
    /usr/local/gdrcopy/lib
    /usr/local/rmm/lib
    /usr/local/tbb/lib
    /usr/local/jemalloc/lib
)

add_subdirectory(Adj6ToGCSR)
add_subdirectory(Adj6ToEL32)
add_subdirectory(Adj6ToGCSR-Quad)
add_subdirectory(BuddySystem)
add_subdirectory(GridCSR)
add_subdirectory(TriangleCounting)
