set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -Og -std=c++17 -g0")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/boost_1_73_0/include
    #/opt/at13.0/lib/gcc/powerpc64le-linux-gnu/9.3.1/include
)

link_directories(
    ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}
    /usr/local/boost_1_73_0/lib
)

add_subdirectory(BuddySystem)
add_subdirectory(Adj6ToGCSR)
add_subdirectory(Adj6ToEL32)
add_subdirectory(Adj6ToGCSR-Shard)
add_subdirectory(Adj6ToGCSR-Quad)
add_subdirectory(GridCSR)
add_subdirectory(TriangleCounting)
