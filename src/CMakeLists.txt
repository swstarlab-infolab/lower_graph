set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17 -mavx2")
#set(CMAKE_CXX_FLAGS "-Wall -Og -std=c++17 -g0")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/boost_1_73_0/include
)

link_directories(
    ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}
    /usr/local/boost_1_73_0/lib
)

add_subdirectory(BuddySystem)
add_subdirectory(Converter)
add_subdirectory(Adj6ToCSR)
add_subdirectory(IntrinsicTest)
add_subdirectory(ReadTestCPP)
add_subdirectory(GridCSR)
add_subdirectory(TriangleCounting)