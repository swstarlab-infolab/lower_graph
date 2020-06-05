set(CMAKE_CXX_COMPILER ${CXX})
#set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
set(CMAKE_CXX_FLAGS "-Wall -Og -std=c++17 -g3")

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
add_subdirectory(GridCSR)
add_subdirectory(TriangleCounting)