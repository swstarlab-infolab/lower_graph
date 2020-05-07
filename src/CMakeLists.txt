set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
#set(CMAKE_CXX_FLAGS "-Wall -O0 -std=c++17 -g")

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})

add_subdirectory(GridCSR)
add_subdirectory(BuddySystem)
add_subdirectory(Converter)
add_subdirectory(TriangleCounting)