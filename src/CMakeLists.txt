set(CMAKE_CXX_COMPILER ${CXX})
#set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++17")
set(CMAKE_CXX_FLAGS "-Wall -Og -std=c++17 -g3")

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})

add_definitions(-DBOOST_ALL_NO_LIB)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS thread system)

add_subdirectory(BuddySystem)
add_subdirectory(Converter)
add_subdirectory(GridCSR)
add_subdirectory(TriangleCounting)