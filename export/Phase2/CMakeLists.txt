set(outFileName Adj6ToCSR-Phase2)

file(GLOB_RECURSE files ${CMAKE_CURRENT_SOURCE_DIR}/*)
add_executable(${outFileName} ${files})
target_link_libraries(${outFileName} pthread tbb stdc++fs boost_fiber boost_context)
