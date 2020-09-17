set(MY_EXE_NAME TriangleCounting-LowerTriangular-BitArray-QuadTree3)

file(GLOB_RECURSE
    MY_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*)

cuda_add_executable(${MY_EXE_NAME} ${MY_SRC_FILES})

#add_dependencies(${MY_EXE_NAME} GridCSR BuddySystem)

target_link_libraries(${MY_EXE_NAME}
    pthread
    stdc++fs
    #boost_fiber
    #boost_context
    #gdrapi
    #cuda
    mysqlclient
    jemalloc
)