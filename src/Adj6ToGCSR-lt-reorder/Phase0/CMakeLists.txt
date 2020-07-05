set(outFileName Adj6ToGCSR-lt-reorder-Phase0)

add_custom_target(${outFileName} ALL
    COMMENT "Golang build ${outFileName}"
    COMMAND go build -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${outFileName} ${CMAKE_CURRENT_SOURCE_DIR}/*.go)