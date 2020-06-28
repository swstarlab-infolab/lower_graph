set(outFileName Adj6ToCSR-Phase1)

add_custom_target(${outFileName} ALL
    COMMENT "Golang build ${outFileName}"
    COMMAND go build -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${outFileName} ${CMAKE_CURRENT_SOURCE_DIR}/*.go)