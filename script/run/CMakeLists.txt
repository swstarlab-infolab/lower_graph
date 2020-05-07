add_custom_target(
	script-run ALL
	COMMENT "Golang build script-run"
	COMMAND go build ${CMAKE_CURRENT_SOURCE_DIR})