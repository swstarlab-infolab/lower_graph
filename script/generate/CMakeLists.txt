add_custom_target(
	script-generate ALL
	COMMENT "Golang build script-generate" 
	COMMAND go build ${CMAKE_CURRENT_SOURCE_DIR})