# --- Define a 'subproject'
project (downscale_nn_lt VERSION 1.0.0 LANGUAGES C CXX)

add_compile_options (/wd5033)

# --- Add header files
file (GLOB_RECURSE HEADER_FILES
	${PROJECT_SOURCE_DIR}/inc/*.h
	${PROJECT_SOURCE_DIR}/inc/*.hpp
)

# --- Add needed DEFINE's
add_compile_definitions (_CRT_SECURE_NO_WARNINGS)

# --- Info Build Type
message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if 		(${PROFILE} EQUAL 1)

	message (STATUS "Preset: comp - DEBUG")
	add_compile_definitions (DEBUG_TYPE=${DEBUG_FLAG_LOGGING}+${DEBUG_FLAG_MEMORY})

elseif 	(${PROFILE} EQUAL 2)

	message (STATUS "Preset: comp - RELEASE")
	add_compile_definitions (DEBUG_TYPE=0)

endif ()

# --- Define the executable
add_executable (${PROJECT_NAME} ${HEADER_FILES} 
	src/main.cpp
	src/stb_image_implementation.cpp
)

# --- LIBS For shared by all presets.
target_link_libraries (${PROJECT_NAME} BLUELIB)
target_link_libraries (${PROJECT_NAME} STB)

# ---
target_include_directories (${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# --- Creates a 'symlink' for output folder in builds directory.
add_custom_command (
	TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink
	${CMAKE_SOURCE_DIR}/project/${PROJECT_NAME}/res
	${CMAKE_CURRENT_BINARY_DIR}/res
)
