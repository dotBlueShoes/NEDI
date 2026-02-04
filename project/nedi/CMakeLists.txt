# --- Define a 'subproject'
project (nedi VERSION 1.0.0 LANGUAGES C CXX CUDA)

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
    src/source.cpp
	src/stb_image_implementation.cpp
    src/main.cu 
)


target_compile_options(nedi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
)

# --- By default the CUDA compiler uses whole-program compilation. 
# Effectively this means that all device functions and variables needed to be located 
#  inside a single file or compilation unit.
set_property (TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# --- LIBS For shared by all presets.
target_link_libraries (${PROJECT_NAME} BLUELIB)
target_link_libraries (${PROJECT_NAME} STB)
#target_link_libraries (${PROJECT_NAME} GLM)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

# ---
target_include_directories (${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# --- Disable CUDA warning C5033: 'register' is no longer supported
target_compile_options (${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler="/wd5033"
    >
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
)

# --- Creates a 'symlink' for output folder in builds directory.
add_custom_command (
	TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink
	${CMAKE_SOURCE_DIR}/project/${PROJECT_NAME}/res
	${CMAKE_CURRENT_BINARY_DIR}/res
)
