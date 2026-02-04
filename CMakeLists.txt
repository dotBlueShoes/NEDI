cmake_minimum_required (VERSION 3.27 FATAL_ERROR)

set (CMAKE_SUPPRESS_REGENERATION true)
set (CMAKE_VERBOSE_MAKEFILE OFF)

# --- Name, Version, Language compilers in use.
project (upscaling-algorithms VERSION 1.0.0 LANGUAGES C CXX CUDA)

# --- Includes
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include (properties)

# --- Configurable Properties
option (BLUELIB_LOCALLY "Link 'bluelib' sources from local cache" OFF)

# --- Dependencies
add_subdirectory (dependencies)

# --- Add 'defines' to all projects
add_compile_definitions (_CRT_SECURE_NO_WARNINGS)
add_compile_definitions (DEBUG_FLAG_LOGGING=${DEBUG_FLAG_LOGGING})
add_compile_definitions (DEBUG_FLAG_MEMORY=${DEBUG_FLAG_MEMORY})
add_compile_definitions (DEBUG_FLAG_CLOCKS=${DEBUG_FLAG_CLOCKS})
add_compile_definitions (DEBUG_FLAG_POSTLOGGING=${DEBUG_FLAG_POSTLOGGING})

# --- Project's sources
add_subdirectory (project/nedi)
add_subdirectory (project/downscale_nn_lt)

