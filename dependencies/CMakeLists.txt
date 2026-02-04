include (CPM)

### LIBARIES

# --- BLUELIB
if (NOT ${BLUELIB_LOCALLY})

	message (STATUS "DISABLED - Locally library BLUE")
	
	FetchContent_Declare (
		BLUELIB
		GIT_REPOSITORY https://github.com/dotBlueShoes/blue.git
		GIT_TAG master
	)
	
	FetchContent_MakeAvailable (BLUELIB)

else ()

	message (STATUS "ENABLED - Locally library BLUE")
	
	# --- Define 'bluelib'
	set (BLUELIB_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/bluelib/include)
	add_library (BLUELIB INTERFACE)
	target_include_directories (BLUELIB INTERFACE ${BLUELIB_INC_DIR})

endif ()

# --- EIGEN
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
)
FetchContent_MakeAvailable(eigen)

# --- STB
set (STB_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/stb)
add_library (STB INTERFACE)
target_include_directories (STB INTERFACE ${STB_INC_DIR})

# # --- GLM
# set (GLM_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/glm/1.0.1)
# add_library (GLM INTERFACE)
# target_include_directories (GLM INTERFACE ${GLM_INC_DIR})
