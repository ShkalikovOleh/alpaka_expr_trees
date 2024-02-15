cmake_minimum_required(VERSION 3.20)
enable_testing()
project(alpaka_expr_trees LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(thirdParty/alpaka)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Werror")

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(include)
add_subdirectory(example)
add_subdirectory(tests)
