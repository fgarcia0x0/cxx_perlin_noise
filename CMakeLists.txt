cmake_minimum_required(VERSION 3.18)
project(cxx_perlin_noise VERSION 0.0.1 LANGUAGES CXX)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE Generator: ${CMAKE_GENERATOR}")

# export compile commands to intelli-sense use
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(perlin_noise STATIC perlin.hpp)
set_target_properties(perlin_noise PROPERTIES LINKER_LANGUAGE CXX)

add_executable(perlin_test perlin_test.cpp)
target_link_libraries(perlin_test PUBLIC perlin_noise)

if (NOT MSVC)
    target_compile_options(perlin_noise PRIVATE
        -Wall
        -Wpedantic
        -Wextra
        -Wno-unused-function
        -Wno-unused-parameter
    )
endif()

set_target_properties(perlin_noise perlin_test PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)