cmake_minimum_required(VERSION 3.18)
project(cuda_boids LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/boids.cu
    src/boids_api.cpp
)

set(HEADERS
    src/boids.hpp
)

add_library(cuda_boids SHARED ${SOURCES} ${HEADERS})

set_target_properties(cuda_boids PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cuda_boids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_boids PROPERTIES CUDA_ARCHITECTURES "60;70;75;86")
set_target_properties(cuda_boids PROPERTIES OUTPUT_NAME "cuda_boids")

target_include_directories(cuda_boids PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# No need for find_package(CUDA) or CUDA::cudart
