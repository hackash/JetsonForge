
cmake_minimum_required(VERSION 3.21)
project(cuda_hello LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

# Try to find the CUDA toolkit in the sysroot
find_package(CUDAToolkit REQUIRED)

add_executable(cuda_hello src/main.cu)
target_include_directories(cuda_hello PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_hello PRIVATE CUDA::cudart)

# Helpful for cross-compiles where RPATH isn't needed at build-time
set_target_properties(cuda_hello PROPERTIES BUILD_RPATH "")
