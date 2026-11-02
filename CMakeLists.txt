cmake_minimum_required(VERSION 3.20)
project(cuda_cnn_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The CPU reference and its tests build anywhere; the CUDA targets need nvcc.
add_library(reference STATIC src/reference.cpp)
target_include_directories(reference PUBLIC include)
target_compile_options(reference PRIVATE -Wall -Wextra)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE reference)
enable_testing()
add_test(NAME reference COMMAND test_reference)

option(WITH_CUDA "Build the CUDA kernels, GPU tests and benchmark" ON)
include(CheckLanguage)
check_language(CUDA)
if(WITH_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # T4 (Colab free tier) = 75, A100 = 80, RTX 30xx = 86, L4 / RTX 40xx = 89, H100 = 90
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
  endif()
  find_package(CUDAToolkit REQUIRED)

  add_library(kernels STATIC src/gemm.cu src/conv.cu)
  target_link_libraries(kernels PUBLIC reference CUDA::cudart)
  target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  add_executable(test_kernels tests/test_kernels.cu)
  target_link_libraries(test_kernels PRIVATE kernels)
  add_test(NAME kernels COMMAND test_kernels)

  add_executable(bench bench/bench.cu)
  target_link_libraries(bench PRIVATE kernels CUDA::cublas)
else()
  message(STATUS "CUDA compiler not found: building only the CPU reference and its tests")
endif()
