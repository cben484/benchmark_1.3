cmake_minimum_required(VERSION 3.25)

project(BENCHMARK_CHOLESKY_LU_QR_Algorithms VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_utils_check.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

add_library(benchmark_cholesky SHARED benchmark_cholesky.cu)
add_library(benchmark_lu SHARED benchmark_lu.cu )
add_library(benchmark_qr SHARED benchmark_qr.cu)


target_include_directories(benchmark_cholesky PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})
target_include_directories(benchmark_lu PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})
target_include_directories(benchmark_qr PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})


# c++版本检查
add_executable(version_exec checktheversion.cpp)

# 编译出相应的benchmark_*.exe
add_executable(benchmark_cholesky_exec benchmark_cholesky.cu)
target_link_libraries(benchmark_cholesky_exec benchmark_cholesky CUDA::curand CUDA::cublas CUDA::cusolver) 
set_target_properties(benchmark_cholesky PROPERTIES CUDA_ARCHITECTURES native)


add_executable(benchmark_lu_exec benchmark_lu.cu)
target_link_libraries(benchmark_lu_exec benchmark_lu CUDA::curand CUDA::cublas CUDA::cusolver) 
set_target_properties(benchmark_lu PROPERTIES CUDA_ARCHITECTURES native)

add_executable(benchmark_qr_exec benchmark_qr.cu)
target_link_libraries(benchmark_qr_exec benchmark_qr CUDA::curand CUDA::cublas CUDA::cusolver) 
set_target_properties(benchmark_qr PROPERTIES CUDA_ARCHITECTURES native)


# target_link_libraries(cholesky my_api_shared_lib)

# target_include_directories(convolution_gemm PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
# target_include_directories(convolution_gemm PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
# target_include_directories(convolution_gemm PRIVATE ${CMAKE_SOURCE_DIR}/inc)

# target_link_libraries(convolution_gemm curand cusolver cublas)