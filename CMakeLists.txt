cmake_minimum_required(VERSION 3.21)

# Enable HIP language if requested
if(DEFINED HECBENCH_ENABLE_HIP AND HECBENCH_ENABLE_HIP)
    project(HeCBench
        LANGUAGES C CXX HIP
        VERSION 1.0.0
        DESCRIPTION "Heterogeneous Computing Benchmark Suite"
    )
else()
    project(HeCBench
        LANGUAGES C CXX
        VERSION 1.0.0
        DESCRIPTION "Heterogeneous Computing Benchmark Suite"
    )
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3rd party
find_package(Boost)
find_package(MPI)
find_package(GDAL CONFIG)
find_package(Eigen3)
find_package(BZip2)

# Build options - which programming models to enable
option(HECBENCH_ENABLE_CUDA "Enable CUDA benchmarks" ON)
option(HECBENCH_ENABLE_HIP "Enable HIP benchmarks" ON)
option(HECBENCH_ENABLE_SYCL "Enable SYCL benchmarks" ON)
option(HECBENCH_ENABLE_OPENMP "Enable OpenMP benchmarks" ON)

# Architecture options
set(HECBENCH_CUDA_ARCH "sm_80" CACHE STRING "CUDA architecture (e.g., sm_60, sm_70, sm_80, sm_90)")
set(HECBENCH_HIP_ARCH "gfx90a" CACHE STRING "HIP architecture (e.g., gfx908, gfx90a, gfx942)")

# Build configuration
option(HECBENCH_ENABLE_TESTING "Enable testing" ON)
option(HECBENCH_BUILD_ALL_BENCHMARKS "Build all benchmarks (vs selective)" ON)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Find programming model compilers/libraries
if(HECBENCH_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(NCCL MODULE)
    message(STATUS "CUDA enabled - Architecture: ${HECBENCH_CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${HECBENCH_CUDA_ARCH})
endif()

if(HECBENCH_ENABLE_HIP)
    find_package(HIP MODULE)
    if(HIP_FOUND)
        message(STATUS "HIP enabled - Architecture: ${HECBENCH_HIP_ARCH}")
    else()
        message(WARNING "HIP requested but not found - HIP benchmarks will be skipped")
        set(HECBENCH_ENABLE_HIP OFF)
    endif()
endif()

if(HECBENCH_ENABLE_SYCL)
    find_package(SYCL MODULE)
    if(SYCL_FOUND)
        message(STATUS "SYCL enabled - Compiler: ${SYCL_COMPILER}")
    else()
        message(WARNING "SYCL requested but not found - SYCL benchmarks will be skipped")
        set(HECBENCH_ENABLE_SYCL OFF)
    endif()
endif()

if(HECBENCH_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP requested but not found - OpenMP benchmarks will be skipped")
        set(HECBENCH_ENABLE_OPENMP OFF)
    endif()
endif()

# Include benchmark macros
include(BenchmarkMacros)

# Enable testing if requested
if(HECBENCH_ENABLE_TESTING)
    enable_testing()
    include(CTest)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    message(STATUS "CTest enabled - use 'ctest' to run benchmarks")
    message(STATUS "Python3 for tests: ${Python3_EXECUTABLE}")
endif()

# Add src directory with all benchmarks
add_subdirectory(src)

# Summary
message(STATUS "")
message(STATUS "=== HeCBench Build Configuration ===")
message(STATUS "CUDA:    ${HECBENCH_ENABLE_CUDA}")
if(HECBENCH_ENABLE_CUDA)
    message(STATUS "  Architecture: ${HECBENCH_CUDA_ARCH}")
endif()
message(STATUS "HIP:     ${HECBENCH_ENABLE_HIP}")
if(HECBENCH_ENABLE_HIP)
    message(STATUS "  Architecture: ${HECBENCH_HIP_ARCH}")
endif()
message(STATUS "SYCL:    ${HECBENCH_ENABLE_SYCL}")
message(STATUS "OpenMP:  ${HECBENCH_ENABLE_OPENMP}")
message(STATUS "Testing: ${HECBENCH_ENABLE_TESTING}")
message(STATUS "====================================")
message(STATUS "")
