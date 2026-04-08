cmake_minimum_required(VERSION 3.21)

# Options that affect enabled languages must appear before project().
option(HECBENCH_ENABLE_CUDA "Enable CUDA benchmarks" ON)
option(HECBENCH_ENABLE_HIP "Enable HIP benchmarks" ON)
option(HECBENCH_ENABLE_SYCL "Enable SYCL benchmarks" ON)
option(HECBENCH_ENABLE_OPENMP "Enable OpenMP benchmarks" ON)

option(HECBENCH_ENABLE_HIP_ONLY
  "Build ONLY HIP benchmarks (disables CUDA, SYCL, OpenMP)"
  OFF)
if(HECBENCH_ENABLE_HIP_ONLY)
  if(HECBENCH_ENABLE_CUDA OR HECBENCH_ENABLE_SYCL OR HECBENCH_ENABLE_OPENMP)
    message(WARNING
      "HECBENCH_ENABLE_HIP_ONLY is ON: CUDA, SYCL, and OpenMP benchmarks are disabled for this build. "
      "Cache entries HECBENCH_ENABLE_CUDA/SYCL/OPENMP may still show ON until you change them; effective behavior is HIP-only.")
  endif()
  set(HECBENCH_ENABLE_CUDA OFF)
  set(HECBENCH_ENABLE_SYCL OFF)
  set(HECBENCH_ENABLE_OPENMP OFF)
  set(HECBENCH_ENABLE_HIP ON)
endif()

set(_hecbench_project_langs C CXX)
if(HECBENCH_ENABLE_HIP)
  list(APPEND _hecbench_project_langs HIP)
endif()

project(HeCBench
  LANGUAGES ${_hecbench_project_langs}
  VERSION 1.0.0
  DESCRIPTION "Heterogeneous Computing Benchmark Suite"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3rd party
find_package(Boost)
find_package(MPI)
find_package(GDAL CONFIG)
find_package(GSL)
find_package(BZip2)

# Architecture options
set(HECBENCH_CUDA_ARCH "sm_80" CACHE STRING "CUDA architecture (e.g., sm_60, sm_70, sm_80, sm_90)")
set(HECBENCH_HIP_ARCH "gfx90a" CACHE STRING "HIP architecture (e.g., gfx908, gfx90a, gfx942)")

option(HECBENCH_ENABLE_SPIRV "Enable SPIR-V mode for HIP benchmarks" OFF)
set(HECBENCH_SPIRV_TARGET "amdgcnspirv" CACHE STRING "SPIR-V offload target")
set(HECBENCH_HIP_EXTRA_CFLAGS "" CACHE STRING "Extra HIP/C++ compile flags")
set(HECBENCH_HIP_LDFLAGS "" CACHE STRING "Extra linker flags for HIP benchmarks")

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
    # SPIR-V: avoid defaulting CMAKE_HIP_ARCHITECTURES to gfx* while SPIR-V flags use ${HECBENCH_SPIRV_TARGET}.
    if(HECBENCH_ENABLE_SPIRV)
        set(CMAKE_HIP_ARCHITECTURES "${HECBENCH_SPIRV_TARGET}")
    else()
        set(CMAKE_HIP_ARCHITECTURES "${HECBENCH_HIP_ARCH}")
    endif()
    find_package(HIP MODULE)
    if(NOT HIP_FOUND)
        message(WARNING
            "FindHIP did not locate hipcc/ROCm in standard locations or PATH. "
            "HIP benchmarks may still build if CMAKE_HIP_COMPILER points to a HIP-capable toolchain.")
    endif()
    message(STATUS "HIP enabled - HECBENCH_HIP_ARCH=${HECBENCH_HIP_ARCH} (CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES})")
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
    find_package(OpenMP)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP requested but not found - OpenMP benchmarks will be skipped")
        set(HECBENCH_ENABLE_OPENMP OFF)
    endif()
endif()

if(HECBENCH_ENABLE_HIP)
    if(HECBENCH_ENABLE_SPIRV)
        if(NOT HECBENCH_HIP_EXTRA_CFLAGS)
            set(HECBENCH_HIP_EXTRA_CFLAGS "-x hip --offload-arch=${HECBENCH_SPIRV_TARGET} -Wall")
        endif()
        if(NOT HECBENCH_HIP_LDFLAGS)
            set(HECBENCH_HIP_LDFLAGS "--hip-link --offload-arch=${HECBENCH_SPIRV_TARGET}")
        endif()
    endif()
    if(HECBENCH_HIP_EXTRA_CFLAGS)
        separate_arguments(_hecbench_hip_extra_cflags NATIVE_COMMAND "${HECBENCH_HIP_EXTRA_CFLAGS}")
        foreach(_flag IN LISTS _hecbench_hip_extra_cflags)
            add_compile_options($<$<COMPILE_LANGUAGE:HIP>:${_flag}>)
        endforeach()
    endif()
    if(HECBENCH_HIP_LDFLAGS)
        separate_arguments(_hecbench_hip_ldflags NATIVE_COMMAND "${HECBENCH_HIP_LDFLAGS}")
        foreach(_flag IN LISTS _hecbench_hip_ldflags)
            add_link_options($<$<LINK_LANGUAGE:HIP>:${_flag}>)
        endforeach()
    endif()
endif()

include(BenchmarkMacros)

if(HECBENCH_ENABLE_TESTING)
    enable_testing()
    include(CTest)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    message(STATUS "CTest enabled - use 'ctest' to run benchmarks")
    message(STATUS "Python3 for tests: ${Python3_EXECUTABLE}")
endif()

add_subdirectory(src)

message(STATUS "")
message(STATUS "=== HeCBench Build Configuration ===")
message(STATUS "HIP-only mode: ${HECBENCH_ENABLE_HIP_ONLY}")
message(STATUS "CUDA: ${HECBENCH_ENABLE_CUDA}")
if(HECBENCH_ENABLE_CUDA)
    message(STATUS "  Architecture: ${HECBENCH_CUDA_ARCH}")
endif()
message(STATUS "HIP: ${HECBENCH_ENABLE_HIP}")
if(HECBENCH_ENABLE_HIP)
    message(STATUS "  HECBENCH_HIP_ARCH: ${HECBENCH_HIP_ARCH} (CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES})")
    message(STATUS "  SPIR-V: ${HECBENCH_ENABLE_SPIRV}")
    if(HECBENCH_ENABLE_SPIRV)
        message(STATUS "  SPIR-V target: ${HECBENCH_SPIRV_TARGET}")
    endif()
    message(STATUS "  Extra CFLAGS: ${HECBENCH_HIP_EXTRA_CFLAGS}")
    message(STATUS "  Extra LDFLAGS: ${HECBENCH_HIP_LDFLAGS}")
endif()
message(STATUS "SYCL: ${HECBENCH_ENABLE_SYCL}")
message(STATUS "OpenMP: ${HECBENCH_ENABLE_OPENMP}")
message(STATUS "Testing: ${HECBENCH_ENABLE_TESTING}")
message(STATUS "====================================")
message(STATUS "")
