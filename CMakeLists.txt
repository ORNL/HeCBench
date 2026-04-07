cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# Build options - which programming models to enable
option(HECBENCH_ENABLE_CUDA "Enable CUDA benchmarks" ON)
option(HECBENCH_ENABLE_HIP "Enable HIP benchmarks" ON)
option(HECBENCH_ENABLE_SYCL "Enable SYCL benchmarks" ON)
option(HECBENCH_ENABLE_OPENMP "Enable OpenMP benchmarks" ON)
# Architecture options
set(HECBENCH_CUDA_ARCH "sm_80" CACHE STRING "CUDA architecture (e.g., sm_60, sm_70, sm_80, sm_90)")
set(HECBENCH_HIP_ARCH "gfx90a" CACHE STRING "HIP architecture (e.g., gfx908, gfx90a, gfx942)")
option(HECBENCH_ENABLE_SPIRV "Enable SPIR-V mode for HIP benchmarks" OFF)
set(HECBENCH_SPIRV_TARGET "amdgcnspirv" CACHE STRING "SPIR-V offload target")
# Integration knobs (upstream-friendly, optional)
set(HECBENCH_HIP_EXTRA_CFLAGS "" CACHE STRING "Extra HIP/C++ compile flags")
set(HECBENCH_HIP_LDFLAGS "" CACHE STRING "Extra linker flags for HIP benchmarks")
option(HECBENCH_FIX_MAKEFILE_LINK_RULES
  "Patch Makefile link rules to exclude CFLAGS from link invocations"
  ON)
# Build configuration
option(HECBENCH_ENABLE_TESTING "Enable testing" ON)
option(HECBENCH_BUILD_ALL_BENCHMARKS "Build all benchmarks (vs selective)" ON)
set(HECBENCH_SKIP_BENCHMARKS "" CACHE STRING
  "Semicolon/comma/space separated benchmark dirs to skip (e.g. layernorm-hip;slu-hip). Synonyms: nicslu_cuda/slu-cuda, nicslu_util_omp/slu-omp")
set(HECBENCH_SKIP_BENCHMARKS_FILE "" CACHE FILEPATH
  "Optional extra skip list: one benchmark id per line (# comments allowed). Merged after HECBENCH_SKIP_BENCHMARKS (for integrators like TheRock).")
option(HECBENCH_ENABLE_HIP_ONLY
  "Build ONLY HIP benchmarks (disables CUDA, SYCL, OpenMP)"
  OFF)

# Apply HIP-only mode
if(HECBENCH_ENABLE_HIP_ONLY)
  set(HECBENCH_ENABLE_CUDA OFF)
  set(HECBENCH_ENABLE_SYCL OFF)
  set(HECBENCH_ENABLE_OPENMP OFF)
  set(HECBENCH_ENABLE_HIP ON)
endif()

# Configure core languages first; optional languages are enabled after detection.
project(HeCBench
    LANGUAGES C CXX
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
# Find programming model compilers/libraries
if(HECBENCH_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(NCCL MODULE)
    set(_hecbench_cuda_arch "${HECBENCH_CUDA_ARCH}")
    string(REGEX REPLACE "^sm_" "" _hecbench_cuda_arch "${_hecbench_cuda_arch}")
    set(CMAKE_CUDA_ARCHITECTURES "${_hecbench_cuda_arch}")
    message(STATUS "CUDA enabled - Architecture: ${HECBENCH_CUDA_ARCH} (CMake: ${CMAKE_CUDA_ARCHITECTURES})")
endif()
if(HECBENCH_ENABLE_HIP)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(CMAKE_HIP_ARCHITECTURES "${HECBENCH_HIP_ARCH}")
        find_package(HIP MODULE QUIET)
        message(STATUS "HIP enabled - Architecture: ${HECBENCH_HIP_ARCH} (CMake: ${CMAKE_HIP_ARCHITECTURES})")
    else()
        message(WARNING "HIP requested but no HIP compiler was detected - HIP benchmarks will be skipped")
        set(HECBENCH_ENABLE_HIP OFF)
    endif()
endif()
if(HECBENCH_ENABLE_SYCL)
    find_package(SYCL MODULE)
    if(SYCL_FOUND)
        message(STATUS "SYCL enabled - Compiler: ${CMAKE_CXX_COMPILER}")
    else()
        message(WARNING "SYCL requested but not found - SYCL benchmarks will be skipped")
        set(HECBENCH_ENABLE_SYCL OFF)
    endif()
endif()
if(HECBENCH_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP requested but not found - OpenMP benchmarks will be skipped")
        set(HECBENCH_ENABLE_OPENMP OFF)
    endif()
endif()
# Apply optional integration flags for HIP flow.
if(HECBENCH_ENABLE_HIP)
    if(HECBENCH_ENABLE_SPIRV)
        if(NOT HECBENCH_HIP_EXTRA_CFLAGS)
            set(HECBENCH_HIP_EXTRA_CFLAGS "-x hip --offload-arch=${HECBENCH_SPIRV_TARGET} -std=c++17 -Wall")
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
# Include benchmark macros
include(BenchmarkMacros)

# ---------------------------------------------------------------------------
# Utility: Patch Makefiles to remove CFLAGS from link rules
# Needed when building with HIP flags like -x hip that should not reach linker
# ---------------------------------------------------------------------------
function(hecbench_patch_makefile_link_rules _target_dir)
  file(GLOB _makefile "${_target_dir}/Makefile")
  if(NOT _makefile)
    return()
  endif()
  file(READ "${_makefile}" _makefile_content)
  # Recipe lines are tab-indented; strip $(CFLAGS) only from the link rule so
  # flags like -x hip are not passed to the linker (compile-only).
  string(REGEX REPLACE
    "([\t ]*)(\\$\\(CC\\))[ \t]+\\$\\(CFLAGS\\)([^\r\n]*-o[ \t]+\\$@[^\r\n]*\\$\\(LDFLAGS\\))"
    "\\1\\2\\3"
    _patched_content
    "${_makefile_content}"
  )
  if(NOT _patched_content STREQUAL _makefile_content)
    file(WRITE "${_makefile}" "${_patched_content}")
    message(STATUS "Patched Makefile link rules in: ${_target_dir}")
  endif()
endfunction()

if(HECBENCH_FIX_MAKEFILE_LINK_RULES)
  file(GLOB_RECURSE _hecbench_benchmark_makefiles "${CMAKE_CURRENT_SOURCE_DIR}/src/*/Makefile")
  foreach(_makefile IN LISTS _hecbench_benchmark_makefiles)
    get_filename_component(_makefile_dir "${_makefile}" DIRECTORY)
    hecbench_patch_makefile_link_rules("${_makefile_dir}")
  endforeach()
endif()

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
message(STATUS "HIP-only mode: ${HECBENCH_ENABLE_HIP_ONLY}")
message(STATUS "CUDA:    ${HECBENCH_ENABLE_CUDA}")
if(HECBENCH_ENABLE_CUDA)
    message(STATUS "  Architecture: ${HECBENCH_CUDA_ARCH} (CMake: ${CMAKE_CUDA_ARCHITECTURES})")
endif()
message(STATUS "HIP:     ${HECBENCH_ENABLE_HIP}")
if(HECBENCH_ENABLE_HIP)
    message(STATUS "  Architecture: ${HECBENCH_HIP_ARCH} (CMake: ${CMAKE_HIP_ARCHITECTURES})")
    message(STATUS "  SPIR-V: ${HECBENCH_ENABLE_SPIRV}")
    if(HECBENCH_ENABLE_SPIRV)
        message(STATUS "  SPIR-V target: ${HECBENCH_SPIRV_TARGET}")
    endif()
    message(STATUS "  Extra CFLAGS: ${HECBENCH_HIP_EXTRA_CFLAGS}")
    message(STATUS "  Extra LDFLAGS: ${HECBENCH_HIP_LDFLAGS}")
endif()
message(STATUS "SYCL:    ${HECBENCH_ENABLE_SYCL}")
message(STATUS "OpenMP:  ${HECBENCH_ENABLE_OPENMP}")
message(STATUS "Testing: ${HECBENCH_ENABLE_TESTING}")
message(STATUS "====================================")
message(STATUS "")