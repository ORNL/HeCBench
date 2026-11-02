# FindSYCL.cmake
# Find SYCL compiler and libraries
#
# This module defines:
#  SYCL_FOUND       - True if SYCL was found
#  SYCL_COMPILER    - Path to SYCL-capable compiler
#  SYCL_VERSION     - SYCL compiler version
#  SYCL_FLAGS       - Compile flags for SYCL (CMake list of separate args)
#  SYCL_LINK_FLAGS  - Link flags for SYCL (no preprocessor -D flags)

# SYCL can be provided by multiple compilers:
# - Intel DPC++ (icpx, clang++)
# - hipSYCL (syclcc, clang++)
# - ComputeCpp (compute++)

# Skip compiler rediscovery on reconfigure, but always recompose SYCL_FLAGS
# below so HECBENCH_SYCL_TARGET* cache changes take effect.
set(_SYCL_NEED_DISCOVERY TRUE)
if(SYCL_FOUND AND SYCL_COMPILER)
    set(_SYCL_NEED_DISCOVERY FALSE)
endif()

if(_SYCL_NEED_DISCOVERY)
    find_program(SYCL_COMPILER
        NAMES icpx clang++
        PATHS
            ENV ONEAPI_ROOT
            ENV DPCPP_HOME
            /opt/intel/oneapi/compiler/latest/linux
            /opt/intel/oneapi/compiler/latest
        PATH_SUFFIXES bin
        DOC "SYCL compiler"
    )

    if(SYCL_COMPILER)
        # Check if compiler supports SYCL
        execute_process(
            COMMAND ${SYCL_COMPILER} -fsycl --version
            OUTPUT_VARIABLE SYCL_VERSION_OUTPUT
            ERROR_VARIABLE SYCL_VERSION_ERROR
            RESULT_VARIABLE SYCL_VERSION_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(SYCL_VERSION_RESULT EQUAL 0)
            # Parse version
            if(SYCL_VERSION_OUTPUT MATCHES "DPC\\+\\+/C\\+\\+ Compiler ([0-9]+\\.[0-9]+\\.[0-9]+)")
                set(SYCL_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "SYCL compiler version")
                set(SYCL_COMPILER_TYPE "Intel DPC++" CACHE INTERNAL "SYCL compiler family")
            elseif(SYCL_VERSION_OUTPUT MATCHES "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)")
                set(SYCL_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "SYCL compiler version")
                set(SYCL_COMPILER_TYPE "Clang with SYCL" CACHE INTERNAL "SYCL compiler family")
            endif()

            # Detect backend support (reset first so a compiler change cannot
            # leave a stale TRUE from a previous configure)
            set(SYCL_SUPPORTS_CUDA FALSE CACHE INTERNAL "SYCL CUDA backend supported")
            set(SYCL_SUPPORTS_HIP FALSE CACHE INTERNAL "SYCL HIP backend supported")
            execute_process(
                COMMAND ${SYCL_COMPILER} -fsycl -fsycl-targets=nvptx64-nvidia-cuda --version
                RESULT_VARIABLE SYCL_CUDA_RESULT
                OUTPUT_QUIET ERROR_QUIET
            )
            if(SYCL_CUDA_RESULT EQUAL 0)
                set(SYCL_SUPPORTS_CUDA TRUE CACHE INTERNAL "SYCL CUDA backend supported")
            endif()

            execute_process(
                COMMAND ${SYCL_COMPILER} -fsycl -fsycl-targets=amdgcn-amd-amdhsa --version
                RESULT_VARIABLE SYCL_HIP_RESULT
                OUTPUT_QUIET ERROR_QUIET
            )
            if(SYCL_HIP_RESULT EQUAL 0)
                set(SYCL_SUPPORTS_HIP TRUE CACHE INTERNAL "SYCL HIP backend supported")
            endif()

        else()
            # Try hipSYCL
            find_program(HIPSYCL_COMPILER
                NAMES syclcc
                PATHS
                    ENV HIPSYCL_HOME
                PATH_SUFFIXES bin
            )

            if(HIPSYCL_COMPILER)
                set(SYCL_COMPILER ${HIPSYCL_COMPILER} CACHE FILEPATH "SYCL compiler" FORCE)
                set(SYCL_COMPILER_TYPE "hipSYCL" CACHE INTERNAL "SYCL compiler family")

                execute_process(
                    COMMAND ${SYCL_COMPILER} --version
                    OUTPUT_VARIABLE SYCL_VERSION_OUTPUT
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )

                if(SYCL_VERSION_OUTPUT MATCHES "hipSYCL version ([0-9]+\\.[0-9]+\\.[0-9]+)")
                    set(SYCL_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "SYCL compiler version")
                endif()
            endif()
        endif()
    endif()
endif()

# Compose flag lists as CMake lists of separate arguments every configure.
# Do not glue several compiler switches into one quoted string: CMake would
# pass them as a single argv and the device image would stay spir64.
# unset() rather than set("") so list(APPEND) cannot keep an empty first item.
unset(SYCL_FLAGS)
unset(SYCL_LINK_FLAGS)
if(SYCL_COMPILER AND NOT SYCL_COMPILER MATCHES "syclcc")
    list(APPEND SYCL_FLAGS -fsycl)
    list(APPEND SYCL_LINK_FLAGS -fsycl)
endif()
if(HECBENCH_SYCL_TARGET)
    list(APPEND SYCL_FLAGS -DUSE_GPU)
    list(APPEND SYCL_FLAGS "-fsycl-targets=${HECBENCH_SYCL_TARGET}")
    list(APPEND SYCL_LINK_FLAGS "-fsycl-targets=${HECBENCH_SYCL_TARGET}")
endif()
if(HECBENCH_SYCL_TARGET_BACKEND)
    list(APPEND SYCL_FLAGS -Xsycl-target-backend "${HECBENCH_SYCL_TARGET_BACKEND}")
    list(APPEND SYCL_LINK_FLAGS -Xsycl-target-backend "${HECBENCH_SYCL_TARGET_BACKEND}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SYCL
    REQUIRED_VARS SYCL_COMPILER
    VERSION_VAR SYCL_VERSION
)

if(SYCL_FOUND)
    message(STATUS "SYCL compiler type: ${SYCL_COMPILER_TYPE}")
    if(SYCL_SUPPORTS_CUDA)
        message(STATUS "  CUDA backend: supported")
    endif()
    if(SYCL_SUPPORTS_HIP)
        message(STATUS "  HIP backend: supported")
    endif()
endif()

mark_as_advanced(
    SYCL_COMPILER
    SYCL_VERSION
    SYCL_COMPILER_TYPE
)
