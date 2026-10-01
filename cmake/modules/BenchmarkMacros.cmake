# BenchmarkMacros.cmake
# Macros and functions for registering HeCBench benchmarks

# Global list to track all registered benchmarks
set_property(GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS "")
set_property(GLOBAL PROPERTY HECBENCH_CATEGORIES "")

# Global list for benchmarks that require Boost
set(DEPEND_ON_BOOST "hbc" "ge-spmm" "mmcsf" "warpsort" "gerbil")

# Global list for benchmarks that require MPI
set(DEPEND_ON_MPI "miniDGS" "miniWeather" "pingpong" "sparkler" "allreduce" "ccl" "halo-finder")

# Global list for benchmarks that require Eigen
set(DEPEND_ON_EIGEN "xlqc")

# Global list for benchmarks that require GDAL
set(DEPEND_ON_GDAL "stsg")

# Global list for benchmarks that require NCCL
set(DEPEND_ON_NCCL "ccl-cuda")

# Global list for benchmarks that require BZip2
set(DEPEND_ON_BZIP2 "gerbil")

# Global list for SYCL benchmarks that require C++20
set(DEPEND_ON_CXX20 "adamw-sycl" "zmddft-sycl")

# MKL GEMM
set(DEPEND_ON_MKLOMPGEMM "blas-gemm-omp")

# Path to test runner script
set(HECBENCH_TEST_RUNNER "${CMAKE_SOURCE_DIR}/cmake/scripts/run_benchmark_test.py")

# add_hecbench_benchmark
#
# Register a HeCBench benchmark
#
# Arguments:
#   NAME        - Benchmark name (e.g., "jacobi")
#   MODEL       - Programming model: cuda, hip, sycl, omp
#   SOURCES     - Source files
#   CATEGORIES  - Categories (e.g., simulation, math)
#   COMPILE_OPTIONS - Additional compile options (optional)
#   LINK_LIBRARIES  - Additional libraries to link (optional)
#   INCLUDE_DIRS  - Additional include path required by the benchmark (optional)
#   TEST_REGEX  - Regex pattern to match output for test verification (optional)
#   TEST_ARGS   - Arguments to pass when running tests (optional)
#   TEST_TIMEOUT - Timeout in seconds for test execution (optional, default 300)
#
# Example:
#   add_hecbench_benchmark(
#       NAME jacobi
#       MODEL cuda
#       SOURCES main.cu
#       CATEGORIES simulation math
#   )
#
function(add_hecbench_benchmark)
    # Parse arguments
    set(options "")
    set(oneValueArgs NAME MODEL TEST_REGEX TEST_TIMEOUT ENABLE_FASTMATH)
    set(multiValueArgs SOURCES CATEGORIES COMPILE_OPTIONS LINK_LIBRARIES INCLUDE_DIRS TEST_ARGS)
    cmake_parse_arguments(BENCH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT BENCH_NAME)
        message(FATAL_ERROR "add_hecbench_benchmark: NAME is required")
    endif()
    if(NOT BENCH_MODEL)
        message(FATAL_ERROR "add_hecbench_benchmark: MODEL is required")
    endif()
    if(NOT BENCH_SOURCES)
        message(FATAL_ERROR "add_hecbench_benchmark: SOURCES is required")
    endif()

    # Normalize model name
    string(TOLOWER "${BENCH_MODEL}" BENCH_MODEL_LOWER)

    # Check if this model is enabled
    set(MODEL_ENABLED FALSE)
    if(BENCH_MODEL_LOWER STREQUAL "cuda" AND HECBENCH_ENABLE_CUDA)
        set(MODEL_ENABLED TRUE)
    elseif(BENCH_MODEL_LOWER STREQUAL "hip" AND HECBENCH_ENABLE_HIP)
        set(MODEL_ENABLED TRUE)
    elseif(BENCH_MODEL_LOWER STREQUAL "sycl" AND HECBENCH_ENABLE_SYCL)
        set(MODEL_ENABLED TRUE)
    elseif((BENCH_MODEL_LOWER STREQUAL "omp" OR BENCH_MODEL_LOWER STREQUAL "openmp") AND HECBENCH_ENABLE_OPENMP)
        set(MODEL_ENABLED TRUE)
    endif()

    if(NOT MODEL_ENABLED)
        message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (model not enabled)")
        return()
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_BOOST)
        if(Boost_FOUND)
            message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
            include_directories(${Boost_INCLUDE_DIRS})
        else()
            message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (Boost not found)")
            return()
        endif()
    endif()

    if("${BENCH_NAME}-${BENCH_MODEL_LOWER}" IN_LIST DEPEND_ON_MKLOMPGEMM)
        message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER}")
        return()
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_MPI)
        if(NOT MPI_FOUND)
            message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (MPI not found)")
            return()
        endif()
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_EIGEN)
        # Eigen is header-only; the imported target carries the include path
        if(NOT TARGET Eigen3::Eigen)
            message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (Eigen not found)")
            return()
        endif()
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_GDAL)
        if(NOT GDAL_FOUND)
            message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (GDAL not found)")
            return()
        endif()
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_BZIP2)
        if(BZip2_FOUND)
            message(STATUS "BZip2 found: ${BZIP2_INCLUDE_DIRS}")
            include_directories(${BZIP2_INCLUDE_DIRS})
        else()
            message(STATUS "Skipping ${BENCH_NAME}-${BENCH_MODEL_LOWER} (BZip2 not found)")
            return()
        endif()
    endif()

    # Create target name
    set(TARGET_NAME "${BENCH_NAME}-${BENCH_MODEL_LOWER}")

    if(${TARGET_NAME} IN_LIST DEPEND_ON_NCCL)
        if(NOT NCCL_FOUND)
            message(STATUS "Skipping ${TARGET_NAME} (NCCL not found)")
            return()
        endif()
    endif()

    # For HIP, ensure .cu files are treated as HIP language
    if(BENCH_MODEL_LOWER STREQUAL "hip")
        set_source_files_properties(${BENCH_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()

    # Add executable
    add_executable(${TARGET_NAME} ${BENCH_SOURCES})

    # Set target properties
    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${BENCH_NAME}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${BENCH_MODEL_LOWER}"
    )

    # Add include directories
    if(BENCH_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} PRIVATE ${BENCH_INCLUDE_DIRS})
    endif()

    # Add common include directory
    #target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src/include")

    # Add benchmark's own directory for local headers (reference.h, kernels.h, etc.)
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    # For non-CUDA models, also check the CUDA variant directory for shared headers
    # Many benchmarks share reference.h, kernels.h, etc. between CUDA and other models
    if(NOT BENCH_MODEL_LOWER STREQUAL "cuda")
        set(CUDA_VARIANT_DIR "${CMAKE_SOURCE_DIR}/src/${BENCH_NAME}-cuda")
        if(EXISTS "${CUDA_VARIANT_DIR}")
            target_include_directories(${TARGET_NAME} PRIVATE "${CUDA_VARIANT_DIR}")
        endif()
    endif()

    # Model-specific configuration
    if(BENCH_MODEL_LOWER STREQUAL "cuda")
        # CUDA configuration
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${HECBENCH_CUDA_ARCH}
        )
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)

	if(${TARGET_NAME} IN_LIST DEPEND_ON_NCCL)
            target_include_directories(${TARGET_NAME} PRIVATE ${NCCL_INCLUDE_DIRS})
            target_link_libraries(${TARGET_NAME} PRIVATE ${NCCL_LIBRARIES})
        endif()

    elseif(BENCH_MODEL_LOWER STREQUAL "hip")
        # HIP configuration
        set_target_properties(${TARGET_NAME} PROPERTIES
            HIP_STANDARD 17
            HIP_ARCHITECTURES ${HECBENCH_HIP_ARCH}
        )
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${HECBENCH_HIP_ARCH}>
        )

    elseif(BENCH_MODEL_LOWER STREQUAL "sycl")
        # SYCL configuration
        target_compile_options(${TARGET_NAME} PRIVATE
            -fsycl
            ${SYCL_FLAGS}
        )
        target_link_options(${TARGET_NAME} PRIVATE
            -fsycl
            ${SYCL_FLAGS}
        )
        # Override the default CXX standard
        if(${TARGET_NAME} IN_LIST DEPEND_ON_CXX20)
            set_target_properties(${TARGET_NAME} PROPERTIES
                                  CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
        endif()

    elseif(BENCH_MODEL_LOWER STREQUAL "omp" OR BENCH_MODEL_LOWER STREQUAL "openmp")
        # OpenMP configuration
        target_compile_options(${TARGET_NAME} PRIVATE 
                               ${HECBENCH_OMP_TARGET}
                               ${HECBENCH_OMP_TARGET_BACKEND})
        if(BENCH_ENABLE_FASTMATH)
            target_compile_options(${TARGET_NAME} PRIVATE ${HECBENCH_OMP_FASTMATH})
        endif()
        target_link_libraries(${TARGET_NAME} PRIVATE
                              ${HECBENCH_OMP_TARGET}
                              ${HECBENCH_OMP_TARGET_BACKEND})
        #target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(${BENCH_NAME} IN_LIST DEPEND_ON_MPI)
        # automatically handles the necessary include paths, compiler flags, and libraries
        target_link_libraries(${TARGET_NAME} PRIVATE MPI::MPI_CXX)
    endif()

    # Add user-specified compile options
    if(BENCH_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME} PRIVATE ${BENCH_COMPILE_OPTIONS})
    endif()

    # Add user-specified link libraries
    if(BENCH_LINK_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${BENCH_LINK_LIBRARIES})
    endif()

    # Register benchmark in global list
    get_property(ALL_BENCHMARKS GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS)
    list(APPEND ALL_BENCHMARKS ${TARGET_NAME})
    set_property(GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS "${ALL_BENCHMARKS}")

    # Register categories
    foreach(CATEGORY ${BENCH_CATEGORIES})
        string(TOLOWER "${CATEGORY}" CATEGORY_LOWER)
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY_LOWER})
        list(APPEND CAT_BENCHMARKS ${TARGET_NAME})
        set_property(GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY_LOWER} "${CAT_BENCHMARKS}")

        # Track unique categories
        get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)
        if(NOT CATEGORY_LOWER IN_LIST ALL_CATEGORIES)
            list(APPEND ALL_CATEGORIES ${CATEGORY_LOWER})
            set_property(GLOBAL PROPERTY HECBENCH_CATEGORIES "${ALL_CATEGORIES}")
        endif()
    endforeach()

    # Create an alias target for just this benchmark (all models)
    if(NOT TARGET ${BENCH_NAME}-all)
        add_custom_target(${BENCH_NAME}-all)
    endif()
    add_dependencies(${BENCH_NAME}-all ${TARGET_NAME})

    # Register CTest test if testing is enabled and we have a regex pattern
    if(HECBENCH_ENABLE_TESTING AND BENCH_TEST_REGEX)
        # Set default timeout
        if(NOT BENCH_TEST_TIMEOUT)
            set(BENCH_TEST_TIMEOUT 300)
        endif()

        # Build the test command
        set(TEST_CMD
            ${Python3_EXECUTABLE}
            ${HECBENCH_TEST_RUNNER}
            $<TARGET_FILE:${TARGET_NAME}>
        )

        # Add test arguments if specified
        if(BENCH_TEST_ARGS)
            list(APPEND TEST_CMD ${BENCH_TEST_ARGS})
        endif()

        # Add the regex pattern
        list(APPEND TEST_CMD --regex "${BENCH_TEST_REGEX}")
        list(APPEND TEST_CMD --timeout ${BENCH_TEST_TIMEOUT})
        list(APPEND TEST_CMD --working-dir "${CMAKE_CURRENT_SOURCE_DIR}")

        # Register the test
        add_test(
            NAME ${TARGET_NAME}
            COMMAND ${TEST_CMD}
        )

        # Set test properties
        set_tests_properties(${TARGET_NAME} PROPERTIES
            TIMEOUT ${BENCH_TEST_TIMEOUT}
            LABELS "${BENCH_MODEL_LOWER};${BENCH_CATEGORIES}"
        )

        message(STATUS "  -> Registered test: ${TARGET_NAME}")
    endif()

    message(STATUS "Registered benchmark: ${TARGET_NAME}")
endfunction()

# create_category_targets
#
# Create convenience targets for building benchmarks by category
# Should be called after all benchmarks are registered
#
function(create_category_targets)
    get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)

    foreach(CATEGORY ${ALL_CATEGORIES})
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY})

        if(CAT_BENCHMARKS)
            # Create target for this category
            add_custom_target(category-${CATEGORY})
            add_dependencies(category-${CATEGORY} ${CAT_BENCHMARKS})
            list(LENGTH CAT_BENCHMARKS CAT_COUNT)
            message(STATUS "Created category target: category-${CATEGORY} (${CAT_COUNT} benchmarks)")
        endif()
    endforeach()
endfunction()

# print_benchmark_summary
#
# Print summary of registered benchmarks
#
function(print_benchmark_summary)
    get_property(ALL_BENCHMARKS GLOBAL PROPERTY HECBENCH_ALL_BENCHMARKS)
    list(LENGTH ALL_BENCHMARKS BENCHMARK_COUNT)

    message(STATUS "")
    message(STATUS "=== Benchmark Registration Summary ===")
    message(STATUS "Total benchmarks registered: ${BENCHMARK_COUNT}")

    get_property(ALL_CATEGORIES GLOBAL PROPERTY HECBENCH_CATEGORIES)
    foreach(CATEGORY ${ALL_CATEGORIES})
        get_property(CAT_BENCHMARKS GLOBAL PROPERTY HECBENCH_CATEGORY_${CATEGORY})
        list(LENGTH CAT_BENCHMARKS CAT_COUNT)
        message(STATUS "  ${CATEGORY}: ${CAT_COUNT}")
    endforeach()

    message(STATUS "======================================")
    message(STATUS "")
endfunction()

# Unzip
#
# Unzip data needed for certain benchmarks
#
MACRO(UNZIPFILE zipfile)
    # get the filetype extension
    GET_FILENAME_COMPONENT(FILEEXT "${zipfile}" EXT)
    GET_FILENAME_COMPONENT(FILEDIR "${zipfile}" DIRECTORY)

    MESSAGE("Going to unzip file ${zipfile}!")
    MESSAGE("File extension: ${FILEEXT}")
    MESSAGE("File dir: ${FILEDIR}")
    IF(FILEEXT MATCHES "\.tar.bz")
        set(TOEXEC "tar -xjf ${zipfile} -C ${FILEDIR}")
    ELSEIF(FILEEXT MATCHES "\.bz2")
        set(TOEXEC "bzip2 -dkf ${zipfile}")
    ELSEIF(FILEEXT MATCHES "\.tar")
        set(TOEXEC "tar -xvf ${zipfile} -C ${FILEDIR}")
    ELSEIF(FILEEXT MATCHES "\.zip")
        set(TOEXEC "unzip ${zipfile}")
    ENDIF()
    MESSAGE("\tAssociated command to unzip: [${TOEXEC}]")
    execute_process(
        COMMAND /bin/bash -c "${TOEXEC}"
        OUTPUT_VARIABLE EXEC_OUTPUT
        RESULT_VARIABLE RESULT_CODE
        WORKING_DIRECTORY ${FILEDIR}
    )
    # MESSAGE("Execution Result: \n${EXEC_OUTPUT}")
    # if the result code was not 0
    IF (NOT(RESULT_CODE EQUAL 0))
        message("Problem unzipping file ${zipfile}!!") 
    ENDIF()
ENDMACRO()
