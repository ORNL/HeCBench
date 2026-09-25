# FindHIP.cmake
# Find the HIP (Heterogeneous-compute Interface for Portability) installation
#
# This module defines:
#  HIP_FOUND        - True if HIP was found
#  HIP_COMPILER     - Path to hipcc compiler
#  HIP_VERSION      - HIP version
#  hip::host        - Imported target for HIP

if(HIP_FOUND)
    return()
endif()

# hip-lang sets HIP_COMPILER to the compiler ID, which hides a cached hipcc path.
unset(HIP_COMPILER)

# Find hipcc compiler
find_program(HIP_COMPILER
    NAMES hipcc
    PATHS
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
    PATH_SUFFIXES bin
    DOC "HIP compiler (hipcc)"
)

if(HIP_COMPILER)
    # Get HIP version
    execute_process(
        COMMAND ${HIP_COMPILER} --version
        OUTPUT_VARIABLE HIP_VERSION_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Parse version from output
    if(HIP_VERSION_OUTPUT MATCHES "HIP version: ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(HIP_VERSION "${CMAKE_MATCH_1}")
    endif()

    # Find HIP include directory
    get_filename_component(HIP_BIN_DIR "${HIP_COMPILER}" DIRECTORY)
    get_filename_component(HIP_ROOT_DIR "${HIP_BIN_DIR}" DIRECTORY)

    find_path(HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        PATHS
            ${HIP_ROOT_DIR}
            ENV ROCM_PATH
            ENV HIP_PATH
            /opt/rocm
            /opt/rocm/hip
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )

    # Find HIP library
    find_library(HIP_LIBRARY
        NAMES amdhip64
        PATHS
            ${HIP_ROOT_DIR}
            ENV ROCM_PATH
            ENV HIP_PATH
            /opt/rocm
            /opt/rocm/hip
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )

    # Create imported target
    if(HIP_INCLUDE_DIR AND HIP_LIBRARY)
        if(NOT TARGET hip::host)
            add_library(hip::host INTERFACE IMPORTED)
            set_target_properties(hip::host PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${HIP_LIBRARY}"
            )
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP
    REQUIRED_VARS HIP_COMPILER HIP_INCLUDE_DIR HIP_LIBRARY
    VERSION_VAR HIP_VERSION
)

mark_as_advanced(
    HIP_COMPILER
    HIP_INCLUDE_DIR
    HIP_LIBRARY
    HIP_VERSION
)
