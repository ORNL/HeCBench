# si-cuda/CMakeLists.txt

if(HECBENCH_ENABLE_CUDA)
    include(FetchContent)
    if(NOT TARGET cxxopts)
        FetchContent_Declare(
            cxxopts
            GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
            GIT_TAG        v3.3.1
        )
        FetchContent_MakeAvailable(cxxopts)
    endif()
    if(NOT TARGET fmt::fmt-header-only)
        set(FMT_DOC OFF CACHE BOOL "" FORCE)
        set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
        set(FMT_TEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG        7.1.2
        )
        FetchContent_MakeAvailable(fmt)
    endif()
endif()

set(CMAKE_CUDA_STANDARD 17)

add_hecbench_benchmark(
    NAME si
    MODEL cuda
    SOURCES src/main.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    CATEGORIES search
    LINK_LIBRARIES cxxopts fmt::fmt-header-only cublas
)
