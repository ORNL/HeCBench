# slu-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/src/nicslu.tar.bz2")

FILE(GLOB NICSLU_LIB "${CMAKE_CURRENT_LIST_DIR}/src/nicslu/source/*.c")
add_library(nicslu_cuda STATIC ${NICSLU_LIB})

FILE(GLOB NICSLU_UTIL "${CMAKE_CURRENT_LIST_DIR}/src/nicslu/util/nicslu_readfile.c")
add_library(nicslu_util_cuda STATIC ${NICSLU_UTIL})
target_include_directories(nicslu_util_cuda AFTER PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src/nicslu/source")

add_hecbench_benchmark(
    NAME slu
    MODEL cuda
    SOURCES src/main.cpp
            src/numeric.cu
            src/preprocess.cpp
            src/symbolic.cpp
            src/Timer.cpp
    INCLUDE_DIRS include src/nicslu/include src/nicslu/util
    COMPILE_OPTIONS -DVERIFY
    CATEGORIES math
    LINK_LIBRARIES nicslu_cuda nicslu_util_cuda
)
