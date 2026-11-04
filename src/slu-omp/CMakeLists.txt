# slu-omp/CMakeLists.txt

set(SRC "${CMAKE_CURRENT_LIST_DIR}/../slu-cuda")

FILE(GLOB NICSLU_LIB "${SRC}/src/nicslu/source/*.c")
add_library(nicslu_omp STATIC ${NICSLU_LIB})

FILE(GLOB NICSLU_UTIL "${SRC}/src/nicslu/util/nicslu_readfile.c")
add_library(nicslu_util_omp STATIC ${NICSLU_UTIL})
target_include_directories(nicslu_util_omp AFTER PRIVATE "${SRC}/src/nicslu/source")

add_hecbench_benchmark(
    NAME slu
    MODEL omp
    SOURCES ${SRC}/src/main.cpp
            ${SRC}/src/preprocess.cpp
            ${SRC}/src/symbolic.cpp
            ${SRC}/src/Timer.cpp
            numeric.cpp
    INCLUDE_DIRS ${SRC}/include ${SRC}/src/nicslu/include ${SRC}/src/nicslu/util
    COMPILE_OPTIONS -DVERIFY
    CATEGORIES math
    LINK_LIBRARIES nicslu_omp nicslu_util_omp
)
