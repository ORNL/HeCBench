# blockexchange-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blockAccess-cuda")
set(INC_DIR2 "${CMAKE_CURRENT_LIST_DIR}/../warpexchange-cuda")

add_hecbench_benchmark(
    NAME blockexchange
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
    INCLUDE_DIRS ${INC_DIR} ${INC_DIR2}
)
