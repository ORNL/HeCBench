# scan3-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../scan2-cuda")

add_hecbench_benchmark(
    NAME scan3
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
    INCLUDE_DIRS ${INC_DIR}
)
