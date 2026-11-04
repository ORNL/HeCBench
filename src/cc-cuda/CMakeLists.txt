# cc-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME cc
    MODEL cuda
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES graph
)
