# permute-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../qkv-cuda")

add_hecbench_benchmark(
    NAME permute
    MODEL cuda
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
)
