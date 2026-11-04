# permute-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../qkv-hip")

add_hecbench_benchmark(
    NAME permute
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
)
