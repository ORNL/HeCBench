# layernorm-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm-hip")

add_hecbench_benchmark(
    NAME layernorm
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
)
