# rmsnorm-quant-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm-cuda")

add_hecbench_benchmark(
    NAME rmsnorm-quant
    MODEL cuda
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
)
