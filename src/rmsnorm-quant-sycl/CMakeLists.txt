# rmsnorm-quant-sycl/CMakeLists.txt

set(INC_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm-quant-cuda"
    "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm-sycl")

add_hecbench_benchmark(
    NAME rmsnorm-quant
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
)
