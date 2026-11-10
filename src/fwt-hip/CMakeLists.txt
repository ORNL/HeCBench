# fwt-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../fwt-cuda")

add_hecbench_benchmark(
    NAME fwt
    MODEL hip
    SOURCES main.cu ${INC_DIR}/reference.cu
    CATEGORIES math
)
