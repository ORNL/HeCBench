# urng-hip/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME urng
    MODEL hip
    SOURCES main.cu
    CATEGORIES rng
    INCLUDE_DIRS ${SDK_DIR}
)
