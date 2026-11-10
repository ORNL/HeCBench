# sobel-hip/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME sobel
    MODEL hip
    SOURCES main.cu reference.cu
    INCLUDE_DIRS ${SDK_DIR}
    CATEGORIES imaging
)
