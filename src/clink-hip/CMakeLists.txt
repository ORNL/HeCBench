# clink-hip/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME clink
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
