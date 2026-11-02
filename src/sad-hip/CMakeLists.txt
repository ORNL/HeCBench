# sad-hip/CMakeLists.txt

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda")
UNZIPFILE("${DATA_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME sad
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    INCLUDE_DIRS ${DATA_DIR}
)
