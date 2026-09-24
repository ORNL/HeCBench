# sad-cuda/CMakeLists.txt

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${DATA_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME sad
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
