# gibbs-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME gibbs
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
