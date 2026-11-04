# sa-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/genome.tar.bz")

add_hecbench_benchmark(
    NAME sa
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
