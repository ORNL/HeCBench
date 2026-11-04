# meanshift-hip/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/../meanshift-cuda/dataset.tar.gz")

add_hecbench_benchmark(
    NAME meanshift
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
