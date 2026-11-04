# meanshift-sycl/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/../meanshift-cuda/dataset.tar.gz")

add_hecbench_benchmark(
    NAME meanshift
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -ffast-math
)
