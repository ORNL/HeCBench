# mcpr-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/alphas.csv.bz2")

add_hecbench_benchmark(
    NAME mcpr
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
