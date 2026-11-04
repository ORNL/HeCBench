# kmeans-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/kmeans/kdd_cup.tar.bz")

add_hecbench_benchmark(
    NAME kmeans
    MODEL cuda
    SOURCES cluster.cu getopt.cpp kmeans.cpp read_input.cpp rmse.cpp
    CATEGORIES ml
)
